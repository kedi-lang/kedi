from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Mapping
from contextlib import (
    AbstractContextManager,
    AsyncExitStack,
    contextmanager,
    nullcontext,
)
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Any, Callable, Iterator, Sequence, cast

import dspy

from ...agent_profile import AgentProfile
from ...agent_profile.settings import DSPY_LM_SETTING_KEYS, dspy_lm_settings
from ...dynamic_models import build_dynamic_model
from ...model_normalization import normalize_for_litellm
from ...resolved_types import ResolvedKediType
from ...errors import KediExecutionError
from ..subagents.attempts import DelegationFailurePolicy, delegation_scope
from ..subagents.limits import SubagentUsageLimits
from ..subagents.runs import safe_subagent_error_message
from ..approval import ApprovalPolicy
from ..invocation import InvocationUsage, invocation_observer, report_invocation
from ..llm_wrapper import ThinkingStrippingLM
from ..meta import AdapterCapabilities, AgentAdapterKind
from ..subagents import SubagentExecutionRequest
from ..tool_override import ToolScope, ToolSpec, with_approval


class DSPyAdapter:
    kind: AgentAdapterKind = "agent-framework"
    shortname = "dspy"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_subagents=True,
        supports_background_subagents=False,
        settings=DSPY_LM_SETTING_KEYS,
    )

    # Class-level defaults so the LM config is readable via native attribute access
    # even on adapters built without __init__ (e.g. test doubles).
    _lm_provider: dspy.Provider | None = None
    _lm_max_tokens: int = 4000
    _lm_temperature: float = 1.0

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        temperature: float = 1.0,
        provider: dspy.Provider | None = None,
        model_settings: dict[str, Any] | None = None,
        subagent_failure_policy: DelegationFailurePolicy = "fail_closed",
        **kwargs,
    ):
        if subagent_failure_policy not in {"recover", "fail_closed"}:
            raise ValueError(f"Unknown subagent failure policy: {subagent_failure_policy!r}")
        model = normalize_for_litellm(model)
        self._lm_model = model
        self._lm_provider = provider
        self._lm_max_tokens = max_tokens
        self._lm_temperature = temperature
        self._lm_settings = dspy_lm_settings(model_settings or {})
        self._lm_extra_kwargs = dict(kwargs)
        self._subagent_failure_policy = subagent_failure_policy
        self._tool_scope = ToolScope()
        self._tool_spec_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_dspy_profile_stack_{id(self)}",
            default=(),
        )
        # Keep a one-time baseline configure so plain (no-profile) produce has a
        # global LM; per-call correctness comes from the ``dspy.context`` wrap in
        # ``produce_sync`` / ``profile_override`` (innermost context wins).
        if getattr(dspy.settings, "lm", None) is None:
            dspy.settings.configure(lm=self._default_lm())

    def _default_lm(self) -> ThinkingStrippingLM:
        provider, kwargs = self._lm_kwargs()
        return ThinkingStrippingLM(model=self._lm_model, provider=provider, **kwargs)

    def _lm_kwargs(
        self,
        profile_settings: dict[str, Any] | None = None,
        *,
        effort: str | None = None,
    ) -> tuple[dspy.Provider | None, dict[str, Any]]:
        kwargs = dict(getattr(self, "_lm_extra_kwargs", {}))
        kwargs.update(getattr(self, "_lm_settings", {}))
        kwargs.update(dspy_lm_settings(profile_settings or {}))
        kwargs.setdefault("max_tokens", getattr(self, "_lm_max_tokens", 4000))
        kwargs.setdefault("temperature", getattr(self, "_lm_temperature", 1.0))
        kwargs.setdefault("cache", False)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        provider = cast(dspy.Provider | None, kwargs.pop("provider", self._lm_provider))
        return provider, kwargs

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        @contextmanager
        def scoped() -> Iterator[None]:
            with (
                self._tool_scope.tool_scope(),
                self._tool_spec_scope_for_current_adapter().tool_scope(),
            ):
                for tool in tools:
                    if isinstance(tool, ToolSpec):
                        self.register_tool(tool)
                    else:
                        self._tool_scope.register_tool(tool)
                yield

        return scoped()

    def register_tool(self, tool: ToolSpec) -> None:
        from ..utils import dspy_sync_tool_function, dspy_tool_arg_desc, dspy_tool_args

        if tool.metadata is not None and tool.metadata.get("kind") == "subagent":
            tool = _dspy_recoverable_subagent_tool(tool)
        kwargs: dict[str, Any] = {
            "func": dspy_sync_tool_function(tool.fn),
            "name": tool.name,
            "desc": tool.description,
        }
        if tool.json_schema is not None:
            kwargs["args"] = dspy_tool_args(tool.json_schema)
            arg_desc = dspy_tool_arg_desc(tool.json_schema)
            if arg_desc:
                kwargs["arg_desc"] = arg_desc
        self._tool_scope.register_tool(dspy.Tool(**kwargs))
        self._tool_spec_scope_for_current_adapter().register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        scope = getattr(self, "_tool_scope", None)
        return () if scope is None else scope.scoped_tools

    @property
    def scoped_tool_specs(self) -> tuple[ToolSpec, ...]:
        return cast(
            tuple[ToolSpec, ...],
            self._tool_spec_scope_for_current_adapter().scoped_tools,
        )

    def _tool_spec_scope_for_current_adapter(self) -> ToolScope:
        scope = getattr(self, "_tool_spec_scope", None)
        if scope is None:
            scope = ToolScope()
            self._tool_spec_scope = scope
        return scope

    @property
    def active_profile(self) -> AgentProfile:
        stack_var = getattr(self, "_profile_stack_var", None)
        stack = stack_var.get() if stack_var is not None else ()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType], **kwargs) -> type[dspy.Signature]:
        """
        Type builder to create a dictionary schema from a given output type.
        Args:
            output_schema (dict): A dictionary defining the output schema.
            string (bool): Whether to use string signatures.
        Returns:
            dict: A dictionary schema based on the provided output type.
        """
        if not output_schema:
            raise ValueError("Output schema must be provided.")
        field_definitions: dict = {
            field: (type_, dspy.OutputField()) for field, type_ in output_schema.items()
        }
        field_definitions.update({"user_prompt": (str, dspy.InputField())})
        return build_dynamic_model("_OutputSignature", dspy.Signature, field_definitions)

    @staticmethod
    def invoke_signature() -> type[dspy.Signature]:
        return build_dynamic_model(
            "_InvokeSignature",
            dspy.Signature,
            {
                "user_prompt": (str, dspy.InputField()),
                "response": (str, dspy.OutputField()),
            },
        )

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs,
    ) -> dspy.Prediction:
        """
        Adapter to use DSPy agents.
        """
        return await asyncio.to_thread(
            self.produce_sync,
            template,
            output_schema,
            **kwargs,
        )

    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            self._produce_subagent_sync,
            template,
            output_schema,
            kwargs,
        )

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs,
    ) -> dspy.Prediction:
        """
        Synchronous version of produce method.
        """
        if not output_schema:
            raise ValueError("Output spec must be provided.")
        from ..utils import run_sync

        kwargs.pop("user_prompt", None)
        sig = self.type_builder(output_schema)
        instructions = kwargs.pop("instructions", None)
        sig = sig.with_instructions(instructions)
        tools = cast(list[Callable[..., Any]], list(self.scoped_tools))
        with delegation_scope(getattr(self, "_subagent_failure_policy", "fail_closed")):
            with self._produce_lm_context():
                if self.active_profile.mcp_servers:
                    prediction = run_sync(self._produce_with_mcp(sig, tools, template, kwargs))
                else:
                    adapter = dspy.ReAct(sig, tools=tools) if tools else dspy.Predict(sig)
                    prediction = adapter(**kwargs, user_prompt=template)
        _report_dspy_invocation(prediction)
        return prediction

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.invoke_sync, prompt=prompt, **kwargs)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        from ..utils import run_sync

        kwargs.pop("user_prompt", None)
        sig = self.invoke_signature()
        instructions = kwargs.pop("instructions", None)
        sig = sig.with_instructions(instructions)
        tools = cast(list[Callable[..., Any]], list(self.scoped_tools))
        with delegation_scope(getattr(self, "_subagent_failure_policy", "fail_closed")):
            with self._produce_lm_context():
                if self.active_profile.mcp_servers:
                    prediction = run_sync(self._produce_with_mcp(sig, tools, prompt, kwargs))
                else:
                    adapter = dspy.ReAct(sig, tools=tools) if tools else dspy.Predict(sig)
                    prediction = adapter(**kwargs, user_prompt=prompt)
        _report_dspy_invocation(prediction)
        return str(_prediction_value(prediction, "response"))

    def _produce_subagent_sync(
        self,
        template: str,
        output_schema: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Mapping[str, Any]:
        from ..utils import run_sync

        run_kwargs = dict(kwargs)
        run_kwargs.pop("user_prompt", None)
        instructions = run_kwargs.pop("instructions", None)
        schema_text = json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
        signature = build_dynamic_model(
            "_SubagentOutputSignature",
            dspy.Signature,
            {
                "user_prompt": (str, dspy.InputField()),
                "result": (
                    dict[str, Any],
                    dspy.OutputField(
                        desc=f"Return one JSON object matching this JSON Schema: {schema_text}"
                    ),
                ),
            },
        ).with_instructions(instructions)
        tools = cast(list[Callable[..., Any]], list(self.scoped_tools))
        with delegation_scope(getattr(self, "_subagent_failure_policy", "fail_closed")):
            with self._produce_lm_context():
                if self.active_profile.mcp_servers:
                    prediction = run_sync(
                        self._produce_with_mcp(signature, tools, template, run_kwargs)
                    )
                else:
                    adapter = (
                        dspy.ReAct(signature, tools=tools) if tools else dspy.Predict(signature)
                    )
                    prediction = adapter(**run_kwargs, user_prompt=template)
        _report_dspy_invocation(prediction)
        result = _prediction_value(prediction, "result")
        if not isinstance(result, Mapping):
            raise TypeError("DSPy subagent structured output must be an object")
        return result

    def _produce_lm_context(self) -> AbstractContextManager[Any]:
        """Make the active LM authoritative for one produce call.

        ``profile_override`` already wraps nested calls in ``dspy.context(lm=...)``;
        the innermost context wins, so we only apply the adapter default when no
        profile override is on the stack. The baseline ``dspy.settings.configure``
        from ``__init__`` remains the fallback for first-run safety.
        """
        stack_var = getattr(self, "_profile_stack_var", None)
        if stack_var is not None and stack_var.get():
            return nullcontext()
        if self._lm_model is None:
            return nullcontext()
        return dspy.context(lm=self._default_lm(), track_usage=True)

    async def _produce_with_mcp(
        self,
        sig: type[dspy.Signature],
        tools: list[Callable[..., Any]],
        template: str,
        kwargs: dict[str, Any],
    ) -> dspy.Prediction:
        mcp_tools: list[Any] = []
        async with AsyncExitStack() as stack:
            for spec in self.active_profile.mcp_servers:
                if spec.transport != "stdio":
                    raise NotImplementedError("DSPy MCP support currently requires stdio transport")
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client

                if spec.command is None:
                    raise ValueError("MCP stdio transport requires a command.")
                server_params = StdioServerParameters(
                    command=spec.command,
                    args=list(spec.args),
                    env=spec.env,
                )
                read, write = await stack.enter_async_context(stdio_client(server_params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed_tools = await session.list_tools()
                mcp_tools.extend(
                    dspy.Tool.from_mcp_tool(session, tool) for tool in listed_tools.tools
                )
            prediction = dspy.ReAct(sig, tools=[*tools, *mcp_tools])
            return await prediction.acall(**kwargs, user_prompt=template)

    def default_model(self) -> str | None:
        return self._lm_model

    def telemetry_model_name(self) -> str | None:
        return self._lm_model

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Iterator[None]:
        policies = request.approval_ceiling
        if request.profile.approval is not None or not policies:
            policies = (*policies, request.profile.approval)
        tools = tuple(_dspy_tool_with_approval_policies(tool, policies) for tool in request.tools)
        with self.profile_override(request.profile), self.tool_scope(tools):
            yield

    @staticmethod
    def subagent_run_kwargs(_limits: SubagentUsageLimits) -> dict[str, Any]:
        return {}

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        needs_model_override = profile.needs_model_override(self.default_model())
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            if not needs_model_override and profile.effort is None and not profile.settings:
                yield
                return
            model = self._lm_model
            if needs_model_override:
                if profile.model is None:
                    raise RuntimeError("Profile requires a model override but no model is set.")
                model = normalize_for_litellm(profile.model)
            provider, lm_kwargs = self._lm_kwargs(dict(profile.settings), effort=profile.effort)
            lm = ThinkingStrippingLM(
                model=model,
                provider=provider,
                **lm_kwargs,
            )
            with dspy.context(lm=lm, track_usage=True):
                yield
        finally:
            stack_var.reset(token)


def _prediction_value(prediction: Any, field: str) -> Any:
    if isinstance(prediction, dict):
        return prediction[field]
    return getattr(prediction, field)


def _dspy_tool_with_approval_policies(
    tool: ToolSpec,
    policies: Sequence[ApprovalPolicy | None],
) -> ToolSpec:
    guarded = tool
    for policy in reversed(tuple(policies)):
        guarded = with_approval(
            guarded,
            policy=policy,
            adapter_shortname=DSPyAdapter.shortname,
        )
    return guarded


def _dspy_recoverable_subagent_tool(tool: ToolSpec) -> ToolSpec:
    @wraps(tool.fn)
    async def recoverable(*args: Any, **kwargs: Any) -> Any:
        try:
            result = tool.fn(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except KediExecutionError as exc:
            return safe_subagent_error_message(exc) or "Subagent failed"

    return replace(tool, fn=recoverable)


def _report_dspy_invocation(prediction: Any) -> None:
    if invocation_observer() is None:
        return
    input_tokens = output_tokens = total_tokens = 0
    usage = prediction.get_lm_usage() if hasattr(prediction, "get_lm_usage") else None
    if isinstance(usage, Mapping):
        for model_usage in usage.values():
            if not isinstance(model_usage, Mapping):
                continue
            model_input = _usage_value(model_usage, "prompt_tokens", "input_tokens")
            model_output = _usage_value(
                model_usage,
                "completion_tokens",
                "output_tokens",
            )
            model_total = _usage_value(model_usage, "total_tokens")
            input_tokens += model_input
            output_tokens += model_output
            total_tokens += model_total or model_input + model_output
    report_invocation(
        InvocationUsage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    )


def _usage_value(usage: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return 0
