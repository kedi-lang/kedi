from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping, Sequence, cast, overload

from pydantic import BaseModel
from pydantic_ai import Agent, AgentModelSettings, AgentRetries, AgentSpec
from pydantic_ai.agent.abstract import AgentInstructions, AgentMetadata, RunOutputDataT
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    HandleDeferredToolCalls,
    ValidatedToolArgs,
    WrapRunHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelMessage, ToolCallPart, UserContent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.run import AgentRun
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ...agent_profile import AgentProfile
from ...agent_profile.settings import (
    PYDANTIC_MODEL_SETTING_KEYS,
    pydantic_model_settings,
)
from ...dynamic_models import build_dynamic_model
from ...effort import pydantic_effort_value
from ...errors import KediExecutionError
from ...model_normalization import normalize_for_pydantic_ai
from ...resolved_types import ResolvedKediType
from .._schema import output_field_definition
from ..approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalDenied,
    ToolRisk,
)
from ..conversation import AdapterConversationContext
from ..invocation import InvocationUsage, invocation_observer, report_invocation
from ..meta import AdapterCapabilities, AgentAdapterKind
from ..subagents import SubagentConversationContext, SubagentExecutionRequest
from ..subagents.attempts import DelegationFailurePolicy, delegation_scope
from ..subagents.limits import SubagentUsageLimits
from ..telemetry import (
    native_agent_span_scope,
    native_tool_span_scope,
    reclassify_native_agent_span,
)
from ..tool_override import (
    ToolScope,
    ToolSpec,
    approval_request_for_tool,
    arguments_after_approval,
    arguments_target_secret_file,
    current_required_tool_calls,
    required_tool_call_scope,
    resolve_tool_arguments_async,
    with_required_before_output,
)
from ._pydantic_compat import (
    application_toolset_count,
    application_toolsets,
    build_base_toolset_list,
    has_application_tools,
    infer_agent_name,
)

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.tools import Tool as PydanticTool


class _KediTelemetryCapability(AbstractCapability[Any]):
    """Reclassify Pydantic AI's native run span with Kedi semantics."""

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        del ctx
        with native_agent_span_scope(True):
            reclassify_native_agent_span()
            return await handler()

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        del ctx, tool_def
        with native_tool_span_scope(call.tool_name):
            return await handler(args)


class _KediRequiredToolCapability(AbstractCapability[Any]):
    """Reject output until every required Kedi tool has completed."""

    async def after_output_process(
        self,
        ctx: RunContext[Any],
        *,
        output_context: Any,
        output: Any,
    ) -> Any:
        del ctx, output_context
        return PydanticAdapter._validate_required_tool_calls(output)


class _KediDeferredApprovalCapability(HandleDeferredToolCalls[Any]):
    """Identify Kedi's approval capability so nested preparation can deduplicate it."""


class _KediExternalApprovalCapability(AbstractCapability[Any]):
    """Normalize deferred approvals without taking ownership of the decision."""

    def __init__(self, adapter: PydanticAdapter[Any, Any]) -> None:
        self.adapter = adapter

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        del ctx, tool_def
        try:
            return await handler(args)
        except ApprovalRequired as exc:
            request = self.adapter._approval_request_for_call(call.tool_name, args)
            call.args = dict(request.arguments)
            metadata = dict(exc.metadata or {})
            metadata[KEDI_APPROVAL_METADATA_KEY] = _approval_request_payload(request)
            raise ApprovalRequired(metadata) from exc


PydanticApprovalResolution = Literal["kedi", "external"]
KEDI_APPROVAL_METADATA_KEY = "kedi_approval"


class _KediPydanticTool(Tool[Any]):
    """A native Pydantic tool that retains its adapter-neutral Kedi specification."""

    __slots__ = ("kedi_spec",)

    kedi_spec: ToolSpec


def _mcp_toolset_from_config(name: str, server_config: dict[str, object]) -> Any:
    from pydantic_ai.mcp import MCPToolset

    config = {"mcpServers": {name: server_config}}
    return MCPToolset(cast(Any, config))


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    """
    Adapter to use Pydantic AI agents.
    """

    kind: AgentAdapterKind = "agent-framework"
    shortname = "pydantic"
    supports_native_subagent_resume = True
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_native_approvals=True,
        supports_native_approval_handler=True,
        supports_subagents=True,
        supports_background_subagents=True,
        supports_artifacts=True,
        supports_native_artifacts=False,
        supports_stateful_history=True,
        supports_history_replay=True,
        settings=PYDANTIC_MODEL_SETTING_KEYS,
    )

    def native_telemetry_enabled(self, operation: str) -> bool:
        """Report per-agent native ownership without relying on process-global state."""

        return operation in {"run_agent", "chat", "call_tool"} and self.instrument is not False

    def __init__(
        self,
        model: str | Model | None = None,
        *,
        model_settings: dict[str, Any] | None = None,
        approval_resolution: PydanticApprovalResolution = "kedi",
        subagent_failure_policy: DelegationFailurePolicy = "fail_closed",
        **kwargs: Any,
    ) -> None:
        """Initialize the native Pydantic AI adapter.

        ``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style
        (``vendor:a/b``). ``approval_resolution='kedi'`` resolves deferred tool
        calls with the active Kedi policy. ``'external'`` keeps those calls
        deferred for an outer controller such as an ACP runtime.
        """
        self._tool_scope = ToolScope()
        self._tool_spec_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_pydantic_profile_stack_{id(self)}",
            default=(),
        )
        self._isolated_run_depth: ContextVar[int] = ContextVar(
            f"kedi_pydantic_isolated_run_depth_{id(self)}",
            default=0,
        )
        self._approval_ceiling_var: ContextVar[tuple[ApprovalPolicy | None, ...]] = ContextVar(
            f"kedi_pydantic_approval_ceiling_{id(self)}",
            default=(),
        )
        self._run_tool_specs_var: ContextVar[tuple[ToolSpec, ...] | None] = ContextVar(
            f"kedi_pydantic_run_tool_specs_{id(self)}",
            default=None,
        )
        self._subagent_conversation_var: ContextVar[
            AdapterConversationContext | SubagentConversationContext | None
        ] = ContextVar(
            f"kedi_pydantic_subagent_conversation_{id(self)}",
            default=None,
        )
        if subagent_failure_policy not in {"recover", "fail_closed"}:
            raise ValueError(f"Unknown subagent failure policy: {subagent_failure_policy!r}")
        if approval_resolution not in {"kedi", "external"}:
            raise ValueError(f"Unknown approval resolution owner: {approval_resolution!r}")
        self.approval_resolution = approval_resolution
        self._subagent_failure_policy = subagent_failure_policy
        if isinstance(model, str):
            model = normalize_for_pydantic_ai(model)
        self._agent_model_arg = model
        self._default_model = model if isinstance(model, str) else None
        if model_settings:
            filtered_settings = pydantic_model_settings(model_settings)
            if filtered_settings:
                kwargs["model_settings"] = filtered_settings
        self._agent_init_kwargs = dict(kwargs)
        super().__init__(model, **kwargs)

    @staticmethod
    def _validate_required_tool_calls(output: Any) -> Any:
        state = current_required_tool_calls()
        if state is None or state.names <= state.returned:
            return output

        from pydantic_ai import ModelRetry

        missing_names = state.names - state.returned
        missing = ", ".join(sorted(missing_names))
        if missing_names & state.attempted:
            raise ModelRetry(
                f"Required tool {missing} did not complete successfully. Correct its previous "
                "error and call it again before returning the parent response."
            )
        raise ModelRetry(
            f"Call required tool {missing} and use its result before returning the parent response."
        )

    @contextmanager
    def tool_scope(self, tools: Sequence[Any] = ()) -> Iterator[None]:
        """Register Pydantic tools and their Kedi metadata in one lexical scope."""
        with (
            self._tool_scope.tool_scope(),
            self._tool_spec_scope_for_current_adapter().tool_scope(),
        ):
            for tool in tools:
                self.register_tool(cast(ToolSpec, tool))
            yield

    def register_tool(self, tool: ToolSpec) -> None:
        tool = with_required_before_output(tool)
        native_tool = pydantic_tool_from_spec(tool)
        self._tool_scope.register_tool(native_tool)
        self._tool_spec_scope_for_current_adapter().register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[PydanticTool[Any], ...]:
        return cast("tuple[PydanticTool[Any], ...]", self._tool_scope.scoped_tools)

    @property
    def scoped_tool_specs(self) -> tuple[ToolSpec, ...]:
        return cast(
            "tuple[ToolSpec, ...]", self._tool_spec_scope_for_current_adapter().scoped_tools
        )

    @contextmanager
    def isolated_run(self) -> Iterator[None]:
        """Run a nested agent with only the tools registered in the current scope."""
        depth = self._isolated_run_depth_for_current_adapter()
        token = depth.set(depth.get() + 1)
        try:
            yield
        finally:
            depth.reset(token)

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Iterator[None]:
        token = self._subagent_conversation_var.set(request.conversation)
        try:
            with (
                self.tool_scope(request.tools),
                self.profile_override(request.profile),
                self.approval_ceiling(request.approval_ceiling),
                self.isolated_run(),
            ):
                yield
        finally:
            self._subagent_conversation_var.reset(token)

    @contextmanager
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> Iterator[None]:
        token = self._subagent_conversation_var.set(conversation)
        try:
            yield
        finally:
            self._subagent_conversation_var.reset(token)

    @staticmethod
    def subagent_run_kwargs(limits: SubagentUsageLimits) -> dict[str, Any]:
        return {"usage_limits": limits.to_pydantic()}

    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        from pydantic_ai.output import StructuredDict

        result = await self.produce(
            template=template,
            output_type=StructuredDict(output_schema),
            **kwargs,
        )
        if not isinstance(result, Mapping):
            raise TypeError("Pydantic subagent structured output must be an object")
        return cast(Mapping[str, Any], result)

    @contextmanager
    def approval_ceiling(
        self,
        policies: Sequence[ApprovalPolicy | None],
    ) -> Iterator[None]:
        stack_var = self._approval_ceiling_for_current_adapter()
        token = stack_var.set(tuple(policies))
        try:
            yield
        finally:
            stack_var.reset(token)

    @property
    def active_profile(self) -> AgentProfile:
        stack_var = getattr(self, "_profile_stack_var", None)
        stack = stack_var.get() if stack_var is not None else ()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[OutputDataT]:
        """
        Type builder to create a Pydantic model from a given output schema.
        Args:
            output_schema (dict): A dictionary defining the output schema.
        Returns:
            type: A Pydantic model class based on the provided schema.
        """
        if not output_schema:
            raise ValueError("Output schema cannot be empty.")
        field_definitions = {
            field: output_field_definition(type_) for field, type_ in output_schema.items()
        }
        model = build_dynamic_model("_OutputModel", BaseModel, field_definitions)
        return cast(type[OutputDataT], model)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs,
    ) -> OutputDataT:
        """
        Invoke the Pydantic AI agent with the provided template and output schema.
        """
        if not output_schema and output_type is None:
            raise ValueError("Output spec must be provided.")
        if output_schema:
            output_type = self.type_builder(output_schema)
        run_kwargs: dict[str, object] = {"output_type": output_type, **kwargs}
        agent = self._agent_for_produce()
        tool_specs = self._tool_specs_for_run(run_kwargs)
        if agent is not self:
            run_kwargs = self._run_kwargs(run_kwargs, tool_specs=tool_specs)
            run_kwargs.update(self._profile_override_kwargs(self.active_profile))
            self._add_nested_native_toolsets(run_kwargs)
        return (await self._run_agent(agent, template, run_kwargs, tool_specs=tool_specs)).output

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        run_kwargs: dict[str, object] = dict(kwargs)
        agent = self._agent_for_produce()
        tool_specs = self._tool_specs_for_run(run_kwargs)
        if agent is not self:
            run_kwargs = self._run_kwargs(run_kwargs, tool_specs=tool_specs)
            run_kwargs.update(self._profile_override_kwargs(self.active_profile))
            self._add_nested_native_toolsets(run_kwargs)
        return str((await self._run_agent(agent, prompt, run_kwargs, tool_specs=tool_specs)).output)

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, Any]]:
        """Apply active Kedi state to every native Pydantic AI run surface."""

        if infer_name and self.name is None:
            infer_agent_name(self, inspect.currentframe())
        return self._iter_with_kedi_state(
            user_prompt,
            {
                "output_type": output_type,
                "message_history": message_history,
                "deferred_tool_results": deferred_tool_results,
                "conversation_id": conversation_id,
                "run_id": run_id,
                "model": model,
                "instructions": instructions,
                "deps": deps,
                "model_settings": model_settings,
                "usage_limits": usage_limits,
                "usage": usage,
                "metadata": metadata,
                "retries": retries,
                "infer_name": False,
                "toolsets": toolsets,
                "capabilities": capabilities,
                "spec": spec,
            },
        )

    @asynccontextmanager
    async def _iter_with_kedi_state(
        self,
        user_prompt: str | Sequence[UserContent] | None,
        kwargs: Mapping[str, Any],
    ) -> AsyncGenerator[AgentRun[AgentDepsT, Any]]:
        tool_specs = self._tool_specs_for_run(kwargs)
        run_kwargs = self._run_kwargs(kwargs, tool_specs=tool_specs)
        with self._run_tool_spec_scope(tool_specs):
            async with super().iter(user_prompt, **cast(Any, run_kwargs)) as agent_run:
                yield agent_run

    async def _run_agent(
        self,
        agent: Agent[Any, Any],
        prompt: str,
        run_kwargs: dict[str, object],
        *,
        tool_specs: Sequence[ToolSpec] | None = None,
    ) -> Any:
        conversation = self._subagent_conversation_var.get()
        if conversation is not None and "pydantic_messages" in conversation.resume_state:
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            run_kwargs.setdefault(
                "message_history",
                ModelMessagesTypeAdapter.validate_python(
                    conversation.resume_state["pydantic_messages"]
                ),
            )
        if tool_specs is None:
            tool_specs = self._tool_specs_for_run(run_kwargs)
        with self._run_tool_spec_scope(tool_specs), delegation_scope(self._subagent_failure_policy):
            observer = invocation_observer()
            if observer is None:
                result = await agent.run(prompt, **cast(Any, run_kwargs))
                self._capture_subagent_messages(conversation, result)
                return result

            from pydantic_ai.usage import RunUsage

            usage = run_kwargs.get("usage")
            if not isinstance(usage, RunUsage):
                usage = RunUsage()
                run_kwargs["usage"] = usage
            try:
                result = await agent.run(prompt, **cast(Any, run_kwargs))
                self._capture_subagent_messages(conversation, result)
                return result
            finally:
                report_invocation(
                    InvocationUsage(
                        requests=usage.requests,
                        tool_calls=usage.tool_calls,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        total_tokens=usage.total_tokens,
                    )
                )

    @contextmanager
    def _required_tool_call_scope(self) -> Iterator[None]:
        tool_specs = self._run_tool_specs_var.get()
        with required_tool_call_scope(
            self.scoped_tool_specs if tool_specs is None else tool_specs,
            inherit=True,
        ):
            yield

    @contextmanager
    def _run_tool_spec_scope(self, tool_specs: Sequence[ToolSpec]) -> Iterator[None]:
        token = self._run_tool_specs_var.set(tuple(tool_specs))
        try:
            with self._required_tool_call_scope():
                yield
        finally:
            self._run_tool_specs_var.reset(token)

    @staticmethod
    def _capture_subagent_messages(
        conversation: AdapterConversationContext | SubagentConversationContext | None,
        result: Any,
    ) -> None:
        if conversation is None:
            return
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        conversation.result_state["pydantic_messages"] = ModelMessagesTypeAdapter.dump_python(
            result.all_messages(),
            mode="json",
        )

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        from ..utils import run_sync

        return run_sync(self.invoke(prompt=prompt, **kwargs))

    def _run_kwargs(
        self,
        kwargs: Mapping[str, Any],
        *,
        tool_specs: Sequence[ToolSpec],
    ) -> dict[str, object]:
        run_kwargs: dict[str, object] = dict(kwargs)
        if self.active_profile.system is not None and run_kwargs.get("instructions") is None:
            run_kwargs["instructions"] = self.active_profile.system
        model_settings = pydantic_model_settings(self.active_profile.settings)
        if model_settings:
            existing_settings = run_kwargs.get("model_settings")
            if callable(existing_settings):
                run_kwargs["model_settings"] = _profile_model_settings(
                    cast(
                        "Callable[[RunContext[Any]], Mapping[str, Any]]",
                        existing_settings,
                    ),
                    model_settings,
                )
            elif isinstance(existing_settings, Mapping):
                run_kwargs["model_settings"] = {**existing_settings, **model_settings}
            else:
                run_kwargs["model_settings"] = model_settings
        from ...tool_context import inside_kedi_tool_call

        toolsets = self._toolsets_for_run(
            include_scoped_tools=(
                self._isolated_run_depth_for_current_adapter().get() > 0
                or not inside_kedi_tool_call()
            )
        )
        caller_toolsets = _toolset_sequence(run_kwargs.get("toolsets"))
        if caller_toolsets or toolsets:
            run_kwargs["toolsets"] = [
                *(_approval_required_toolset(toolset) for toolset in caller_toolsets),
                *(toolsets or ()),
            ]
        capabilities = run_kwargs.get("capabilities")
        if self.approval_resolution == "kedi" and (
            caller_toolsets or toolsets is not None or self._has_native_approval_tools()
        ):
            capabilities = self._with_deferred_approval_capability(capabilities)
        elif self.approval_resolution == "external" and (
            caller_toolsets or toolsets is not None or self._has_native_approval_tools()
        ):
            capabilities = self._with_external_approval_capability(capabilities)
        if any((tool.metadata or {}).get("required_before_output") is True for tool in tool_specs):
            capabilities = _with_capability(
                capabilities,
                _KediRequiredToolCapability,
                _KediRequiredToolCapability,
            )
        capabilities = self._with_telemetry_capability(capabilities)
        if capabilities:
            run_kwargs["capabilities"] = capabilities
        return run_kwargs

    def _with_telemetry_capability(self, existing: object | None) -> list[Any]:
        from ...telemetry import owns_native_spans

        if existing is None:
            capabilities: list[Any] = []
        elif isinstance(existing, (list, tuple)):
            capabilities = list(existing)
        else:
            raise TypeError("Pydantic AI capabilities must be a list or tuple.")
        if (
            owns_native_spans("pydantic", "run_agent")
            and self.native_telemetry_enabled("run_agent")
            and not any(
                isinstance(capability, _KediTelemetryCapability) for capability in capabilities
            )
        ):
            capabilities.append(_KediTelemetryCapability())
        return capabilities

    def _agent_for_produce(self) -> Agent[Any, Any]:
        from ...tool_context import inside_kedi_tool_call

        if (
            not inside_kedi_tool_call()
            and self._isolated_run_depth_for_current_adapter().get() == 0
        ):
            return self
        # Keep nested Kedi tool calls isolated from the outer agent's scoped
        # Kedi tools. Native constructor and inherited toolsets are supplied
        # below through the normal per-run toolset surface with approval.
        init_kwargs = {
            name: value
            for name, value in self._agent_init_kwargs.items()
            if name not in {"tools", "toolsets"}
        }
        return Agent(self._agent_model_arg, **init_kwargs)

    def _add_nested_native_toolsets(self, run_kwargs: dict[str, object]) -> None:
        if not self._has_native_approval_tools():
            return
        native_toolsets = [
            _approval_required_toolset(toolset) for toolset in application_toolsets(self)
        ]
        existing_toolsets = run_kwargs.get("toolsets")
        if existing_toolsets is None:
            run_kwargs["toolsets"] = native_toolsets
            return
        if isinstance(existing_toolsets, (str, bytes)) or not isinstance(
            existing_toolsets, Sequence
        ):
            raise TypeError("Pydantic AI toolsets must be a sequence.")
        run_kwargs["toolsets"] = [*existing_toolsets, *native_toolsets]

    def _build_toolset_list(self, cap_toolsets: Sequence[Any] | None = None) -> list[Any]:
        """Require native approval for every application-provided Pydantic toolset."""

        toolsets = build_base_toolset_list(self, cap_toolsets)
        toolset_count = application_toolset_count(self)
        return [
            _approval_required_toolset(toolset) if index < toolset_count else toolset
            for index, toolset in enumerate(toolsets)
        ]

    def _application_toolset_count(self) -> int:
        """Count function, user, and dynamic toolsets without capability toolsets."""

        return application_toolset_count(self)

    def _has_native_approval_tools(self) -> bool:
        """Whether constructor or inherited Pydantic toolsets can issue a call."""

        return has_application_tools(self)

    def _toolsets_for_run(self, *, include_scoped_tools: bool = True) -> list[Any] | None:
        toolsets: list[Any] = []
        scoped_tools = self.scoped_tools if include_scoped_tools else ()
        if scoped_tools:
            toolsets.append(FunctionToolset(tools=scoped_tools).approval_required())
        toolsets.extend(
            toolset.approval_required()
            for toolset in self._mcp_toolsets_for_profile(self.active_profile)
        )
        return toolsets or None

    def _tool_specs_for_run(self, run_kwargs: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        from ...tool_context import inside_kedi_tool_call

        include_scoped_tools = (
            self._isolated_run_depth_for_current_adapter().get() > 0 or not inside_kedi_tool_call()
        )
        toolsets = [
            *application_toolsets(self),
            *_toolset_sequence(run_kwargs.get("toolsets")),
        ]
        return (
            *(self.scoped_tool_specs if include_scoped_tools else ()),
            *_kedi_tool_specs_from_toolsets(toolsets),
        )

    def _with_deferred_approval_capability(self, existing: object | None) -> list[Any]:
        return _with_capability(
            existing,
            _KediDeferredApprovalCapability,
            lambda: _KediDeferredApprovalCapability(handler=self._handle_deferred_tool_calls),
        )

    def _with_external_approval_capability(self, existing: object | None) -> list[Any]:
        return _with_capability(
            existing,
            _KediExternalApprovalCapability,
            lambda: _KediExternalApprovalCapability(self),
        )

    def _approval_request_for_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ApprovalRequest:
        tool = self._tool_spec_for_call(tool_name)
        return approval_request_for_tool(tool, arguments, self.shortname)

    def _tool_spec_for_call(self, tool_name: str) -> ToolSpec:
        active_tool_specs = self._run_tool_specs_var.get()
        known_tools = {
            tool.name: tool
            for tool in (self.scoped_tool_specs if active_tool_specs is None else active_tool_specs)
        }
        return known_tools.get(tool_name, _external_tool_spec(tool_name))

    async def _handle_deferred_tool_calls(self, _context: Any, requests: Any) -> Any:
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for call in requests.approvals:
            try:
                arguments = call.args_as_dict()
            except ValueError as exc:
                approvals[call.tool_call_id] = ToolDenied(
                    f"Kedi could not parse arguments for tool {call.tool_name!r}: {exc}"
                )
                continue

            tool = self._tool_spec_for_call(call.tool_name)
            policies = self._approval_ceiling_for_current_adapter().get()
            local_policy = self.active_profile.approval
            if local_policy is not None or not policies:
                policies = (*policies, local_policy)

            approved_arguments = arguments
            for policy in policies:
                try:
                    approved_arguments = await resolve_tool_arguments_async(
                        tool,
                        approved_arguments,
                        policy=policy,
                        adapter_shortname=self.shortname,
                    )
                except ToolApprovalDenied as exc:
                    message = str(exc.args[0]) if exc.args else f"Tool {tool.name!r} was denied"
                    approvals[call.tool_call_id] = ToolDenied(message)
                    break
            else:
                approvals[call.tool_call_id] = ToolApproved(override_args=approved_arguments)
        return requests.build_results(approvals=approvals)

    def _tool_spec_scope_for_current_adapter(self) -> ToolScope:
        try:
            return self._tool_spec_scope
        except AttributeError:
            scope = ToolScope()
            self._tool_spec_scope = scope
            return scope

    def _isolated_run_depth_for_current_adapter(self) -> ContextVar[int]:
        try:
            return self._isolated_run_depth
        except AttributeError:
            depth = ContextVar(
                f"kedi_pydantic_isolated_run_depth_{id(self)}",
                default=0,
            )
            self._isolated_run_depth = depth
            return depth

    def _approval_ceiling_for_current_adapter(
        self,
    ) -> ContextVar[tuple[ApprovalPolicy | None, ...]]:
        try:
            return self._approval_ceiling_var
        except AttributeError:
            ceiling: ContextVar[tuple[ApprovalPolicy | None, ...]] = ContextVar(
                f"kedi_pydantic_approval_ceiling_{id(self)}",
                default=(),
            )
            self._approval_ceiling_var = ceiling
            return ceiling

    def _mcp_toolsets_for_profile(self, profile: AgentProfile) -> list[Any]:
        if not profile.mcp_servers:
            return []
        from pydantic_ai.mcp import MCPToolset

        toolsets: list[Any] = []
        for index, spec in enumerate(profile.mcp_servers):
            if spec.transport == "stdio":
                if spec.command is None:
                    raise ValueError("MCP stdio transport requires a command.")
                stdio_config: dict[str, object] = {
                    "command": spec.command,
                    "args": list(spec.args),
                }
                if spec.env is not None:
                    stdio_config["env"] = spec.env
                toolsets.append(_mcp_toolset_from_config(f"kedi_mcp_{index}", stdio_config))
            elif spec.transport == "sse":
                if spec.url is None:
                    raise ValueError("MCP sse transport requires a url.")
                sse_config: dict[str, object] = {"url": spec.url, "transport": "sse"}
                if spec.headers is not None:
                    sse_config["headers"] = spec.headers
                toolsets.append(_mcp_toolset_from_config(f"kedi_mcp_{index}", sse_config))
            elif spec.transport == "http":
                if spec.url is None:
                    raise ValueError("MCP http transport requires a url.")
                toolsets.append(MCPToolset(spec.url, headers=spec.headers))
            else:
                raise ValueError(f"Unsupported MCP transport: {spec.transport}")
        return toolsets

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs,
    ) -> OutputDataT:
        """
        Synchronous version of produce method.
        """
        from ..utils import run_sync

        return run_sync(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        default_model = getattr(self, "_default_model", None)
        if isinstance(default_model, str):
            return default_model
        model = getattr(self, "_model", None)
        return model if isinstance(model, str) else None

    def telemetry_model_name(self) -> str | None:
        if isinstance(self._agent_model_arg, str):
            return self._agent_model_arg
        if isinstance(self._agent_model_arg, Model):
            return self._agent_model_arg.model_name
        return None

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        override_kwargs = self._profile_override_kwargs(profile)
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            if not override_kwargs:
                yield
                return
            with self.override(**override_kwargs):
                yield
        finally:
            stack_var.reset(token)

    def _profile_override_kwargs(self, profile: AgentProfile) -> dict[str, Any]:
        override_kwargs: dict[str, Any] = {}
        if profile.needs_model_override(self.default_model()):
            if profile.model is None:
                raise RuntimeError("Profile requires a model override but no model is set.")
            override_kwargs["model"] = normalize_for_pydantic_ai(profile.model)
        if profile.effort is not None:
            override_kwargs["spec"] = {
                "capabilities": [
                    {
                        "name": "Thinking",
                        "arguments": {"effort": pydantic_effort_value(profile.effort)},
                    }
                ]
            }
        return override_kwargs


def _validated_tool_function(tool: ToolSpec) -> Any:
    """Preserve Kedi argument validation when Pydantic AI invokes a native tool."""

    if _is_async_callable(tool.fn):

        @wraps(tool.fn)
        async def validated_async_tool(*args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")
            arguments = _validated_tool_arguments(tool, kwargs)
            try:
                result = tool.fn(**arguments)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as exc:
                retry = _model_retry_for_tool_error(tool, exc)
                if retry is None:
                    raise
                raise retry from exc

        return validated_async_tool

    @wraps(tool.fn)
    def validated_tool(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")
        try:
            result = tool.fn(**_validated_tool_arguments(tool, kwargs))
            return result
        except Exception as exc:
            retry = _model_retry_for_tool_error(tool, exc)
            if retry is None:
                raise
            raise retry from exc

    return validated_tool


def pydantic_tool_from_spec(
    tool: ToolSpec,
    *,
    args_validator: Any = None,
) -> Tool[Any]:
    """Convert an adapter-neutral Kedi tool specification to a Pydantic AI tool."""

    tool = with_required_before_output(tool)
    function = _validated_tool_function(tool)
    if tool.json_schema is None:
        native_tool = _KediPydanticTool(
            function,
            name=tool.name,
            description=tool.description,
            sequential=tool.sequential,
            args_validator=args_validator,
        )
    else:
        native_tool = _KediPydanticTool.from_schema(
            function=function,
            name=tool.name,
            description=tool.description or "",
            json_schema=tool.json_schema,
            takes_ctx=False,
            sequential=tool.sequential,
            args_validator=args_validator,
        )
    native_tool.kedi_spec = tool
    native_tool.metadata = tool.metadata
    return native_tool


def pydantic_toolset_from_specs(tools: Sequence[ToolSpec]) -> FunctionToolset[Any]:
    """Convert Kedi tool specifications into a semantics-preserving Pydantic toolset."""

    return FunctionToolset(tools=[pydantic_tool_from_spec(tool) for tool in tools])


def pydantic_approval_request_from_deferred(
    requests: DeferredToolRequests,
    tool_call_id: str,
) -> ApprovalRequest:
    """Restore Kedi's canonical approval request from a deferred Pydantic call."""

    metadata = requests.metadata.get(tool_call_id, {})
    payload = metadata.get(KEDI_APPROVAL_METADATA_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Deferred tool call {tool_call_id!r} has no Kedi approval metadata")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"Deferred tool call {tool_call_id!r} has invalid Kedi arguments")
    tool_metadata = payload.get("metadata")
    if tool_metadata is not None and not isinstance(tool_metadata, Mapping):
        raise ValueError(f"Deferred tool call {tool_call_id!r} has invalid Kedi tool metadata")
    risk = payload.get("risk")
    if risk not in {"read_only", "mutating", "sensitive"}:
        raise ValueError(f"Deferred tool call {tool_call_id!r} has invalid Kedi risk")
    return ApprovalRequest.create(
        tool_name=str(payload.get("tool_name", "")),
        arguments=arguments,
        risk=cast(ToolRisk, risk),
        adapter_shortname=(
            str(payload["adapter_shortname"])
            if payload.get("adapter_shortname") is not None
            else None
        ),
        description=(
            str(payload["description"]) if payload.get("description") is not None else None
        ),
        metadata=tool_metadata,
    )


def pydantic_deferred_results_from_decisions(
    requests: DeferredToolRequests,
    decisions: Mapping[str, ApprovalDecision | bool],
    *,
    tool_specs: Sequence[ToolSpec],
) -> DeferredToolResults:
    """Validate external approval decisions and build Pydantic deferred results."""

    known_tools = {tool.name: tool for tool in tool_specs}
    approvals: dict[str, bool | ToolApproved | ToolDenied] = {}
    calls_by_id = {call.tool_call_id: call for call in requests.approvals}
    for tool_call_id, raw_decision in decisions.items():
        call = calls_by_id.get(tool_call_id)
        if call is None:
            raise ValueError(f"Unknown deferred approval tool call ID: {tool_call_id!r}")
        if raw_decision is True:
            decision = ApprovalDecision.allow()
        elif raw_decision is False:
            decision = ApprovalDecision.deny()
        else:
            decision = raw_decision
        request = pydantic_approval_request_from_deferred(requests, tool_call_id)
        tool = known_tools.get(call.tool_name, _external_tool_spec(call.tool_name))
        if decision.decision == "deny":
            message = decision.reason or f"Tool {call.tool_name!r} was denied"
            approvals[tool_call_id] = ToolDenied(message)
            continue
        approvals[tool_call_id] = ToolApproved(
            override_args=arguments_after_approval(tool, request.arguments, decision)
        )
    return requests.build_results(approvals=approvals)


def _kedi_tool_specs_from_toolsets(
    toolsets: Sequence[AbstractToolset[Any]],
) -> tuple[ToolSpec, ...]:
    specs: list[ToolSpec] = []

    def collect(toolset: AbstractToolset[Any]) -> None:
        if isinstance(toolset, FunctionToolset):
            specs.extend(
                tool.kedi_spec
                for tool in toolset.tools.values()
                if isinstance(tool, _KediPydanticTool)
            )

    for toolset in toolsets:
        toolset.apply(collect)
    return tuple(specs)


def _toolset_sequence(value: object | None) -> list[AbstractToolset[Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("Pydantic AI toolsets must be a sequence.")
    return [cast(AbstractToolset[Any], toolset) for toolset in value]


def _with_capability(
    existing: object | None,
    capability_type: type[Any],
    factory: Callable[[], Any],
) -> list[Any]:
    if existing is None:
        capabilities: list[Any] = []
    elif isinstance(existing, (list, tuple)):
        capabilities = list(existing)
    else:
        raise TypeError("Pydantic AI capabilities must be a list or tuple.")
    if not any(isinstance(capability, capability_type) for capability in capabilities):
        capabilities.append(factory())
    return capabilities


def _profile_model_settings(
    settings_factory: Callable[[RunContext[Any]], Mapping[str, Any]],
    profile_settings: Mapping[str, Any],
) -> Callable[[RunContext[Any]], dict[str, Any]]:
    def merged(ctx: RunContext[Any]) -> dict[str, Any]:
        return {**settings_factory(ctx), **profile_settings}

    return merged


def _approval_request_payload(request: ApprovalRequest) -> dict[str, Any]:
    return {
        "tool_name": request.tool_name,
        "arguments": dict(request.arguments),
        "risk": request.risk,
        "adapter_shortname": request.adapter_shortname,
        "description": request.description,
        "metadata": dict(request.metadata) if request.metadata is not None else None,
    }


def _model_retry_for_tool_error(tool: ToolSpec, exc: Exception) -> Exception | None:
    kind = (tool.metadata or {}).get("kind")
    if kind == "dynamic_workflow":
        from pydantic_ai import ModelRetry

        from ..subagents.workflow import WorkflowError

        if isinstance(exc, WorkflowError):
            return ModelRetry(str(exc))
    if kind == "subagent" and isinstance(exc, KediExecutionError):
        from pydantic_ai import ModelRetry

        from ..subagents.runs import safe_subagent_error_message

        message = safe_subagent_error_message(exc) or "Subagent failed"
        return ModelRetry(message)
    if kind == "artifact_management":
        from pydantic_ai import ModelRetry

        from kedi.artifacts import ArtifactCodeError

        if isinstance(exc, ArtifactCodeError):
            return ModelRetry(
                f"{exc}. Correct the artifact Python using the documented helper signatures "
                "and retry."
            )
    return None


def _validated_tool_arguments(tool: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if tool.argument_validator is None:
        return dict(arguments)
    return tool.argument_validator(arguments)


def _is_async_callable(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


def _external_tool_spec(name: str) -> ToolSpec:
    """Describe an MCP/external tool whose implementation stays inside Pydantic AI."""

    return ToolSpec(
        fn=lambda **_kwargs: None,
        name=name,
        description="External Pydantic AI tool",
        metadata={"source": "external"},
        risk="mutating",
        risk_resolver=_external_tool_risk,
    )


def _approval_required_toolset(toolset: Any) -> Any:
    from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

    if isinstance(toolset, ApprovalRequiredToolset):
        return toolset
    return toolset.approval_required()


def _external_tool_risk(arguments: Mapping[str, Any]) -> ToolRisk:
    return "sensitive" if arguments_target_secret_file(arguments) else "mutating"
