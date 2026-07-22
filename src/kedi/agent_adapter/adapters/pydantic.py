from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT

from ...agent_profile import AgentProfile
from ...agent_profile.settings import (
    PYDANTIC_MODEL_SETTING_KEYS,
    pydantic_model_settings,
)
from ...dynamic_models import build_dynamic_model
from ...effort import pydantic_effort_value
from ...model_normalization import normalize_for_pydantic_ai
from ...resolved_types import ResolvedKediType
from .._schema import output_field_definition
from ..meta import AdapterCapabilities, AgentAdapterKind
from ..approval import ToolRisk, resolve_tool_approval_async
from ..tool_override import (
    ToolScope,
    ToolSpec,
    approval_request_for_tool,
    arguments_target_secret_file,
    arguments_after_approval,
)

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool as PydanticTool


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
        settings=PYDANTIC_MODEL_SETTING_KEYS,
    )

    def __init__(
        self,
        model: str | Model | None = None,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """``model`` may be LiteLLM-style (``vendor/a/b``) or Pydantic-style (``vendor:a/b``)."""
        self._tool_scope = ToolScope()
        self._tool_spec_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_pydantic_profile_stack_{id(self)}",
            default=(),
        )
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
        from pydantic_ai.tools import Tool

        function = _validated_tool_function(tool)
        if tool.json_schema is None:
            native_tool = Tool(function, name=tool.name)
        else:
            native_tool = Tool.from_schema(
                function=function,
                name=tool.name,
                description=tool.description or "",
                json_schema=tool.json_schema,
                takes_ctx=False,
            )
        native_tool.metadata = tool.metadata
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
        run_kwargs = self._run_kwargs({"output_type": output_type, **kwargs})
        agent = self._agent_for_produce()
        if agent is not self:
            run_kwargs.update(self._profile_override_kwargs(self.active_profile))
            self._add_nested_native_toolsets(run_kwargs)
        return (await agent.run(template, **cast(Any, run_kwargs))).output

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        run_kwargs = self._run_kwargs(kwargs)
        agent = self._agent_for_produce()
        if agent is not self:
            run_kwargs.update(self._profile_override_kwargs(self.active_profile))
            self._add_nested_native_toolsets(run_kwargs)
        return str((await agent.run(prompt, **cast(Any, run_kwargs))).output)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        from ..utils import run_sync

        return run_sync(self.invoke(prompt=prompt, **kwargs))

    def _run_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, object]:
        run_kwargs: dict[str, object] = dict(kwargs)
        if self.active_profile.system is not None and "instructions" not in run_kwargs:
            run_kwargs["instructions"] = self.active_profile.system
        model_settings = pydantic_model_settings(self.active_profile.settings)
        if model_settings:
            existing_settings = run_kwargs.get("model_settings")
            if isinstance(existing_settings, dict):
                run_kwargs["model_settings"] = {**existing_settings, **model_settings}
            else:
                run_kwargs["model_settings"] = model_settings
        from ..utils import inside_kedi_tool_call

        toolsets = self._toolsets_for_run(include_scoped_tools=not inside_kedi_tool_call())
        if toolsets is not None:
            run_kwargs["toolsets"] = toolsets
        if toolsets is not None or self._has_native_approval_tools():
            run_kwargs["capabilities"] = self._with_deferred_approval_capability(
                run_kwargs.get("capabilities")
            )
        return run_kwargs

    def _agent_for_produce(self) -> Agent[Any, Any]:
        from ..utils import inside_kedi_tool_call

        if not inside_kedi_tool_call():
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
        native_toolsets = self._build_toolset_list()[: self._application_toolset_count()]
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

        toolsets = super()._build_toolset_list(cap_toolsets=cast(Any, cap_toolsets))
        application_toolset_count = self._application_toolset_count()
        return [
            _approval_required_toolset(toolset) if index < application_toolset_count else toolset
            for index, toolset in enumerate(toolsets)
        ]

    def _application_toolset_count(self) -> int:
        """Count function, user, and dynamic toolsets without capability toolsets."""

        overridden_toolsets = self._override_toolsets.get()
        if overridden_toolsets is not None:
            return 1 + len(overridden_toolsets.value)
        return 1 + len(self._user_toolsets) + len(self._dynamic_toolsets)

    def _has_native_approval_tools(self) -> bool:
        """Whether constructor or inherited Pydantic toolsets can issue a call."""

        try:
            overridden_tools = self._override_tools.get()
            function_tools = self._function_toolset.tools
            overridden_toolsets = self._override_toolsets.get()
            user_toolsets = self._user_toolsets
            dynamic_toolsets = self._dynamic_toolsets
        except AttributeError:
            # Some embedders construct a partial adapter solely to stub
            # ``run``. Such an object has no Pydantic tool surface to protect.
            return False
        if overridden_tools is not None:
            return bool(overridden_tools.value)
        if function_tools:
            return True
        if overridden_toolsets is not None:
            return bool(overridden_toolsets.value)
        return bool(user_toolsets or dynamic_toolsets)

    def _toolsets_for_run(self, *, include_scoped_tools: bool = True) -> list[Any] | None:
        toolsets: list[Any] = []
        scoped_tools = self.scoped_tools if include_scoped_tools else ()
        if scoped_tools:
            from pydantic_ai.toolsets import FunctionToolset

            toolsets.append(FunctionToolset(tools=scoped_tools).approval_required())
        toolsets.extend(
            toolset.approval_required()
            for toolset in self._mcp_toolsets_for_profile(self.active_profile)
        )
        return toolsets or None

    def _with_deferred_approval_capability(self, existing: object | None) -> list[Any]:
        from pydantic_ai.capabilities import HandleDeferredToolCalls

        capability = HandleDeferredToolCalls(handler=self._handle_deferred_tool_calls)
        if existing is None:
            return [capability]
        if isinstance(existing, (list, tuple)):
            return [*existing, capability]
        raise TypeError("Pydantic AI capabilities must be a list or tuple.")

    async def _handle_deferred_tool_calls(self, _context: Any, requests: Any) -> Any:
        from pydantic_ai.tools import ToolApproved, ToolDenied

        known_tools = {tool.name: tool for tool in self.scoped_tool_specs}
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for call in requests.approvals:
            try:
                arguments = call.args_as_dict()
            except ValueError as exc:
                approvals[call.tool_call_id] = ToolDenied(
                    f"Kedi could not parse arguments for tool {call.tool_name!r}: {exc}"
                )
                continue

            tool = known_tools.get(call.tool_name, _external_tool_spec(call.tool_name))
            request = approval_request_for_tool(tool, arguments, self.shortname)
            decision = await resolve_tool_approval_async(self.active_profile.approval, request)
            if decision.decision == "deny":
                suffix = f": {decision.reason}" if decision.reason else ""
                approvals[call.tool_call_id] = ToolDenied(f"Tool {tool.name!r} was denied{suffix}")
                continue

            validated_arguments = arguments_after_approval(tool, arguments, decision)
            approvals[call.tool_call_id] = ToolApproved(override_args=validated_arguments)
        return requests.build_results(approvals=approvals)

    def _tool_spec_scope_for_current_adapter(self) -> ToolScope:
        try:
            return self._tool_spec_scope
        except AttributeError:
            scope = ToolScope()
            self._tool_spec_scope = scope
            return scope

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
            result = tool.fn(**arguments)
            if inspect.isawaitable(result):
                return await result
            return result

        return validated_async_tool

    @wraps(tool.fn)
    def validated_tool(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")
        return tool.fn(**_validated_tool_arguments(tool, kwargs))

    return validated_tool


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
