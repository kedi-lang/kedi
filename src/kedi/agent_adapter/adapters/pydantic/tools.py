from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Sequence,
    cast,
    get_type_hints,
)

from pydantic_ai import (
    ModelRetry,
)
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

from kedi.agent_adapter.adapters.pydantic.compat import (
    application_toolset_count,
    application_toolsets,
    build_base_toolset_list,
)
from kedi.agent_adapter.approvals.policy import (
    ApprovalDecision,
    ApprovalRequest,
    ToolApprovalDenied,
    ToolRisk,
)
from kedi.agent_adapter.approvals.reason import accepts_tool_reason, split_tool_reason
from kedi.agent_adapter.tools.execution import (
    approval_request_for_tool,
    arguments_after_approval,
    arguments_target_secret_file,
    resolve_tool_arguments_async,
    with_required_before_output,
)
from kedi.agent_adapter.tools.scope import (
    current_required_tool_calls,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile

if TYPE_CHECKING:
    from kedi.agent_adapter.adapters.pydantic.adapter import PydanticAdapter

PydanticApprovalResolution = Literal["kedi", "external"]


KEDI_APPROVAL_METADATA_KEY = "kedi_approval"


_DEFAULT_TOOL_RETRIES = 3


class _KediPydanticTool(Tool[Any]):
    """A native Pydantic tool that retains its adapter-neutral Kedi specification."""

    __slots__ = ("kedi_spec",)

    kedi_spec: ToolSpec


def _mcp_toolset_from_config(name: str, server_config: dict[str, object]) -> Any:
    from pydantic_ai.mcp import MCPToolset

    config = {"mcpServers": {name: server_config}}
    return MCPToolset(cast(Any, config))


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

        return _preserve_tool_introspection(validated_async_tool, tool.fn)

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

    return _preserve_tool_introspection(validated_tool, tool.fn)


def _preserve_tool_introspection(wrapper: Any, function: Callable[..., Any]) -> Any:
    """Keep wrapped tool signatures introspectable across Python annotation models."""

    signature = inspect.signature(function)
    setattr(wrapper, "__signature__", signature)
    required = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
    }
    if signature.return_annotation is not inspect.Signature.empty:
        required.add("return")

    annotations: dict[str, Any] = {}
    current: Any = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        try:
            resolved = get_type_hints(current, include_extras=True)
        except (NameError, TypeError):
            resolved = inspect.get_annotations(current, eval_str=False)
        for name, annotation in resolved.items():
            annotations.setdefault(name, annotation)
        if required.issubset(annotations):
            break
        current = getattr(current, "__wrapped__", None)

    for name, parameter in signature.parameters.items():
        if parameter.annotation is not inspect.Parameter.empty:
            annotations.setdefault(name, parameter.annotation)
    if signature.return_annotation is not inspect.Signature.empty:
        annotations.setdefault("return", signature.return_annotation)
    wrapper.__annotations__ = annotations
    return wrapper


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
        reason=payload.get("reason"),
        tool_reason_enabled=payload.get("tool_reason_enabled", False),
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


def _approval_request_payload(request: ApprovalRequest) -> dict[str, Any]:
    return {
        "tool_name": request.tool_name,
        "arguments": dict(request.arguments),
        "risk": request.risk,
        "adapter_shortname": request.adapter_shortname,
        "description": request.description,
        "metadata": dict(request.metadata) if request.metadata is not None else None,
        "reason": request.reason,
        "tool_reason_enabled": request.tool_reason_enabled,
    }


def _model_retry_for_tool_error(tool: ToolSpec, exc: Exception) -> Exception | None:

    from kedi.agent_adapter.tools.errors import correctable_tool_error

    message = correctable_tool_error(tool, exc)
    return ModelRetry(message) if message is not None else None


def _validated_tool_arguments(tool: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if tool.argument_validator is None:
        return dict(arguments)
    return tool.argument_validator(arguments)


def _is_async_callable(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


def _external_tool_spec(
    name: str,
    tool_def: ToolDefinition | None = None,
) -> ToolSpec:
    """Describe an MCP/external tool whose implementation stays inside Pydantic AI."""

    metadata = dict(tool_def.metadata or {}) if tool_def is not None else {}
    metadata["source"] = "external"
    return ToolSpec(
        fn=lambda **_kwargs: None,
        name=name,
        description=(
            tool_def.description
            if tool_def is not None and tool_def.description is not None
            else "External Pydantic AI tool"
        ),
        json_schema=(
            dict(tool_def.parameters_json_schema) if tool_def is not None else {"type": "object"}
        ),
        return_schema=(
            None
            if tool_def is None or tool_def.return_schema is None
            else dict(tool_def.return_schema)
        ),
        metadata=metadata,
        risk="mutating",
        risk_resolver=_external_tool_risk,
        sequential=False if tool_def is None else tool_def.sequential,
    )


def _approval_required_toolset(
    toolset: Any,
    *,
    adapter_shortname: str = "pydantic",
) -> Any:
    if isinstance(toolset, ApprovalRequiredToolset):
        return toolset
    if not hasattr(toolset, "apply"):
        return toolset.approval_required()
    tool_specs = {tool.name: tool for tool in _kedi_tool_specs_from_toolsets((toolset,))}
    if not tool_specs:
        return toolset.approval_required()

    def requires_approval(
        ctx: RunContext[Any],
        tool_def: ToolDefinition,
        tool_args: dict[str, Any],
    ) -> bool:
        del ctx
        tool = tool_specs.get(tool_def.name)
        if tool is None:
            return True
        try:
            request = approval_request_for_tool(
                tool,
                tool_args,
                adapter_shortname,
            )
        except Exception:
            return True
        return request.risk != "read_only"

    return toolset.approval_required(requires_approval)


def _external_tool_risk(arguments: Mapping[str, Any]) -> ToolRisk:
    return "sensitive" if arguments_target_secret_file(arguments) else "mutating"


def _add_nested_native_toolsets(
    self: PydanticAdapter[Any, Any], run_kwargs: dict[str, object]
) -> None:
    if not self._has_native_approval_tools():
        return
    native_toolsets = [
        _approval_required_toolset(
            toolset,
            adapter_shortname=self.shortname,
        )
        for toolset in application_toolsets(self)
    ]
    existing_toolsets = run_kwargs.get("toolsets")
    if existing_toolsets is None:
        run_kwargs["toolsets"] = native_toolsets
        return
    if isinstance(existing_toolsets, (str, bytes)) or not isinstance(existing_toolsets, Sequence):
        raise TypeError("Pydantic AI toolsets must be a sequence.")
    run_kwargs["toolsets"] = [*existing_toolsets, *native_toolsets]


def _build_toolset_list(
    self: PydanticAdapter[Any, Any],
    cap_toolsets: Sequence[Any] | None = None,
    *,
    ignore_overrides: bool = False,
) -> list[Any]:
    """Require native approval for every application-provided Pydantic toolset."""

    toolsets = build_base_toolset_list(self, cap_toolsets, ignore_overrides=ignore_overrides)
    toolset_count = application_toolset_count(self, ignore_overrides=ignore_overrides)
    return [
        (
            _approval_required_toolset(
                toolset,
                adapter_shortname=self.shortname,
            )
            if index < toolset_count
            else toolset
        )
        for index, toolset in enumerate(toolsets)
    ]


def _toolsets_for_run(
    self: PydanticAdapter[Any, Any], *, include_scoped_tools: bool = True
) -> list[Any] | None:
    toolsets: list[Any] = []
    scoped_tools = self.scoped_tools if include_scoped_tools else ()
    if scoped_tools:
        toolsets.append(
            _approval_required_toolset(
                FunctionToolset(tools=scoped_tools),
                adapter_shortname=self.shortname,
            )
        )
    toolsets.extend(
        _approval_required_toolset(
            toolset,
            adapter_shortname=self.shortname,
        )
        for toolset in self._mcp_toolsets_for_profile(self.active_profile)
    )
    return toolsets or None


def _tool_specs_for_run(
    self: PydanticAdapter[Any, Any], run_kwargs: Mapping[str, Any]
) -> tuple[ToolSpec, ...]:
    from kedi.tool_context import inside_kedi_tool_call

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


async def _handle_deferred_tool_calls(
    self: PydanticAdapter[Any, Any], _context: Any, requests: Any
) -> Any:
    approvals: dict[str, ToolApproved | ToolDenied] = {}
    for call in requests.approvals:
        try:
            arguments = call.args_as_dict()
        except ValueError as exc:
            approvals[call.tool_call_id] = ToolDenied(
                f"Kedi could not parse arguments for tool {call.tool_name!r}: {exc}"
            )
            continue

        tool, _origin = self._hook_tool_for_call_id(
            call.tool_call_id,
            (self._tool_spec_for_call(call.tool_name), "adapter_native"),
        )
        arguments = self._hook_arguments_for_call(call.tool_call_id, arguments)
        reason_enabled = self.active_tool_reason
        if reason_enabled and accepts_tool_reason(tool):
            arguments, _ = split_tool_reason(arguments)
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
                    reason=(
                        self._tool_reasons_var.get().get(call.tool_call_id)
                        if reason_enabled
                        else None
                    ),
                    tool_reason_enabled=reason_enabled,
                )
            except ToolApprovalDenied as exc:
                message = str(exc.args[0]) if exc.args else f"Tool {tool.name!r} was denied"
                approvals[call.tool_call_id] = ToolDenied(message)
                break
        else:
            approvals[call.tool_call_id] = ToolApproved(override_args=approved_arguments)
    return requests.build_results(approvals=approvals)


def _mcp_toolsets_for_profile(self: PydanticAdapter[Any, Any], profile: AgentProfile) -> list[Any]:
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


def _validate_required_tool_calls(output: Any) -> Any:
    state = current_required_tool_calls()
    if state is None or state.names <= state.returned:
        return output

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
