from __future__ import annotations

import inspect
import json
import os
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar, copy_context
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence, cast, get_type_hints

from pydantic import BaseModel, create_model

from ...agent_profile import AgentProfile
from ...agent_profile.settings import CODEX_AGENT_SETTING_KEYS, codex_agent_settings
from ...dynamic_models import build_dynamic_model
from ...effort import pydantic_effort_value
from ...resolved_types import ResolvedKediType
from .._schema import output_field_definition
from ..approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ToolRisk,
    resolve_tool_approval_async,
)
from ..meta import AdapterCapabilities, AgentAdapterKind
from ..tool_override import ToolScope, ToolSpec, arguments_target_secret_file, with_approval
from ._codex_app_server import (
    CodexAppServerClient,
    CodexAppServerProtocolError,
    CodexTurnResult,
    ServerApprovalHandler,
)

OutputDataT = Any
CODEX_DEFAULT_SANDBOX = "workspace-write"
CODEX_SUPPORTED_STRING_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "time",
    }
)


@dataclass(frozen=True)
class _CodexRun:
    thread_params: dict[str, Any]
    turn_params: dict[str, Any]
    tools: dict[str, Any]
    native_approval_handler: ServerApprovalHandler
    timeout: float
    output_type: type[Any] | None = None


class CodexAdapter:
    """Stateless Kedi adapter backed directly by Codex App Server JSON-RPC."""

    kind: AgentAdapterKind = "agent-harness"
    shortname = "codex"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=False,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_native_approvals=True,
        supports_native_approval_handler=False,
        supported_json_schema_formats=CODEX_SUPPORTED_STRING_FORMATS,
        settings=CODEX_AGENT_SETTING_KEYS,
    )

    def __init__(
        self,
        model: str | None = None,
        *,
        model_settings: Mapping[str, Any] | None = None,
        command: Sequence[str] = ("codex", "app-server", "--stdio"),
        env: Mapping[str, str] | None = None,
        request_timeout: float = 30,
        turn_timeout: float = 300,
        client: CodexAppServerClient | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._settings = {
            **codex_agent_settings(model_settings or {}),
            **self._validated_settings(kwargs),
        }
        self._turn_timeout = turn_timeout
        self._client = client or CodexAppServerClient(
            command,
            env=env,
            request_timeout=request_timeout,
        )
        self._tool_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_codex_profile_stack_{id(self)}",
            default=(),
        )

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tool_scope.register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[ToolSpec, ...]:
        return cast(tuple[ToolSpec, ...], self._tool_scope.scoped_tools)

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack_var.get()
        return stack[-1] if stack else AgentProfile.empty()

    @staticmethod
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type[Any]:
        if not output_schema:
            raise ValueError("Output schema cannot be empty.")
        fields = {name: output_field_definition(type_) for name, type_ in output_schema.items()}
        return build_dynamic_model("_OutputModel", BaseModel, fields)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        run = self._prepare_run(template, output_schema, output_type, kwargs)
        result = await self._client.run_stateless_turn(
            thread_params=run.thread_params,
            turn_params=run.turn_params,
            tools=run.tools,
            tool_context=copy_context(),
            turn_timeout=run.timeout,
            approval_handler=run.native_approval_handler,
            approval_context=copy_context(),
        )
        return _parse_structured_result(result, cast(type[Any], run.output_type))

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        run = self._prepare_invoke(prompt, kwargs)
        result = await self._client.run_stateless_turn(
            thread_params=run.thread_params,
            turn_params=run.turn_params,
            tools=run.tools,
            tool_context=copy_context(),
            turn_timeout=run.timeout,
            approval_handler=run.native_approval_handler,
            approval_context=copy_context(),
        )
        return result.text

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        run = self._prepare_run(template, output_schema, output_type, kwargs)
        result = self._client.run_stateless_turn_sync(
            thread_params=run.thread_params,
            turn_params=run.turn_params,
            tools=run.tools,
            tool_context=copy_context(),
            turn_timeout=run.timeout,
            approval_handler=run.native_approval_handler,
            approval_context=copy_context(),
        )
        return _parse_structured_result(result, cast(type[Any], run.output_type))

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        run = self._prepare_invoke(prompt, kwargs)
        result = self._client.run_stateless_turn_sync(
            thread_params=run.thread_params,
            turn_params=run.turn_params,
            tools=run.tools,
            tool_context=copy_context(),
            turn_timeout=run.timeout,
            approval_handler=run.native_approval_handler,
            approval_context=copy_context(),
        )
        return result.text

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._client.aclose()

    def default_model(self) -> str | None:
        return self._model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        token = self._profile_stack_var.set((*self._profile_stack_var.get(), profile))
        try:
            yield
        finally:
            self._profile_stack_var.reset(token)

    def _prepare_run(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None,
        output_type: type[OutputDataT] | None,
        kwargs: Mapping[str, Any],
    ) -> _CodexRun:
        resolved_output_type: type[Any] | None = output_type
        if output_schema:
            resolved_output_type = self.type_builder(output_schema)
        if resolved_output_type is None:
            raise ValueError("Output spec must be provided.")
        return self._prepare_turn(
            template,
            kwargs,
            output_schema=_output_json_schema(resolved_output_type),
            output_type=resolved_output_type,
        )

    def _prepare_invoke(
        self,
        prompt: str,
        kwargs: Mapping[str, Any],
    ) -> _CodexRun:
        return self._prepare_turn(prompt, kwargs, output_schema=None, output_type=None)

    def _prepare_turn(
        self,
        prompt: str,
        kwargs: Mapping[str, Any],
        *,
        output_schema: dict[str, Any] | None,
        output_type: type[Any] | None,
    ) -> _CodexRun:
        profile = self.active_profile
        if profile.mcp_servers:
            raise NotImplementedError(
                "CodexAdapter does not yet support Kedi MCP server declarations."
            )

        run_kwargs = dict(kwargs)
        instructions = run_kwargs.pop("instructions", None)
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("Codex instructions must be a string.")
        settings = {
            **self._settings,
            **codex_agent_settings(profile.settings),
            **self._validated_settings(run_kwargs),
        }
        model = profile.model or self._model
        system = instructions if instructions is not None else profile.system
        approved_tools = tuple(
            with_approval(
                tool,
                policy=profile.approval,
                adapter_shortname=self.shortname,
            )
            for tool in self.scoped_tools
        )
        tools = {tool.name: tool.fn for tool in approved_tools}
        dynamic_tools = [_dynamic_tool_spec(tool) for tool in approved_tools]

        thread_params: dict[str, Any] = {
            "ephemeral": True,
            "approvalPolicy": (
                "untrusted"
                if profile.approval is not None
                # Leave the App Server in approval-requesting mode by default so
                # the Kedi bridge can enforce the same default-safe policy for
                # Codex builtin and MCP tools as it does for Kedi ToolSpecs.
                # An explicit Codex setting remains an intentional opt-out.
                else settings.get("approval_policy", "untrusted")
            ),
            "sandbox": _codex_sandbox_value(settings.get("sandbox", CODEX_DEFAULT_SANDBOX)),
            "cwd": str(settings.get("cwd", os.getcwd())),
            "serviceName": "kedi",
        }
        if model is not None:
            thread_params["model"] = model
        if system is not None:
            thread_params["developerInstructions"] = system
        if dynamic_tools:
            thread_params["dynamicTools"] = dynamic_tools
        _copy_setting(settings, "model_provider", thread_params, "modelProvider")
        _copy_setting(settings, "personality", thread_params, "personality")
        _copy_setting(settings, "service_tier", thread_params, "serviceTier")
        _copy_setting(settings, "config", thread_params, "config")

        turn_params: dict[str, Any] = {
            "input": [{"type": "text", "text": prompt}],
        }
        if output_schema is not None:
            turn_params["outputSchema"] = output_schema
        if profile.effort is not None:
            turn_params["effort"] = pydantic_effort_value(profile.effort)
        _copy_setting(settings, "summary", turn_params, "summary")
        timeout = settings.get("timeout", self._turn_timeout)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("Codex timeout must be a positive number.")
        return _CodexRun(
            thread_params=thread_params,
            turn_params=turn_params,
            tools=tools,
            native_approval_handler=_codex_native_approval_handler(profile.approval),
            timeout=float(timeout),
            output_type=output_type,
        )

    @staticmethod
    def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(settings).difference(CODEX_AGENT_SETTING_KEYS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unsupported Codex setting(s): {names}")
        return dict(settings)


def _dynamic_tool_spec(tool: ToolSpec) -> dict[str, Any]:
    input_schema = tool.json_schema or _callable_json_schema(tool)
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or tool.name,
        "inputSchema": _codex_json_schema(input_schema),
    }


def _callable_json_schema(tool: ToolSpec) -> dict[str, Any]:
    signature = inspect.signature(tool.fn)
    resolved_hints = True
    try:
        hints = get_type_hints(tool.fn)
    except (NameError, TypeError):
        resolved_hints = False
        hints = {}
    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError(f"Codex dynamic tool {tool.name!r} cannot use variadic parameters.")
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"Codex dynamic tool {tool.name!r} cannot use positional-only parameters."
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if not resolved_hints and isinstance(annotation, str):
            annotation = Any
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = parameter.default
        if default is inspect.Parameter.empty:
            default = ...
        fields[parameter.name] = (annotation, default)
    model = create_model(f"_{tool.name}_CodexToolInput", **cast(dict[str, Any], fields))
    return model.model_json_schema()


def _output_json_schema(output_type: type[Any]) -> dict[str, Any]:
    if issubclass(output_type, BaseModel):
        return _codex_json_schema(output_type.model_json_schema())
    if output_type is str:
        return {"type": "string"}
    if output_type is int:
        return {"type": "integer"}
    if output_type is float:
        return {"type": "number"}
    if output_type is bool:
        return {"type": "boolean"}
    return {"type": "object", "additionalProperties": False}


def _parse_structured_result(result: CodexTurnResult, output_type: type[Any]) -> Any:
    try:
        value = json.loads(result.text)
    except json.JSONDecodeError as exc:
        raise CodexAppServerProtocolError(
            f"Codex returned invalid structured output: {result.text}"
        ) from exc
    if issubclass(output_type, BaseModel):
        return output_type.model_validate(value)
    return value


def _codex_native_approval_handler(policy: ApprovalPolicy | None) -> ServerApprovalHandler:
    async def decide(method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        request = _codex_native_approval_request(method, params)
        decision = await resolve_tool_approval_async(policy, request)
        return _codex_native_approval_response(method, params, decision)

    return decide


def _codex_native_approval_request(
    method: str,
    params: Mapping[str, Any],
) -> ApprovalRequest:
    tool_name, risk, description = _codex_native_approval_details(method, params)
    return ApprovalRequest.create(
        tool_name=tool_name,
        arguments=params,
        risk=risk,
        adapter_shortname="codex",
        description=description,
        metadata={"source": "codex-app-server", "method": method},
    )


def _codex_native_approval_details(
    method: str,
    params: Mapping[str, Any],
) -> tuple[str, ToolRisk, str]:
    if method == "item/commandExecution/requestApproval":
        return (
            "codex.command_execution",
            _codex_command_risk(params),
            "Codex requested approval to execute a shell command.",
        )
    if method == "item/fileChange/requestApproval":
        risk: ToolRisk = "sensitive" if arguments_target_secret_file(params) else "mutating"
        return (
            "codex.file_change",
            risk,
            "Codex requested approval to change files.",
        )
    if method == "item/permissions/requestApproval":
        return (
            "codex.permissions",
            "sensitive",
            "Codex requested additional filesystem or network permissions.",
        )
    raise ValueError(f"Unsupported Codex native approval method: {method}")


def _codex_command_risk(params: Mapping[str, Any]) -> ToolRisk:
    if arguments_target_secret_file(params):
        return "sensitive"
    if any(
        params.get(name) is not None
        for name in (
            "additionalPermissions",
            "networkApprovalContext",
            "proposedExecpolicyAmendment",
            "proposedNetworkPolicyAmendments",
        )
    ):
        return "sensitive"
    actions = params.get("commandActions")
    if (
        isinstance(actions, list)
        and actions
        and all(_is_read_only_command_action(action) for action in actions)
    ):
        return "read_only"
    return "mutating"


def _is_read_only_command_action(action: object) -> bool:
    if not isinstance(action, Mapping):
        return False
    return cast(Mapping[str, Any], action).get("type") in {
        "read",
        "listFiles",
        "search",
    }


def _codex_native_approval_response(
    method: str,
    params: Mapping[str, Any],
    decision: ApprovalDecision,
) -> dict[str, Any]:
    if decision.decision != "allow":
        # App Server approval requests cannot carry Kedi's edited arguments. Declining is the
        # only safe behavior for an edit decision as well as an explicit denial.
        if method == "item/permissions/requestApproval":
            return {"permissions": {}, "scope": "turn"}
        return {"decision": "decline"}
    if method == "item/permissions/requestApproval":
        permissions = params.get("permissions")
        if not isinstance(permissions, Mapping):
            return {"permissions": {}, "scope": "turn"}
        return {"permissions": dict(permissions), "scope": "turn"}
    return {"decision": "accept"}


def _copy_setting(
    settings: Mapping[str, Any],
    source: str,
    target: dict[str, Any],
    destination: str,
) -> None:
    if source in settings:
        target[destination] = settings[source]


def _codex_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(schema))

    def visit(node: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            schema_format = node.get("format")
            if (
                node.get("type") == "string"
                and isinstance(schema_format, str)
                and schema_format not in CODEX_SUPPORTED_STRING_FORMATS
            ):
                location = ".".join(path) or "<root>"
                raise TypeError(
                    f"Codex does not support JSON Schema format {schema_format!r} "
                    f"at {location}. Choose a different Kedi type for this output "
                    'or tool field. Use Annotated[str, "field description"] when '
                    "a string with semantic guidance is sufficient."
                )
            for key, child in node.items():
                visit(child, (*path, str(key)))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, (*path, str(index)))

    visit(value)
    return value


def _codex_sandbox_value(value: Any) -> str:
    aliases = {
        "readOnly": "read-only",
        "read_only": "read-only",
        "read-only": "read-only",
        "workspaceWrite": "workspace-write",
        "workspace_write": "workspace-write",
        "workspace-write": "workspace-write",
        "dangerFullAccess": "danger-full-access",
        "danger_full_access": "danger-full-access",
        "danger-full-access": "danger-full-access",
    }
    if isinstance(value, str) and value in aliases:
        return aliases[value]
    raise ValueError(
        "Codex sandbox must be one of read-only, workspace-write, or danger-full-access."
    )
