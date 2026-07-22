from __future__ import annotations

import inspect
import json
import sys
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from importlib.util import find_spec
from typing import Any, Coroutine, Iterator, Mapping, Sequence, cast

from pydantic import BaseModel

from ...agent_profile import AgentProfile
from ...agent_profile.settings import CLAUDE_AGENT_SETTING_KEYS, claude_agent_settings
from ...dynamic_models import build_dynamic_model
from ...effort import pydantic_effort_value
from ...resolved_types import ResolvedKediType
from .._schema import output_field_definition
from ..approval import ToolRisk, resolve_tool_approval_async
from ..meta import AdapterCapabilities, AgentAdapterKind
from ..tool_override import (
    ToolScope,
    ToolSpec,
    approval_request_for_tool,
    arguments_target_secret_file,
    arguments_after_approval,
)

OutputDataT = Any
CLAUDE_CODE_TOOLS_PRESET: Mapping[str, str] = {
    "type": "preset",
    "preset": "claude_code",
}
CLAUDE_CODE_SYSTEM_PROMPT_PRESET: Mapping[str, str] = {
    "type": "preset",
    "preset": "claude_code",
}
CLAUDE_CODE_PERMISSION_MODE = "acceptEdits"
_CLAUDE_READ_ONLY_TOOLS = frozenset({"Read", "Glob", "Grep", "WebFetch", "WebSearch"})
_CLAUDE_KEDI_MCP_TOOL_PREFIX = "mcp__kedi__"


class ClaudeAdapter:
    """Adapter for Anthropic's Claude Agent SDK harness."""

    kind: AgentAdapterKind = "agent-harness"
    shortname = "claude"
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
        settings=CLAUDE_AGENT_SETTING_KEYS,
    )

    def __init__(
        self,
        model: str | None = None,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._settings = claude_agent_settings(model_settings or {})
        self._extra_kwargs = dict(kwargs)
        self._tool_scope = ToolScope()
        self._profile_stack_var: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_claude_profile_stack_{id(self)}",
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
    def type_builder(output_schema: dict[str, ResolvedKediType]) -> type:
        if not output_schema:
            raise ValueError("Output schema cannot be empty.")
        field_definitions = {
            field: output_field_definition(type_) for field, type_ in output_schema.items()
        }
        return build_dynamic_model("_OutputModel", BaseModel, field_definitions)

    async def produce(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        resolved_output_type = output_type
        if output_schema:
            resolved_output_type = cast(type[OutputDataT], self.type_builder(output_schema))
        if resolved_output_type is None:
            raise ValueError("Output spec must be provided.")

        sdk = _claude_sdk()
        instructions = cast(str | None, kwargs.pop("instructions", None))
        options = self._options_for_run(
            sdk,
            output_type=resolved_output_type,
            instructions=instructions,
            run_kwargs=kwargs,
        )
        result = None
        async for message in sdk.query(prompt=template, options=options):
            if _is_result_message(sdk, message):
                result = message

        if result is None:
            raise RuntimeError("Claude Agent SDK did not return a result message.")
        if bool(getattr(result, "is_error", False)):
            raise RuntimeError(str(getattr(result, "result", "Claude Agent SDK run failed.")))

        structured_output = getattr(result, "structured_output", None)
        if structured_output is None:
            raise ValueError("Claude Agent SDK did not return structured_output.")
        return _structured_output_value(structured_output, resolved_output_type)

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        sdk = _claude_sdk()
        instructions = cast(str | None, kwargs.pop("instructions", None))
        options = self._options_for_run(
            sdk,
            output_type=None,
            instructions=instructions,
            run_kwargs=kwargs,
        )
        result = None
        async for message in sdk.query(prompt=prompt, options=options):
            if _is_result_message(sdk, message):
                result = message

        if result is None:
            raise RuntimeError("Claude Agent SDK did not return a result message.")
        if bool(getattr(result, "is_error", False)):
            raise RuntimeError(str(getattr(result, "result", "Claude Agent SDK run failed.")))
        return _claude_text_response(result)

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        from ..utils import run_sync

        return run_sync(self.invoke(prompt=prompt, **kwargs))

    def produce_sync(
        self,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[OutputDataT] | None = None,
        **kwargs: Any,
    ) -> OutputDataT:
        from ..utils import run_sync

        return run_sync(
            self.produce(template, output_schema=output_schema, output_type=output_type, **kwargs)
        )

    def default_model(self) -> str | None:
        return self._model

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        stack_var = self._profile_stack_var
        token = stack_var.set((*stack_var.get(), profile))
        try:
            yield
        finally:
            stack_var.reset(token)

    def _model_for_run(self) -> str | None:
        return self.active_profile.model or self._model

    def _settings_for_run(self) -> dict[str, Any]:
        settings = {
            **dict(self._extra_kwargs),
            **dict(self._settings),
            **claude_agent_settings(self.active_profile.settings),
        }
        if self.active_profile.effort is not None:
            settings["effort"] = pydantic_effort_value(self.active_profile.effort)
        return settings

    def _options_for_run(
        self,
        sdk: Any,
        *,
        output_type: type[Any] | None,
        instructions: str | None,
        run_kwargs: Mapping[str, Any],
    ) -> Any:
        options_kwargs = {
            **self._settings_for_run(),
            **dict(run_kwargs),
        }
        options_kwargs.setdefault("tools", dict(CLAUDE_CODE_TOOLS_PRESET))
        options_kwargs.setdefault("permission_mode", CLAUDE_CODE_PERMISSION_MODE)
        if output_type is not None:
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": _output_json_schema(output_type),
            }
        model = self._model_for_run()
        if model is not None:
            options_kwargs["model"] = model
        system_prompt = instructions or self.active_profile.system
        if system_prompt is not None:
            options_kwargs.setdefault(
                "system_prompt",
                {
                    **CLAUDE_CODE_SYSTEM_PROMPT_PRESET,
                    "append": system_prompt,
                },
            )
        else:
            options_kwargs.setdefault("system_prompt", dict(CLAUDE_CODE_SYSTEM_PROMPT_PRESET))

        mcp_servers, allowed_tools = self._mcp_servers_for_run(sdk)
        if mcp_servers:
            existing_servers = options_kwargs.get("mcp_servers")
            if isinstance(existing_servers, Mapping):
                options_kwargs["mcp_servers"] = {
                    **dict(existing_servers),
                    **mcp_servers,
                }
            else:
                options_kwargs["mcp_servers"] = mcp_servers
        if allowed_tools:
            existing_allowed = options_kwargs.get("allowed_tools")
            if isinstance(existing_allowed, (list, tuple)):
                options_kwargs["allowed_tools"] = [*existing_allowed, *allowed_tools]
            else:
                options_kwargs["allowed_tools"] = allowed_tools
        options_kwargs["hooks"] = _with_kedi_approval_hook(
            sdk,
            options_kwargs.get("hooks"),
            profile=self.active_profile,
            tools=self.scoped_tools,
        )
        options_cls = getattr(sdk, "ClaudeAgentOptions")
        return options_cls(**options_kwargs)

    def _mcp_servers_for_run(self, sdk: Any) -> tuple[dict[str, Any], list[str]]:
        mcp_servers = _claude_external_mcp_servers(self.active_profile)
        allowed_tools: list[str] = []
        if self.scoped_tools:
            mcp_servers["kedi"] = _claude_sdk_mcp_server(sdk, self.scoped_tools)
            allowed_tools.extend(f"mcp__kedi__{tool.name}" for tool in self.scoped_tools)
        return mcp_servers, allowed_tools


def _claude_sdk() -> Any:
    if "claude_agent_sdk" in sys.modules:
        return sys.modules["claude_agent_sdk"]
    if find_spec("claude_agent_sdk") is not None:
        return __import__("claude_agent_sdk")
    raise ModuleNotFoundError(
        "ClaudeAdapter requires claude-agent-sdk. In a Kedi source checkout, run "
        "`uv sync --extra claude`; for an installed package, run "
        "`uv pip install 'kedi[claude]'`."
    )


def _output_json_schema(output_type: type[Any]) -> dict[str, Any]:
    if issubclass(output_type, BaseModel):
        return output_type.model_json_schema()
    return {"type": _json_schema_type_name(output_type)}


def _json_schema_type_name(output_type: type[Any]) -> str:
    if output_type is str:
        return "string"
    if output_type is int:
        return "integer"
    if output_type is float:
        return "number"
    if output_type is bool:
        return "boolean"
    return "object"


def _structured_output_value(value: Any, output_type: type[Any]) -> Any:
    if issubclass(output_type, BaseModel):
        if isinstance(value, output_type):
            return value
        return output_type.model_validate(value)
    return value


def _claude_text_response(message: Any) -> str:
    result = getattr(message, "result", None)
    if isinstance(result, str):
        return result
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(result, Mapping):
        text = result.get("text")
        if isinstance(text, str):
            return text
    raise ValueError("Claude Agent SDK did not return a text response.")


def _is_result_message(sdk: Any, message: Any) -> bool:
    result_cls = getattr(sdk, "ResultMessage", None)
    if result_cls is not None and isinstance(message, result_cls):
        return True
    return hasattr(message, "structured_output") or hasattr(message, "is_error")


def _claude_sdk_mcp_server(sdk: Any, tools: Sequence[ToolSpec]) -> Any:
    sdk_tools = [_claude_sdk_tool(sdk, tool) for tool in tools]
    create_server = getattr(sdk, "create_sdk_mcp_server")
    return create_server(name="kedi", tools=sdk_tools)


def _claude_sdk_tool(sdk: Any, tool: ToolSpec) -> Any:
    decorator = getattr(sdk, "tool")
    json_schema = tool.json_schema or {"type": "object", "properties": {}}
    description = tool.description or tool.name

    async def _call(args: dict[str, Any]) -> dict[str, Any]:
        arguments = _validated_tool_arguments(tool, args)
        result = tool.fn(**arguments)
        if inspect.isawaitable(result):
            result = await cast(Coroutine[Any, Any, Any], result)
        return {
            "content": [
                {
                    "type": "text",
                    "text": _tool_result_text(result),
                }
            ]
        }

    _call.__name__ = tool.name
    _call.__doc__ = description
    return decorator(name=tool.name, description=description, input_schema=json_schema)(_call)


def _with_kedi_approval_hook(
    sdk: Any,
    existing_hooks: object,
    *,
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
) -> dict[str, list[Any]]:
    """Append a PreToolUse policy bridge without discarding caller-provided hooks."""

    if existing_hooks is None:
        hooks: dict[str, list[Any]] = {}
    elif isinstance(existing_hooks, Mapping):
        hooks = {}
        for event, matchers in existing_hooks.items():
            if not isinstance(matchers, (list, tuple)):
                raise TypeError("Claude hook matchers must be lists or tuples.")
            hooks[str(event)] = list(matchers)
    else:
        raise TypeError("Claude hooks must be a mapping of hook events to matcher lists.")

    matcher_cls = _claude_hook_matcher_class(sdk)
    hooks.setdefault("PreToolUse", []).append(
        matcher_cls(
            matcher=None,
            hooks=[_claude_pre_tool_use_hook(profile=profile, tools=tools)],
        )
    )
    return hooks


def _claude_hook_matcher_class(sdk: Any) -> Any:
    """Return the public Claude SDK hook matcher type."""

    matcher_cls = getattr(sdk, "HookMatcher", None)
    if matcher_cls is None:
        raise RuntimeError("claude-agent-sdk does not expose its public HookMatcher type")
    return matcher_cls


def _claude_pre_tool_use_hook(
    *,
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
) -> Any:
    """Build a Claude hook that applies Kedi policy to every native tool call."""

    known_tools = _claude_tool_specs_by_native_name(tools)

    async def decide(hook_input: Any, _tool_use_id: str | None, _context: Any) -> dict[str, Any]:
        if not isinstance(hook_input, Mapping):
            return _claude_hook_deny("Claude supplied invalid PreToolUse hook input.")
        tool_name = hook_input.get("tool_name")
        arguments = hook_input.get("tool_input")
        if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
            return _claude_hook_deny("Claude supplied an incomplete tool permission request.")

        tool = known_tools.get(tool_name, _claude_native_tool_spec(tool_name))
        try:
            original_arguments = _validated_tool_arguments(tool, arguments)
            request = approval_request_for_tool(tool, original_arguments, "claude")
            decision = await resolve_tool_approval_async(profile.approval, request)
            if decision.decision == "deny":
                return _claude_hook_deny(decision.reason or f"Kedi denied tool {tool.name!r}.")
            final_arguments = arguments_after_approval(tool, original_arguments, decision)
        except Exception as exc:
            return _claude_hook_deny(f"Kedi rejected tool {tool.name!r}: {exc}")

        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": final_arguments,
            }
        }

    return decide


def _claude_hook_deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _claude_tool_specs_by_native_name(tools: Sequence[ToolSpec]) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    for tool in tools:
        specs[tool.name] = tool
        specs[f"{_CLAUDE_KEDI_MCP_TOOL_PREFIX}{tool.name}"] = tool
    return specs


def _claude_native_tool_spec(name: str) -> ToolSpec:
    risk: ToolRisk = "read_only" if name in _CLAUDE_READ_ONLY_TOOLS else "mutating"
    return ToolSpec(
        fn=lambda **_kwargs: None,
        name=name,
        description="Claude native or MCP tool",
        metadata={"source": "claude-native"},
        risk=risk,
        risk_resolver=_claude_native_tool_risk,
    )


def _claude_native_tool_risk(arguments: Mapping[str, Any]) -> ToolRisk:
    return "sensitive" if arguments_target_secret_file(arguments) else "read_only"


def _validated_tool_arguments(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if tool.argument_validator is None:
        return dict(arguments)
    return tool.argument_validator(arguments)


def _tool_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _claude_external_mcp_servers(profile: AgentProfile) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(profile.mcp_servers):
        name = f"kedi_mcp_{index}"
        config: dict[str, Any]
        if spec.transport == "stdio":
            if spec.command is None:
                raise ValueError("MCP stdio transport requires a command.")
            config = {
                "type": "stdio",
                "command": spec.command,
                "args": list(spec.args),
            }
            if spec.env is not None:
                config["env"] = spec.env
        elif spec.transport == "sse":
            if spec.url is None:
                raise ValueError("MCP sse transport requires a url.")
            config = {"type": "sse", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        elif spec.transport == "http":
            if spec.url is None:
                raise ValueError("MCP http transport requires a url.")
            config = {"type": "http", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        else:
            raise ValueError(f"Unsupported MCP transport: {spec.transport}")
        configs[name] = config
    return configs
