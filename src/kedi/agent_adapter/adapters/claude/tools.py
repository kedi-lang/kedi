from __future__ import annotations

import inspect
import json
import sys
from dataclasses import asdict, is_dataclass
from importlib.util import find_spec
from typing import Any, Coroutine, Mapping, Sequence, cast

from pydantic import BaseModel

from kedi.agent_adapter.approvals.policy import ToolRisk
from kedi.agent_adapter.subagents.runs import safe_subagent_error_message
from kedi.agent_adapter.tools.execution import arguments_target_secret_file
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile
from kedi.errors import KediExecutionError


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


def _claude_sdk_mcp_server(sdk: Any, tools: Sequence[ToolSpec]) -> Any:
    sdk_tools = [_claude_sdk_tool(sdk, tool) for tool in tools]
    create_server = getattr(sdk, "create_sdk_mcp_server")
    return create_server(name="kedi", tools=sdk_tools)


def _claude_sdk_tool(sdk: Any, tool: ToolSpec) -> Any:
    from kedi.artifacts.errors import ArtifactError, artifact_tool_error_message

    decorator = getattr(sdk, "tool")
    json_schema = tool.json_schema or {"type": "object", "properties": {}}
    description = tool.description or tool.name

    async def _call(args: dict[str, Any]) -> dict[str, Any]:
        arguments = _validated_tool_arguments(tool, args)
        try:
            result = tool.fn(**arguments)
            if inspect.isawaitable(result):
                result = await cast(Coroutine[Any, Any, Any], result)
        except KediExecutionError as exc:
            if tool.metadata is None or tool.metadata.get("kind") != "subagent":
                raise
            return {
                "content": [
                    {
                        "type": "text",
                        "text": safe_subagent_error_message(exc) or "Subagent failed",
                    }
                ],
                "is_error": True,
            }
        except ArtifactError as exc:
            metadata = tool.metadata or {}
            if (metadata.get("kedi_tool_kind") or metadata.get("kind")) != "artifact_management":
                raise
            message = artifact_tool_error_message(exc)
            if message is None:
                raise
            return {"content": [{"type": "text", "text": message}], "is_error": True}
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


def _claude_tool_call_input(
    hook_input: Any,
    event: str,
) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(hook_input, Mapping):
        raise TypeError(f"Claude supplied invalid {event} hook input")
    tool_name = hook_input.get("tool_name")
    arguments = hook_input.get("tool_input")
    if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
        raise TypeError(f"Claude supplied incomplete {event} hook input")
    return tool_name, arguments


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


_CLAUDE_READ_ONLY_TOOLS = frozenset({"Read", "Glob", "Grep", "WebFetch", "WebSearch"})


_CLAUDE_KEDI_MCP_TOOL_PREFIX = "mcp__kedi__"
