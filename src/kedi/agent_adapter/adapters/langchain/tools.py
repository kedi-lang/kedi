from __future__ import annotations

import inspect
from dataclasses import replace
from functools import wraps
from typing import (
    Any,
    Callable,
    Coroutine,
    Mapping,
    cast,
)

from kedi.agent_adapter.adapters.langchain.messages import _raise_langchain_codemode_error
from kedi.agent_adapter.approvals.policy import ToolRisk
from kedi.agent_adapter.codemode import CodeModeError
from kedi.agent_adapter.codemode.tool_specs import CodeModeProjection
from kedi.agent_adapter.subagents.tool_activity import with_live_tool_tracking
from kedi.agent_adapter.tools.errors import correctable_tool_error
from kedi.agent_adapter.tools.execution import arguments_target_secret_file
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile


def _langchain_tool_from_spec(tool: ToolSpec) -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChainAdapter tool support requires langchain-core. "
            "Install with: uv add 'kedi[langchain]'"
        ) from exc

    description = tool.description or tool.name
    kwargs: dict[str, Any] = {
        "name": tool.name,
        "description": description,
        "args_schema": tool.json_schema,
    }
    metadata = tool.metadata or {}
    if (metadata.get("kedi_tool_kind") or metadata.get("kind")) in {
        "subagent",
        "codemode_control",
        "artifact_management",
        "filesystem",
        "terminal",
        "dynamic_workflow",
    }:
        kwargs["handle_tool_error"] = True
    if _is_async_callable(tool.fn):
        kwargs["coroutine"] = _langchain_async_tool_function(tool)
    else:
        kwargs["func"] = _langchain_sync_tool_function(tool)
    native_tool = StructuredTool.from_function(**kwargs)
    native_tool.metadata = (
        {**metadata, "kedi_sequential": True} if tool.sequential else tool.metadata
    )
    return native_tool


def _langchain_projection_tools(projection: CodeModeProjection) -> list[Any]:
    if not projection.enabled:
        return list(projection.tools)
    return [_langchain_tool_from_spec(cast(ToolSpec, tool)) for tool in projection.tools]


def _langchain_tool_parameters_schema(tool: Any) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, Mapping):
        return dict(args_schema)
    model_json_schema = getattr(args_schema, "model_json_schema", None)
    return (
        cast(dict[str, Any], model_json_schema())
        if callable(model_json_schema)
        else {"type": "object", "properties": {}}
    )


def _langchain_tool_spec_from_native(tool: Any) -> ToolSpec:
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("LangChain CodeMode tools must expose a non-empty name")
    json_schema = _langchain_tool_parameters_schema(tool)

    async def invoke(**arguments: Any) -> Any:
        ainvoke = getattr(tool, "ainvoke", None)
        if callable(ainvoke):
            result = ainvoke(arguments)
            if inspect.isawaitable(result):
                return await result
            return result
        invoke_sync = getattr(tool, "invoke", None)
        if not callable(invoke_sync):
            raise TypeError(f"LangChain tool {name!r} is not invokable")
        result = invoke_sync(arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    invoke.__name__ = name
    description = getattr(tool, "description", None)
    metadata = getattr(tool, "metadata", None)
    return ToolSpec(
        fn=invoke,
        name=name,
        description=description if isinstance(description, str) else name,
        json_schema=json_schema,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {"source": "langchain"},
        sequential=isinstance(metadata, Mapping) and metadata.get("kedi_sequential") is True,
        risk="mutating",
        risk_resolver=_langchain_external_tool_risk,
    )


def _langchain_sync_tool_function(tool: ToolSpec) -> Callable[..., Any]:
    @wraps(tool.fn)
    def _tool(*args: Any, **kwargs: Any) -> Any:
        from kedi.agent_adapter.adapters.langchain.runtime import record_live_tool_result
        from kedi.agent_adapter.execution import run_sync

        try:
            result = tool.fn(*args, **_validated_tool_arguments(tool, kwargs))
            if inspect.isawaitable(result):
                result = run_sync(cast(Coroutine[Any, Any, Any], result))
            record_live_tool_result(result)
            return result
        except CodeModeError as exc:
            _raise_langchain_codemode_error(exc)
        except Exception as exc:
            _raise_langchain_tool_error(tool, exc)
            raise

    # The native result journal must settle before the physical tool barrier opens.
    return with_live_tool_tracking(replace(tool, fn=_tool)).fn


def _langchain_async_tool_function(tool: ToolSpec) -> Callable[..., Any]:
    @wraps(tool.fn)
    async def _tool(*args: Any, **kwargs: Any) -> Any:
        from kedi.agent_adapter.adapters.langchain.runtime import record_live_tool_result

        try:
            result = tool.fn(*args, **_validated_tool_arguments(tool, kwargs))
            if inspect.isawaitable(result):
                result = await result
            record_live_tool_result(result)
            return result
        except CodeModeError as exc:
            _raise_langchain_codemode_error(exc)
        except Exception as exc:
            _raise_langchain_tool_error(tool, exc)
            raise

    return with_live_tool_tracking(replace(tool, fn=_tool)).fn


def _raise_langchain_tool_error(tool: ToolSpec, exc: Exception) -> None:
    message = correctable_tool_error(tool, exc)
    if message is not None:
        from langchain_core.tools import ToolException

        raise ToolException(message) from exc


def _langchain_tool_error(tool_name: str, tool_call_id: object, message: str) -> Any:
    from langchain.messages import ToolMessage

    return ToolMessage(
        content=message,
        name=tool_name,
        tool_call_id=str(tool_call_id or "kedi-approval"),
        status="error",
    )


def _validated_tool_arguments(tool: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if tool.argument_validator is None:
        return dict(arguments)
    return tool.argument_validator(arguments)


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


def _langchain_external_tool_spec(name: str) -> ToolSpec:
    return ToolSpec(
        fn=lambda **_kwargs: None,
        name=name,
        description="External LangChain or MCP tool",
        metadata={"source": "langchain-external"},
        risk="mutating",
        risk_resolver=_langchain_external_tool_risk,
    )


def _langchain_external_tool_risk(arguments: Mapping[str, Any]) -> ToolRisk:
    return "sensitive" if arguments_target_secret_file(arguments) else "mutating"


def _langchain_mcp_server_configs(profile: AgentProfile) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(profile.mcp_servers):
        name = f"kedi_mcp_{index}"
        config: dict[str, Any]
        if spec.transport == "stdio":
            if spec.command is None:
                raise ValueError("MCP stdio transport requires a command.")
            config = {
                "transport": "stdio",
                "command": spec.command,
                "args": list(spec.args),
            }
            if spec.env is not None:
                config["env"] = spec.env
        elif spec.transport == "sse":
            if spec.url is None:
                raise ValueError("MCP sse transport requires a url.")
            config = {"transport": "sse", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        elif spec.transport == "http":
            if spec.url is None:
                raise ValueError("MCP http transport requires a url.")
            config = {"transport": "streamable_http", "url": spec.url}
            if spec.headers is not None:
                config["headers"] = spec.headers
        else:
            raise ValueError(f"Unsupported MCP transport: {spec.transport}")
        configs[name] = config
    return configs
