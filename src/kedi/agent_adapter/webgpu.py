from __future__ import annotations

import asyncio
import inspect
import json
import re
from contextlib import AbstractContextManager, AsyncExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from textwrap import indent
from typing import Any, Callable, Collection, Iterator, Mapping, Protocol, Sequence
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import format_as_xml
from pydantic_core import to_jsonable_python

from ..agent_profile import AgentProfile, McpServerSpec, McpTransport
from ..dynamic_models import build_dynamic_model
from ..resolved_types import ResolvedKediType
from .tool_override import ToolScope, ToolSpec


class BrowserModelBridge(Protocol):
    """Synchronous transport used by WebGPUAdapter to reach a browser model."""

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]: ...


class WebGPUInstrumentation(Protocol):
    """Optional instrumentation hooks for browser-backed adapter operations."""

    def tool_span(
        self,
        name: str,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> AbstractContextManager[Callable[[Any], None]]: ...


@dataclass(frozen=True)
class _RuntimeTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    return_schema: dict[str, Any] | None
    fn: Any

    async def call(self, arguments: dict[str, Any]) -> Any:
        result = self.fn(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result


@dataclass(frozen=True)
class _McpRuntimeTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    return_schema: dict[str, Any] | None
    client: Any

    async def call(self, arguments: dict[str, Any]) -> Any:
        result = await self.client.call_tool(self.name, arguments)
        if result.isError:
            raise RuntimeError(_mcp_result_text(result))
        if result.structuredContent is not None:
            return result.structuredContent
        return _mcp_result_text(result)


BrowserTool = _RuntimeTool | _McpRuntimeTool


class WebGPUAdapter:
    """Route Kedi agent calls to a browser-hosted model through a small bridge."""

    _SETTING_KEYS = frozenset(
        {
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "stop_sequences",
            "timeout",
        }
    )

    def __init__(
        self,
        bridge: BrowserModelBridge,
        *,
        model: str,
        supported_models: Sequence[str] = (),
        model_settings: Mapping[str, Any] | None = None,
        timeout: float = 300,
        max_tool_steps: int = 8,
        allowed_mcp_transports: Collection[McpTransport] | None = None,
        instrumentation: WebGPUInstrumentation | None = None,
    ) -> None:
        if max_tool_steps < 1:
            raise ValueError("max_tool_steps must be at least 1")
        self._bridge = bridge
        self._model = model
        self._supported_models = frozenset(supported_models)
        self._model_settings = dict(model_settings or {})
        self._timeout = timeout
        self._max_tool_steps = max_tool_steps
        self._allowed_mcp_transports = (
            frozenset(allowed_mcp_transports) if allowed_mcp_transports is not None else None
        )
        self._instrumentation = instrumentation
        self._tool_scope = ToolScope()
        self._profile_stack: ContextVar[tuple[AgentProfile, ...]] = ContextVar(
            f"kedi_webgpu_profile_stack_{id(self)}",
            default=(),
        )
        self._validate_model(model)

    def default_model(self) -> str:
        return self._model

    @property
    def active_profile(self) -> AgentProfile:
        stack = self._profile_stack.get()
        return stack[-1] if stack else AgentProfile.empty()

    @contextmanager
    def profile_override(self, profile: AgentProfile) -> Iterator[None]:
        if profile.model is not None:
            self._validate_model(profile.model)
        token = self._profile_stack.set((*self._profile_stack.get(), profile))
        try:
            yield
        finally:
            self._profile_stack.reset(token)

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._tool_scope.tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tool_scope.register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._tool_scope.scoped_tools

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        instructions: str | None = None,
        **_: Any,
    ) -> Any:
        output_adapter = _output_adapter(output_schema, output_type)
        profile = self.active_profile
        settings = self._settings(profile)
        timeout = float(settings.pop("timeout", self._timeout))
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": template,
            }
        ]
        output_json_schema = output_adapter.json_schema()

        async with AsyncExitStack() as stack:
            tools = await self._runtime_tools(stack, profile)
            tool_contracts = [_tool_contract(tool) for tool in tools]
            required_tools = set(_mentioned_tools(template, tools))
            called_tools: set[str] = set()
            for step in range(self._max_tool_steps):
                payload = {
                    "model": profile.model or self._model,
                    "messages": [dict(message) for message in messages],
                    "instructions": instructions or profile.system,
                    "effort": profile.effort,
                    "settings": settings,
                    "outputSchema": output_json_schema,
                    "outputPrompt": _format_output_contract(output_json_schema),
                    "tools": tool_contracts,
                    "toolsPrompt": _format_tool_catalog(tool_contracts),
                    "requiredTools": sorted(required_tools - called_tools),
                    "step": step,
                }
                response = await asyncio.to_thread(
                    self._bridge.request,
                    payload,
                    timeout=timeout,
                )
                kind = response.get("kind", "final")
                if kind == "final":
                    missing_tools = required_tools - called_tools
                    if missing_tools:
                        names = ", ".join(sorted(missing_tools))
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Call the explicitly requested tool(s) before answering: "
                                    f"{names}. Return a tool call now; do not simulate the result."
                                ),
                            }
                        )
                        continue
                    data = response.get("data")
                    if not isinstance(data, dict):
                        raise TypeError("Browser model final response data must be an object")
                    return output_adapter.validate_python(data)
                if kind == "error":
                    raise RuntimeError(str(response.get("error") or "Browser model request failed"))
                if kind == "retry":
                    required_tool = response.get("requiredTool")
                    if isinstance(required_tool, str):
                        if not any(tool.name == required_tool for tool in tools):
                            raise ValueError(
                                f"Browser model retry named unknown tool: {required_tool!r}"
                            )
                        required_tools.add(required_tool)
                    telemetry = response.get("telemetry")
                    if isinstance(telemetry, Mapping):
                        output_text = telemetry.get("outputText")
                        if isinstance(output_text, str) and output_text:
                            messages.append({"role": "assistant", "content": output_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": str(
                                response.get("error")
                                or "Return exactly one call_tool or final action JSON object."
                            ),
                        }
                    )
                    continue
                if kind != "tool_call":
                    raise ValueError(f"Unknown browser model response kind: {kind!r}")
                name, succeeded, tool_messages = await self._execute_tool_call(response, tools)
                if succeeded:
                    called_tools.add(name)
                else:
                    required_tools.add(name)
                messages.extend(tool_messages)

        raise RuntimeError(f"Browser model exceeded the {self._max_tool_steps}-step tool limit")

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from .utils import run_sync

        return run_sync(
            self.produce(
                template=template,
                output_schema=output_schema,
                output_type=output_type,
                instructions=instructions,
                **kwargs,
            )
        )

    def _validate_model(self, model: str) -> None:
        if self._supported_models and model not in self._supported_models:
            raise ValueError(f"Unknown browser model: {model!r}")

    def _settings(self, profile: AgentProfile) -> dict[str, Any]:
        settings = {**self._model_settings, **profile.settings}
        unsupported = set(settings) - self._SETTING_KEYS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported WebGPU model settings: {names}")
        return settings

    async def _runtime_tools(
        self,
        stack: AsyncExitStack,
        profile: AgentProfile,
    ) -> list[BrowserTool]:
        if self._allowed_mcp_transports is not None:
            disallowed = {
                spec.transport
                for spec in profile.mcp_servers
                if spec.transport not in self._allowed_mcp_transports
            }
            if disallowed:
                names = ", ".join(sorted(disallowed))
                raise ValueError(f"WebGPUAdapter does not allow {names} MCP transport")
        tools: list[BrowserTool] = [
            _RuntimeTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.json_schema or {"type": "object", "properties": {}},
                return_schema=tool.return_schema,
                fn=tool.fn,
            )
            for tool in self.scoped_tools
            if isinstance(tool, ToolSpec)
        ]
        tools.extend(await _mcp_tools(stack, profile.mcp_servers))
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique within one WebGPU agent run")
        return tools

    async def _execute_tool_call(
        self,
        response: Mapping[str, Any],
        tools: Sequence[BrowserTool],
    ) -> tuple[str, bool, list[dict[str, Any]]]:
        name = response.get("name")
        arguments = response.get("arguments", {})
        call_id = response.get("callId") or f"tool-{uuid4().hex}"
        if not isinstance(name, str):
            raise TypeError("Browser tool call name must be a string")
        if not isinstance(arguments, dict):
            raise TypeError("Browser tool call arguments must be an object")
        tool = next((candidate for candidate in tools if candidate.name == name), None)
        if tool is None:
            raise ValueError(f"Browser model requested unknown tool: {name!r}")
        try:
            span = (
                self._instrumentation.tool_span(name, str(call_id), arguments)
                if self._instrumentation is not None
                else nullcontext(lambda _: None)
            )
            with span as finish:
                result = await tool.call(arguments)
                result = to_jsonable_python(result, fallback=str)
                finish(result)
            content = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - tool failures are returned to the model.
            content = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return (
            name,
            content["ok"] is True,
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action": "call_tool",
                            "callId": call_id,
                            "name": name,
                            "arguments": arguments,
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "role": "tool",
                    "name": name,
                    "toolCallId": call_id,
                    "content": json.dumps(content, ensure_ascii=False),
                },
            ],
        )


def _tool_contract(tool: BrowserTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
        "returnSchema": tool.return_schema,
    }


def _format_tool_catalog(tools: Sequence[Mapping[str, Any]]) -> str:
    if not tools:
        return "<Tools />"
    rendered = "\n".join(
        format_as_xml(_tool_prompt_contract(tool), root_tag="Tool") for tool in tools
    )
    return f"<Tools>\n{indent(rendered, '  ')}\n</Tools>"


def _format_output_contract(schema: Mapping[str, Any]) -> str:
    return format_as_xml(_output_prompt_contract(schema), root_tag="Output")


def _tool_prompt_contract(tool: Mapping[str, Any]) -> dict[str, Any]:
    input_schema = tool.get("inputSchema")
    schema = input_schema if isinstance(input_schema, Mapping) else {}
    properties = schema.get("properties")
    fields = properties if isinstance(properties, Mapping) else {}
    required_value = schema.get("required")
    required = set(required_value) if isinstance(required_value, list) else set()
    arguments = {
        name: _schema_type_name(field_schema, schema)
        for name, field_schema in fields.items()
        if isinstance(name, str) and name in required and isinstance(field_schema, Mapping)
    }
    optional_arguments = {
        name: _schema_type_name(field_schema, schema)
        for name, field_schema in fields.items()
        if isinstance(name, str) and name not in required and isinstance(field_schema, Mapping)
    }
    result: dict[str, Any] = {
        "name": tool.get("name", ""),
        "arguments": arguments,
    }
    if optional_arguments:
        result["optional_arguments"] = optional_arguments
    description = tool.get("description")
    if isinstance(description, str) and description:
        result["description"] = description
    return_schema = tool.get("returnSchema")
    if isinstance(return_schema, Mapping):
        result["returns"] = _schema_type_name(return_schema, return_schema)
    return result


def _output_prompt_contract(schema: Mapping[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return {"type": _schema_type_name(schema, schema)}
    required_value = schema.get("required")
    required = set(required_value) if isinstance(required_value, list) else set()
    fields = {
        name: _schema_type_name(field_schema, schema)
        for name, field_schema in properties.items()
        if isinstance(name, str) and name in required and isinstance(field_schema, Mapping)
    }
    optional_fields = {
        name: _schema_type_name(field_schema, schema)
        for name, field_schema in properties.items()
        if isinstance(name, str) and name not in required and isinstance(field_schema, Mapping)
    }
    result: dict[str, Any] = {"fields": fields}
    if optional_fields:
        result["optional_fields"] = optional_fields
    return result


def _schema_type_name(schema: Mapping[str, Any], root_schema: Mapping[str, Any]) -> str:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        return reference.rsplit("/", 1)[-1]
    enum = schema.get("enum")
    if isinstance(enum, list):
        return " | ".join(map(str, enum))
    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if isinstance(variants, list):
            return " | ".join(
                _schema_type_name(item, root_schema)
                for item in variants
                if isinstance(item, Mapping)
            )
    schema_type = schema.get("type")
    if schema_type == "array":
        items = schema.get("items")
        item_type = _schema_type_name(items, root_schema) if isinstance(items, Mapping) else "value"
        return f"list[{item_type}]"
    if schema_type == "object":
        title = schema.get("title")
        return str(title) if title else "object"
    if isinstance(schema_type, list):
        return " | ".join(map(str, schema_type))
    return str(schema_type or schema.get("title") or "value")


def _mentioned_tools(template: str, tools: Sequence[BrowserTool]) -> tuple[str, ...]:
    return tuple(
        tool.name for tool in tools if re.search(rf"(?<!\w){re.escape(tool.name)}(?!\w)", template)
    )


async def _mcp_tools(
    stack: AsyncExitStack,
    specs: Sequence[McpServerSpec],
) -> list[_McpRuntimeTool]:
    if not specs:
        return []
    from fastmcp import Client

    tools: list[_McpRuntimeTool] = []
    for index, spec in enumerate(specs):
        client = Client(_fastmcp_config(spec, index))
        await stack.enter_async_context(client)
        for tool in await client.list_tools():
            tools.append(
                _McpRuntimeTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    return_schema=tool.outputSchema,
                    client=client,
                )
            )
    return tools


def _fastmcp_config(spec: McpServerSpec, index: int) -> dict[str, Any]:
    server: dict[str, Any]
    if spec.transport == "stdio":
        server = {
            "command": spec.command,
            "args": list(spec.args),
        }
        if spec.env is not None:
            server["env"] = spec.env
    else:
        server = {
            "transport": "sse" if spec.transport == "sse" else "streamable-http",
            "url": spec.url,
        }
        if spec.headers is not None:
            server["headers"] = spec.headers
    return {"mcpServers": {f"kedi_webgpu_{index}": server}}


def _mcp_result_text(result: Any) -> str:
    from mcp.types import TextContent

    parts: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        else:
            parts.append(json.dumps(item.model_dump(mode="json")))
    return "\n".join(parts)


def _output_adapter(
    output_schema: dict[str, ResolvedKediType] | None,
    output_type: type[Any] | None,
) -> TypeAdapter[Any]:
    if output_schema:
        output_type = build_dynamic_model(
            "_WebGPUOutput",
            BaseModel,
            {name: (field_type, ...) for name, field_type in output_schema.items()},
        )
    if output_type is None:
        raise ValueError("Output spec must be provided.")
    return TypeAdapter(output_type)


__all__ = ["BrowserModelBridge", "WebGPUAdapter", "WebGPUInstrumentation"]
