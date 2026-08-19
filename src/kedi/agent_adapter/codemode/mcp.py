from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any, Sequence

from ...agent_profile import McpServerSpec
from ..telemetry import mcp_initialization_span, mcp_server_name, mcp_transport_name
from ..tool_override import ToolSpec, arguments_target_secret_file


async def materialize_mcp_tool_specs(
    stack: AsyncExitStack,
    specs: Sequence[McpServerSpec],
    *,
    adapter_shortname: str,
) -> tuple[ToolSpec, ...]:
    """Expose profile MCP tools as lifecycle-owned local ToolSpecs."""

    from fastmcp import Client

    tools: list[ToolSpec] = []
    for index, spec in enumerate(specs):
        with mcp_initialization_span(
            adapter_shortname=adapter_shortname,
            server_name=mcp_server_name(command=spec.command, url=spec.url, index=index),
            transport=mcp_transport_name(spec.transport),
        ) as active:
            client = Client(_fastmcp_config(spec, index))
            await stack.enter_async_context(client)
            discovered = await client.list_tools()
            tools.extend(_tool_spec(client, tool) for tool in discovered)
            active.set_attribute("kedi.mcp.tool_count", len(discovered))
            active.add_event(
                "mcp tools discovered",
                {"kedi.mcp.tool_count": len(discovered)},
            )
    return tuple(tools)


def _tool_spec(client: Any, tool: Any) -> ToolSpec:
    name = tool.name

    async def call(**arguments: Any) -> Any:
        result = await client.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(_mcp_result_text(result))
        if result.structuredContent is not None:
            return result.structuredContent
        return _mcp_result_text(result)

    call.__name__ = name
    call.__doc__ = tool.description or name
    return ToolSpec(
        fn=call,
        name=name,
        description=tool.description or name,
        json_schema=tool.inputSchema,
        return_schema=tool.outputSchema,
        metadata={"source": "mcp"},
        risk="mutating",
        risk_resolver=lambda arguments: (
            "sensitive" if arguments_target_secret_file(arguments) else "mutating"
        ),
    )


def _fastmcp_config(spec: McpServerSpec, index: int) -> dict[str, Any]:
    server: dict[str, Any]
    if spec.transport == "stdio":
        server = {"command": spec.command, "args": list(spec.args)}
        if spec.env is not None:
            server["env"] = spec.env
    else:
        server = {
            "transport": "sse" if spec.transport == "sse" else "streamable-http",
            "url": spec.url,
        }
        if spec.headers is not None:
            server["headers"] = spec.headers
    return {"mcpServers": {f"kedi_codemode_{index}": server}}


def _mcp_result_text(result: Any) -> str:
    from mcp.types import TextContent

    parts: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        else:
            parts.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    return "\n".join(parts)


__all__ = ["materialize_mcp_tool_specs"]
