"""Strict MCP fixture: approval metadata must never reach the remote tool."""

import asyncio
import json

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("reason-test")


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="write",
            description="Write a test value.",
            inputSchema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    ]


@server.call_tool()
async def call_tool(name, arguments):
    if name != "write" or set(arguments) != {"value"}:
        raise ValueError("Approval metadata leaked to MCP")
    return [types.TextContent(type="text", text=json.dumps(arguments))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
