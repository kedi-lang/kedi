from __future__ import annotations

from typing import Literal, cast

McpTransport = Literal["stdio", "sse", "http"]
MCP_TRANSPORT_VALUES = ("http", "sse", "stdio", "streamable-http")


def normalize_mcp_transport(value: str) -> McpTransport:
    transport = value.strip()
    if transport == "streamable-http":
        return "http"
    if transport in {"stdio", "sse", "http"}:
        return cast(McpTransport, transport)
    expected = ", ".join(sorted(MCP_TRANSPORT_VALUES))
    raise ValueError(f"MCP transport must be one of: {expected}")


def is_valid_mcp_transport(value: str) -> bool:
    return value.strip() in MCP_TRANSPORT_VALUES
