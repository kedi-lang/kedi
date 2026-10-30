from __future__ import annotations

from typing import Literal, cast

from .directive_values import FIELD_RULES

McpTransport = Literal["stdio", "sse", "http"]
MCP_TRANSPORT_VALUES = FIELD_RULES["mcp"]["transport"].options


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
