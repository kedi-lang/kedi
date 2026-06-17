from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..mcp import McpTransport, normalize_mcp_transport


@dataclass(frozen=True, kw_only=True)
class McpServerSpec:
    transport: McpTransport = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None

    def normalized(self) -> McpServerSpec:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio MCP directive requires command")
            return replace(self, url=None, headers=None)
        if self.transport in {"sse", "http"}:
            if not self.url:
                raise ValueError(f"{self.transport} MCP directive requires url")
            return replace(self, command=None, args=(), env=None)
        raise ValueError(f"Unsupported MCP transport: {self.transport}")


def mcp_spec_from_mapping(values: dict[str, Any]) -> McpServerSpec:
    transport = values.get("transport")
    if transport is None:
        transport = "stdio" if values.get("command") is not None else None
    if not isinstance(transport, str):
        raise ValueError("MCP transport must be one of: http, sse, stdio, streamable-http")
    transport = normalize_mcp_transport(transport)

    args = values.get("args", ())
    if args is None:
        args = ()
    if isinstance(args, str) or not isinstance(args, (list, tuple)):
        raise TypeError("MCP args must be a list of strings")
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("MCP args must be a list of strings")

    env = _optional_str_dict(values.get("env"), "env")
    headers = _optional_str_dict(values.get("headers"), "headers")
    command = values.get("command")
    url = values.get("url")
    if command is not None and not isinstance(command, str):
        raise TypeError("MCP command must be a string")
    if url is not None and not isinstance(url, str):
        raise TypeError("MCP url must be a string")

    return McpServerSpec(
        transport=transport,
        command=command,
        args=tuple(args),
        env=env,
        url=url,
        headers=headers,
    ).normalized()


def _optional_str_dict(value: Any, name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"MCP {name} must be a dict of strings")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise TypeError(f"MCP {name} must be a dict of strings")
    return dict(value)
