from __future__ import annotations

# Public re-exports for adapter implementations and compatibility helpers.
from pydantic_ai import Agent

from .._schema import make_repr
from .._schema import output_field_definition as _output_field_definition
from . import claude, codex, dspy, langchain, lazy, pydantic, webgpu
from .claude import (
    ClaudeAdapter,
    _claude_external_mcp_servers,
    _claude_sdk_tool,
    _output_json_schema,
)
from .codex import CodexAdapter
from .dspy import DSPyAdapter
from .langchain import (
    LangChainAdapter,
    _apply_langchain_effort_settings,
    _langchain_mcp_server_configs,
    _langchain_structured_response,
    _langchain_sync_tool_function,
    _langchain_text_response,
    _langchain_tool_from_spec,
)
from .lazy import LazyAdapter
from .pydantic import PydanticAdapter, _mcp_toolset_from_config
from .webgpu import BrowserModelBridge, WebGPUAdapter

__all__ = [
    "Agent",
    "BrowserModelBridge",
    "ClaudeAdapter",
    "CodexAdapter",
    "DSPyAdapter",
    "LangChainAdapter",
    "LazyAdapter",
    "PydanticAdapter",
    "WebGPUAdapter",
    "_apply_langchain_effort_settings",
    "_claude_external_mcp_servers",
    "_claude_sdk_tool",
    "_langchain_mcp_server_configs",
    "_langchain_structured_response",
    "_langchain_sync_tool_function",
    "_langchain_text_response",
    "_langchain_tool_from_spec",
    "_mcp_toolset_from_config",
    "_output_json_schema",
    "_output_field_definition",
    "claude",
    "codex",
    "dspy",
    "langchain",
    "lazy",
    "make_repr",
    "pydantic",
    "webgpu",
]
