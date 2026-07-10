from __future__ import annotations

# Backwards-compatible imports for existing user code and tests.
import dspy
from pydantic_ai import Agent

from ._schema import make_repr, output_field_definition as _output_field_definition
from .claude import (
    ClaudeAdapter,
    _claude_external_mcp_servers,
    _claude_sdk_tool,
    _output_json_schema,
)
from .dspy import DSPyAdapter
from .langchain import (
    LangChainAdapter,
    _apply_langchain_effort_settings,
    _langchain_mcp_server_configs,
    _langchain_structured_response,
    _langchain_sync_tool_function,
    _langchain_tool_from_spec,
)
from .pydantic import PydanticAdapter, _mcp_toolset_from_config

__all__ = [
    "Agent",
    "ClaudeAdapter",
    "DSPyAdapter",
    "LangChainAdapter",
    "PydanticAdapter",
    "_apply_langchain_effort_settings",
    "_claude_external_mcp_servers",
    "_claude_sdk_tool",
    "_langchain_mcp_server_configs",
    "_langchain_structured_response",
    "_langchain_sync_tool_function",
    "_langchain_tool_from_spec",
    "_mcp_toolset_from_config",
    "_output_json_schema",
    "_output_field_definition",
    "dspy",
    "make_repr",
]
