from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Public re-exports for adapter implementations and compatibility helpers.
from pydantic_ai import Agent

from ..codemode.pydantic import PydanticCodeModeCapability, pydantic_code_mode_capability
from ..output_schema import make_repr
from ..output_schema import output_field_definition as _output_field_definition
from . import acp, claude, codex, langchain, lazy, pydantic, webgpu
from .acp import ACPAdapter
from .claude import (
    ClaudeAdapter,
    _claude_external_mcp_servers,
    _claude_sdk_tool,
    _output_json_schema,
)
from .codex import CodexAdapter
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
from .pydantic import (
    PydanticAdapter,
    _mcp_toolset_from_config,
    pydantic_approval_request_from_deferred,
    pydantic_deferred_results_from_decisions,
    pydantic_tool_from_spec,
    pydantic_toolset_from_specs,
)
from .webgpu import BrowserModelBridge, WebGPUAdapter

if TYPE_CHECKING:
    from . import dspy
    from .dspy import DSPyAdapter


def __getattr__(name: str) -> Any:
    if name == "DSPyAdapter":
        from .dspy import DSPyAdapter

        globals()[name] = DSPyAdapter
        return DSPyAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Agent",
    "ACPAdapter",
    "BrowserModelBridge",
    "ClaudeAdapter",
    "CodexAdapter",
    "DSPyAdapter",
    "LangChainAdapter",
    "LazyAdapter",
    "PydanticAdapter",
    "PydanticCodeModeCapability",
    "pydantic_approval_request_from_deferred",
    "pydantic_code_mode_capability",
    "pydantic_deferred_results_from_decisions",
    "pydantic_tool_from_spec",
    "pydantic_toolset_from_specs",
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
    "acp",
    "claude",
    "codex",
    "dspy",
    "langchain",
    "lazy",
    "make_repr",
    "pydantic",
    "webgpu",
]
