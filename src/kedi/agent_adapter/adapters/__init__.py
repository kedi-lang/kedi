from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from ..codemode.pydantic import PydanticCodeModeCapability, pydantic_code_mode_capability
    from ..output_schema import make_repr
    from ..output_schema import output_field_definition as _output_field_definition
    from . import a2a, acp, claude, codex, dspy, langchain, lazy, pydantic, webgpu
    from .a2a import A2AAdapter
    from .acp import ACPAdapter
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
    from .pydantic import (
        PydanticAdapter,
        _mcp_toolset_from_config,
        pydantic_approval_request_from_deferred,
        pydantic_deferred_results_from_decisions,
        pydantic_tool_from_spec,
        pydantic_toolset_from_specs,
    )
    from .webgpu import BrowserModelBridge, WebGPUAdapter


_EXPORT_MODULES = {
    "Agent": "pydantic_ai",
    "PydanticCodeModeCapability": "..codemode.pydantic",
    "pydantic_code_mode_capability": "..codemode.pydantic",
    "make_repr": "..output_schema",
    "_output_field_definition": "..output_schema",
    "A2AAdapter": ".a2a",
    "ACPAdapter": ".acp",
    "CodexAdapter": ".codex",
    "DSPyAdapter": ".dspy",
    "LazyAdapter": ".lazy",
    **dict.fromkeys(("BrowserModelBridge", "WebGPUAdapter"), ".webgpu"),
    **dict.fromkeys(
        (
            "ClaudeAdapter",
            "_claude_external_mcp_servers",
            "_claude_sdk_tool",
            "_output_json_schema",
        ),
        ".claude",
    ),
    **dict.fromkeys(
        (
            "LangChainAdapter",
            "_apply_langchain_effort_settings",
            "_langchain_mcp_server_configs",
            "_langchain_structured_response",
            "_langchain_sync_tool_function",
            "_langchain_text_response",
            "_langchain_tool_from_spec",
        ),
        ".langchain",
    ),
    **dict.fromkeys(
        (
            "PydanticAdapter",
            "_mcp_toolset_from_config",
            "pydantic_approval_request_from_deferred",
            "pydantic_deferred_results_from_decisions",
            "pydantic_tool_from_spec",
            "pydantic_toolset_from_specs",
        ),
        ".pydantic",
    ),
}
_MODULES = frozenset(
    {"a2a", "acp", "claude", "codex", "dspy", "langchain", "lazy", "pydantic", "webgpu"}
)


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        value = import_module("." + name, __name__)
    elif name in _EXPORT_MODULES:
        symbol = "output_field_definition" if name == "_output_field_definition" else name
        value = getattr(import_module(_EXPORT_MODULES[name], __name__), symbol)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "Agent",
    "A2AAdapter",
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
    "a2a",
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
