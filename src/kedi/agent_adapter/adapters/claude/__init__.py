from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "CLAUDE_CODE_PERMISSION_MODE",
    "CLAUDE_CODE_SYSTEM_PROMPT_PRESET",
    "CLAUDE_CODE_TOOLS_PRESET",
    "ClaudeAdapter",
    "OutputDataT",
]

if TYPE_CHECKING:
    from .adapter import (
        CLAUDE_CODE_PERMISSION_MODE as CLAUDE_CODE_PERMISSION_MODE,
    )
    from .adapter import (
        CLAUDE_CODE_TOOLS_PRESET as CLAUDE_CODE_TOOLS_PRESET,
    )
    from .adapter import (
        ClaudeAdapter as ClaudeAdapter,
    )
    from .adapter import (
        OutputDataT as OutputDataT,
    )
    from .hooks import (
        _claude_hook_deny as _claude_hook_deny,
    )
    from .hooks import (
        _claude_hook_matcher_class as _claude_hook_matcher_class,
    )
    from .hooks import (
        _claude_hook_tool as _claude_hook_tool,
    )
    from .hooks import (
        _claude_post_tool_use_failure_hook as _claude_post_tool_use_failure_hook,
    )
    from .hooks import (
        _claude_post_tool_use_hook as _claude_post_tool_use_hook,
    )
    from .hooks import (
        _claude_pre_tool_use_hook as _claude_pre_tool_use_hook,
    )
    from .hooks import (
        _with_kedi_approval_hook as _with_kedi_approval_hook,
    )
    from .messages import (
        CLAUDE_CODE_SYSTEM_PROMPT_PRESET as CLAUDE_CODE_SYSTEM_PROMPT_PRESET,
    )
    from .messages import (
        _active_claude_hook_identities as _active_claude_hook_identities,
    )
    from .messages import (
        _claude_assistant_text as _claude_assistant_text,
    )
    from .messages import (
        _claude_fingerprint_default as _claude_fingerprint_default,
    )
    from .messages import (
        _claude_has_tool_use as _claude_has_tool_use,
    )
    from .messages import (
        _claude_invocation_usage as _claude_invocation_usage,
    )
    from .messages import (
        _claude_non_retryable_system_error as _claude_non_retryable_system_error,
    )
    from .messages import (
        _claude_session_fingerprint as _claude_session_fingerprint,
    )
    from .messages import (
        _claude_text_response as _claude_text_response,
    )
    from .messages import (
        _is_result_message as _is_result_message,
    )
    from .messages import (
        _json_schema_type_name as _json_schema_type_name,
    )
    from .messages import (
        _observe_claude_message as _observe_claude_message,
    )
    from .messages import (
        _output_json_schema as _output_json_schema,
    )
    from .messages import (
        _set_claude_system_prompt as _set_claude_system_prompt,
    )
    from .messages import (
        _structured_output_value as _structured_output_value,
    )
    from .messages import (
        _usage_int as _usage_int,
    )
    from .tools import (
        _CLAUDE_KEDI_MCP_TOOL_PREFIX as _CLAUDE_KEDI_MCP_TOOL_PREFIX,
    )
    from .tools import (
        _CLAUDE_READ_ONLY_TOOLS as _CLAUDE_READ_ONLY_TOOLS,
    )
    from .tools import (
        _claude_external_mcp_servers as _claude_external_mcp_servers,
    )
    from .tools import (
        _claude_native_tool_risk as _claude_native_tool_risk,
    )
    from .tools import (
        _claude_native_tool_spec as _claude_native_tool_spec,
    )
    from .tools import (
        _claude_sdk as _claude_sdk,
    )
    from .tools import (
        _claude_sdk_mcp_server as _claude_sdk_mcp_server,
    )
    from .tools import (
        _claude_sdk_tool as _claude_sdk_tool,
    )
    from .tools import (
        _claude_tool_call_input as _claude_tool_call_input,
    )
    from .tools import (
        _claude_tool_specs_by_native_name as _claude_tool_specs_by_native_name,
    )
    from .tools import (
        _jsonable as _jsonable,
    )
    from .tools import (
        _tool_result_text as _tool_result_text,
    )
    from .tools import (
        _validated_tool_arguments as _validated_tool_arguments,
    )

_EXPORTS = {
    "OutputDataT": ("adapter", "OutputDataT"),
    "CLAUDE_CODE_TOOLS_PRESET": ("adapter", "CLAUDE_CODE_TOOLS_PRESET"),
    "CLAUDE_CODE_SYSTEM_PROMPT_PRESET": ("messages", "CLAUDE_CODE_SYSTEM_PROMPT_PRESET"),
    "CLAUDE_CODE_PERMISSION_MODE": ("adapter", "CLAUDE_CODE_PERMISSION_MODE"),
    "_CLAUDE_READ_ONLY_TOOLS": ("tools", "_CLAUDE_READ_ONLY_TOOLS"),
    "_CLAUDE_KEDI_MCP_TOOL_PREFIX": ("tools", "_CLAUDE_KEDI_MCP_TOOL_PREFIX"),
    "ClaudeAdapter": ("adapter", "ClaudeAdapter"),
    "_claude_session_fingerprint": ("messages", "_claude_session_fingerprint"),
    "_active_claude_hook_identities": ("messages", "_active_claude_hook_identities"),
    "_claude_fingerprint_default": ("messages", "_claude_fingerprint_default"),
    "_set_claude_system_prompt": ("messages", "_set_claude_system_prompt"),
    "_output_json_schema": ("messages", "_output_json_schema"),
    "_json_schema_type_name": ("messages", "_json_schema_type_name"),
    "_structured_output_value": ("messages", "_structured_output_value"),
    "_claude_text_response": ("messages", "_claude_text_response"),
    "_observe_claude_message": ("messages", "_observe_claude_message"),
    "_claude_assistant_text": ("messages", "_claude_assistant_text"),
    "_claude_has_tool_use": ("messages", "_claude_has_tool_use"),
    "_is_result_message": ("messages", "_is_result_message"),
    "_claude_non_retryable_system_error": ("messages", "_claude_non_retryable_system_error"),
    "_claude_invocation_usage": ("messages", "_claude_invocation_usage"),
    "_usage_int": ("messages", "_usage_int"),
    "_claude_sdk": ("tools", "_claude_sdk"),
    "_claude_sdk_mcp_server": ("tools", "_claude_sdk_mcp_server"),
    "_claude_sdk_tool": ("tools", "_claude_sdk_tool"),
    "_claude_tool_call_input": ("tools", "_claude_tool_call_input"),
    "_claude_tool_specs_by_native_name": ("tools", "_claude_tool_specs_by_native_name"),
    "_claude_native_tool_spec": ("tools", "_claude_native_tool_spec"),
    "_claude_native_tool_risk": ("tools", "_claude_native_tool_risk"),
    "_validated_tool_arguments": ("tools", "_validated_tool_arguments"),
    "_tool_result_text": ("tools", "_tool_result_text"),
    "_jsonable": ("tools", "_jsonable"),
    "_claude_external_mcp_servers": ("tools", "_claude_external_mcp_servers"),
    "_with_kedi_approval_hook": ("hooks", "_with_kedi_approval_hook"),
    "_claude_hook_matcher_class": ("hooks", "_claude_hook_matcher_class"),
    "_claude_pre_tool_use_hook": ("hooks", "_claude_pre_tool_use_hook"),
    "_claude_post_tool_use_hook": ("hooks", "_claude_post_tool_use_hook"),
    "_claude_post_tool_use_failure_hook": ("hooks", "_claude_post_tool_use_failure_hook"),
    "_claude_hook_tool": ("hooks", "_claude_hook_tool"),
    "_claude_hook_deny": ("hooks", "_claude_hook_deny"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value
