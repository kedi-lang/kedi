from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "CODEX_DEFAULT_SANDBOX",
    "CODEX_SUPPORTED_STRING_FORMATS",
    "CodexAdapter",
    "OutputDataT",
]

if TYPE_CHECKING:
    from .adapter import (
        CODEX_DEFAULT_SANDBOX as CODEX_DEFAULT_SANDBOX,
    )
    from .adapter import (
        CODEX_SUPPORTED_STRING_FORMATS as CODEX_SUPPORTED_STRING_FORMATS,
    )
    from .adapter import (
        CodexAdapter as CodexAdapter,
    )
    from .adapter import (
        OutputDataT as OutputDataT,
    )
    from .adapter import (
        _callable_json_schema as _callable_json_schema,
    )
    from .adapter import (
        _codex_command_risk as _codex_command_risk,
    )
    from .adapter import (
        _codex_invocation_usage as _codex_invocation_usage,
    )
    from .adapter import (
        _codex_json_schema as _codex_json_schema,
    )
    from .adapter import (
        _codex_native_approval_details as _codex_native_approval_details,
    )
    from .adapter import (
        _codex_native_approval_handler as _codex_native_approval_handler,
    )
    from .adapter import (
        _codex_native_approval_request as _codex_native_approval_request,
    )
    from .adapter import (
        _codex_native_approval_response as _codex_native_approval_response,
    )
    from .adapter import (
        _codex_sandbox_value as _codex_sandbox_value,
    )
    from .adapter import (
        _codex_thread_cleanup as _codex_thread_cleanup,
    )
    from .adapter import (
        _codex_thread_fingerprint as _codex_thread_fingerprint,
    )
    from .adapter import (
        _CodexRun as _CodexRun,
    )
    from .adapter import (
        _copy_setting as _copy_setting,
    )
    from .adapter import (
        _dynamic_tool_spec as _dynamic_tool_spec,
    )
    from .adapter import (
        _is_read_only_command_action as _is_read_only_command_action,
    )
    from .adapter import (
        _json_object_result as _json_object_result,
    )
    from .adapter import (
        _output_json_schema as _output_json_schema,
    )
    from .adapter import (
        _parse_structured_result as _parse_structured_result,
    )
    from .adapter import (
        _tool_with_approval_policies as _tool_with_approval_policies,
    )
    from .adapter import (
        _usage_int as _usage_int,
    )

_EXPORTS = {
    "OutputDataT": ("adapter", "OutputDataT"),
    "CODEX_DEFAULT_SANDBOX": ("adapter", "CODEX_DEFAULT_SANDBOX"),
    "CODEX_SUPPORTED_STRING_FORMATS": ("adapter", "CODEX_SUPPORTED_STRING_FORMATS"),
    "_CodexRun": ("adapter", "_CodexRun"),
    "CodexAdapter": ("adapter", "CodexAdapter"),
    "_dynamic_tool_spec": ("adapter", "_dynamic_tool_spec"),
    "_callable_json_schema": ("adapter", "_callable_json_schema"),
    "_output_json_schema": ("adapter", "_output_json_schema"),
    "_parse_structured_result": ("adapter", "_parse_structured_result"),
    "_json_object_result": ("adapter", "_json_object_result"),
    "_tool_with_approval_policies": ("adapter", "_tool_with_approval_policies"),
    "_codex_native_approval_handler": ("adapter", "_codex_native_approval_handler"),
    "_codex_invocation_usage": ("adapter", "_codex_invocation_usage"),
    "_usage_int": ("adapter", "_usage_int"),
    "_codex_native_approval_request": ("adapter", "_codex_native_approval_request"),
    "_codex_native_approval_details": ("adapter", "_codex_native_approval_details"),
    "_codex_command_risk": ("adapter", "_codex_command_risk"),
    "_is_read_only_command_action": ("adapter", "_is_read_only_command_action"),
    "_codex_native_approval_response": ("adapter", "_codex_native_approval_response"),
    "_copy_setting": ("adapter", "_copy_setting"),
    "_codex_thread_fingerprint": ("adapter", "_codex_thread_fingerprint"),
    "_codex_thread_cleanup": ("adapter", "_codex_thread_cleanup"),
    "_codex_json_schema": ("adapter", "_codex_json_schema"),
    "_codex_sandbox_value": ("adapter", "_codex_sandbox_value"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value
