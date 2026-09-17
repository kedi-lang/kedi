from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["LangChainAdapter"]

if TYPE_CHECKING:
    from .adapter import LangChainAdapter as LangChainAdapter
    from .messages import (
        _apply_langchain_effort_settings as _apply_langchain_effort_settings,
    )
    from .messages import (
        _langchain_completed_message_text as _langchain_completed_message_text,
    )
    from .messages import (
        _langchain_content_text as _langchain_content_text,
    )
    from .messages import (
        _langchain_current_messages as _langchain_current_messages,
    )
    from .messages import (
        _langchain_event_handler as _langchain_event_handler,
    )
    from .messages import (
        _langchain_invocation_usage as _langchain_invocation_usage,
    )
    from .messages import (
        _langchain_message_identity as _langchain_message_identity,
    )
    from .messages import (
        _langchain_message_text as _langchain_message_text,
    )
    from .messages import (
        _langchain_structured_response as _langchain_structured_response,
    )
    from .messages import (
        _langchain_text_response as _langchain_text_response,
    )
    from .messages import (
        _langchain_uses_explicit_cache_blocks as _langchain_uses_explicit_cache_blocks,
    )
    from .messages import (
        _langchain_with_cache_control as _langchain_with_cache_control,
    )
    from .messages import (
        _langchain_without_cache_control as _langchain_without_cache_control,
    )
    from .messages import (
        _raise_langchain_codemode_error as _raise_langchain_codemode_error,
    )
    from .messages import (
        _same_langchain_message as _same_langchain_message,
    )
    from .messages import (
        _usage_int as _usage_int,
    )
    from .middleware import (
        _langchain_admit_native_tool_result as _langchain_admit_native_tool_result,
    )
    from .middleware import (
        _langchain_approval_middleware as _langchain_approval_middleware,
    )
    from .middleware import (
        _langchain_cache_middleware as _langchain_cache_middleware,
    )
    from .middleware import (
        _langchain_hook_origin as _langchain_hook_origin,
    )
    from .middleware import (
        _langchain_tool_message_for_call as _langchain_tool_message_for_call,
    )
    from .tools import (
        _is_async_callable as _is_async_callable,
    )
    from .tools import (
        _langchain_async_tool_function as _langchain_async_tool_function,
    )
    from .tools import (
        _langchain_external_tool_risk as _langchain_external_tool_risk,
    )
    from .tools import (
        _langchain_external_tool_spec as _langchain_external_tool_spec,
    )
    from .tools import (
        _langchain_mcp_server_configs as _langchain_mcp_server_configs,
    )
    from .tools import (
        _langchain_projection_tools as _langchain_projection_tools,
    )
    from .tools import (
        _langchain_sync_tool_function as _langchain_sync_tool_function,
    )
    from .tools import (
        _langchain_tool_error as _langchain_tool_error,
    )
    from .tools import (
        _langchain_tool_from_spec as _langchain_tool_from_spec,
    )
    from .tools import (
        _langchain_tool_spec_from_native as _langchain_tool_spec_from_native,
    )
    from .tools import (
        _raise_langchain_tool_error as _raise_langchain_tool_error,
    )
    from .tools import (
        _validated_tool_arguments as _validated_tool_arguments,
    )

_EXPORTS = {
    "LangChainAdapter": ("adapter", "LangChainAdapter"),
    "_langchain_uses_explicit_cache_blocks": ("messages", "_langchain_uses_explicit_cache_blocks"),
    "_langchain_with_cache_control": ("messages", "_langchain_with_cache_control"),
    "_langchain_without_cache_control": ("messages", "_langchain_without_cache_control"),
    "_raise_langchain_codemode_error": ("messages", "_raise_langchain_codemode_error"),
    "_langchain_event_handler": ("messages", "_langchain_event_handler"),
    "_langchain_completed_message_text": ("messages", "_langchain_completed_message_text"),
    "_langchain_invocation_usage": ("messages", "_langchain_invocation_usage"),
    "_langchain_current_messages": ("messages", "_langchain_current_messages"),
    "_same_langchain_message": ("messages", "_same_langchain_message"),
    "_langchain_message_identity": ("messages", "_langchain_message_identity"),
    "_usage_int": ("messages", "_usage_int"),
    "_apply_langchain_effort_settings": ("messages", "_apply_langchain_effort_settings"),
    "_langchain_structured_response": ("messages", "_langchain_structured_response"),
    "_langchain_text_response": ("messages", "_langchain_text_response"),
    "_langchain_message_text": ("messages", "_langchain_message_text"),
    "_langchain_content_text": ("messages", "_langchain_content_text"),
    "_langchain_cache_middleware": ("middleware", "_langchain_cache_middleware"),
    "_langchain_approval_middleware": ("middleware", "_langchain_approval_middleware"),
    "_langchain_admit_native_tool_result": ("middleware", "_langchain_admit_native_tool_result"),
    "_langchain_tool_message_for_call": ("middleware", "_langchain_tool_message_for_call"),
    "_langchain_hook_origin": ("middleware", "_langchain_hook_origin"),
    "_langchain_tool_from_spec": ("tools", "_langchain_tool_from_spec"),
    "_langchain_projection_tools": ("tools", "_langchain_projection_tools"),
    "_langchain_tool_spec_from_native": ("tools", "_langchain_tool_spec_from_native"),
    "_langchain_sync_tool_function": ("tools", "_langchain_sync_tool_function"),
    "_langchain_async_tool_function": ("tools", "_langchain_async_tool_function"),
    "_raise_langchain_tool_error": ("tools", "_raise_langchain_tool_error"),
    "_langchain_tool_error": ("tools", "_langchain_tool_error"),
    "_validated_tool_arguments": ("tools", "_validated_tool_arguments"),
    "_is_async_callable": ("tools", "_is_async_callable"),
    "_langchain_external_tool_spec": ("tools", "_langchain_external_tool_spec"),
    "_langchain_external_tool_risk": ("tools", "_langchain_external_tool_risk"),
    "_langchain_mcp_server_configs": ("tools", "_langchain_mcp_server_configs"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value
