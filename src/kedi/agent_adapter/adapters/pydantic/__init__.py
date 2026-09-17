from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "KEDI_APPROVAL_METADATA_KEY",
    "PydanticAdapter",
    "PydanticApprovalResolution",
    "pydantic_approval_request_from_deferred",
    "pydantic_deferred_results_from_decisions",
    "pydantic_tool_from_spec",
    "pydantic_toolset_from_specs",
]

if TYPE_CHECKING:
    from .adapter import PydanticAdapter as PydanticAdapter
    from .capabilities import _KediAgentEventCapability as _KediAgentEventCapability
    from .capabilities import _KediArtifactCapability as _KediArtifactCapability
    from .capabilities import _KediDeferredApprovalCapability as _KediDeferredApprovalCapability
    from .capabilities import _KediExternalApprovalCapability as _KediExternalApprovalCapability
    from .capabilities import _KediRequiredToolCapability as _KediRequiredToolCapability
    from .capabilities import _KediTelemetryCapability as _KediTelemetryCapability
    from .hooks import _KediHookCapability as _KediHookCapability
    from .run_config import _cached_model_settings as _cached_model_settings
    from .run_config import _profile_model_settings as _profile_model_settings
    from .run_config import _responses_session_model as _responses_session_model
    from .run_config import _with_capability as _with_capability
    from .tools import _DEFAULT_TOOL_RETRIES as _DEFAULT_TOOL_RETRIES
    from .tools import KEDI_APPROVAL_METADATA_KEY as KEDI_APPROVAL_METADATA_KEY
    from .tools import PydanticApprovalResolution as PydanticApprovalResolution
    from .tools import _approval_request_payload as _approval_request_payload
    from .tools import _approval_required_toolset as _approval_required_toolset
    from .tools import _external_tool_risk as _external_tool_risk
    from .tools import _external_tool_spec as _external_tool_spec
    from .tools import _is_async_callable as _is_async_callable
    from .tools import _kedi_tool_specs_from_toolsets as _kedi_tool_specs_from_toolsets
    from .tools import _KediPydanticTool as _KediPydanticTool
    from .tools import _mcp_toolset_from_config as _mcp_toolset_from_config
    from .tools import _model_retry_for_tool_error as _model_retry_for_tool_error
    from .tools import _preserve_tool_introspection as _preserve_tool_introspection
    from .tools import _toolset_sequence as _toolset_sequence
    from .tools import _validated_tool_arguments as _validated_tool_arguments
    from .tools import _validated_tool_function as _validated_tool_function
    from .tools import (
        pydantic_approval_request_from_deferred as pydantic_approval_request_from_deferred,
    )
    from .tools import (
        pydantic_deferred_results_from_decisions as pydantic_deferred_results_from_decisions,
    )
    from .tools import pydantic_tool_from_spec as pydantic_tool_from_spec
    from .tools import pydantic_toolset_from_specs as pydantic_toolset_from_specs

_EXPORTS = {
    "PydanticAdapter": ("adapter", "PydanticAdapter"),
    "_KediAgentEventCapability": ("capabilities", "_KediAgentEventCapability"),
    "_KediArtifactCapability": ("capabilities", "_KediArtifactCapability"),
    "_KediDeferredApprovalCapability": ("capabilities", "_KediDeferredApprovalCapability"),
    "_KediExternalApprovalCapability": ("capabilities", "_KediExternalApprovalCapability"),
    "_KediRequiredToolCapability": ("capabilities", "_KediRequiredToolCapability"),
    "_KediTelemetryCapability": ("capabilities", "_KediTelemetryCapability"),
    "_KediHookCapability": ("hooks", "_KediHookCapability"),
    "_cached_model_settings": ("run_config", "_cached_model_settings"),
    "_profile_model_settings": ("run_config", "_profile_model_settings"),
    "_responses_session_model": ("run_config", "_responses_session_model"),
    "_with_capability": ("run_config", "_with_capability"),
    "_DEFAULT_TOOL_RETRIES": ("tools", "_DEFAULT_TOOL_RETRIES"),
    "KEDI_APPROVAL_METADATA_KEY": ("tools", "KEDI_APPROVAL_METADATA_KEY"),
    "PydanticApprovalResolution": ("tools", "PydanticApprovalResolution"),
    "_approval_request_payload": ("tools", "_approval_request_payload"),
    "_approval_required_toolset": ("tools", "_approval_required_toolset"),
    "_external_tool_risk": ("tools", "_external_tool_risk"),
    "_external_tool_spec": ("tools", "_external_tool_spec"),
    "_is_async_callable": ("tools", "_is_async_callable"),
    "_kedi_tool_specs_from_toolsets": ("tools", "_kedi_tool_specs_from_toolsets"),
    "_KediPydanticTool": ("tools", "_KediPydanticTool"),
    "_mcp_toolset_from_config": ("tools", "_mcp_toolset_from_config"),
    "_model_retry_for_tool_error": ("tools", "_model_retry_for_tool_error"),
    "_preserve_tool_introspection": ("tools", "_preserve_tool_introspection"),
    "_toolset_sequence": ("tools", "_toolset_sequence"),
    "_validated_tool_arguments": ("tools", "_validated_tool_arguments"),
    "_validated_tool_function": ("tools", "_validated_tool_function"),
    "pydantic_approval_request_from_deferred": ("tools", "pydantic_approval_request_from_deferred"),
    "pydantic_deferred_results_from_decisions": (
        "tools",
        "pydantic_deferred_results_from_decisions",
    ),
    "pydantic_tool_from_spec": ("tools", "pydantic_tool_from_spec"),
    "pydantic_toolset_from_specs": ("tools", "pydantic_toolset_from_specs"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value
