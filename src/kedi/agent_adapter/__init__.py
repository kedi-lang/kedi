from importlib import import_module
from typing import TYPE_CHECKING, Any

from .approvals.policy import ApprovalCancelled
from .compaction.models import HistoryOrigin
from .conversation import (
    AdapterConversationContext,
    AdapterStateSnapshot,
    ConversationState,
    ConversationTurn,
    SupportsConversationState,
)
from .events import (
    AgentEventCallback,
    AgentEventDispatcher,
    AgentEventSink,
    AgentMessageEvent,
    AgentMessagePhase,
    AgentRunState,
    AgentRunStateEvent,
    AgentStreamEvent,
    AsyncAgentEventQueue,
    bind_agent_event_dispatcher,
    observe_agent_events,
)
from .history_processing import (
    HistoryProcessingError,
    HistoryProcessingValidationError,
    HistoryProcessorCallback,
    HistoryProcessorCondition,
    HistoryProcessorContext,
    HistoryProcessorState,
)
from .hooks import (
    HOOK_EVENT_NAMES,
    AdapterHookSnapshot,
    HookContext,
    HookDecisionKind,
    HookDeniedError,
    HookEvent,
    HookEventName,
    HookExecutionError,
    HookHandler,
    HookRegistrationMixin,
    HookResult,
    HookTimeoutError,
    PostToolResultFailureEvent,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreToolUseDecision,
    PreToolUseRequest,
    ToolHookContext,
    ToolOrigin,
    UserPromptSubmitDecision,
    UserPromptSubmitRequest,
)
from .meta import (
    AdapterCapabilities,
    AdapterCapability,
    AgentAdapter,
    AgentAdapterKind,
    HookCapabilities,
    HookCapabilityEvent,
    HookEventCapability,
    T,
)
from .models import (
    CodexModel,
    CodexModelAdapter,
    CodexResponsesConnection,
    CodexResponsesFallback,
    codex_responses_model,
)
from .requirements import (
    MissingAdapterCapabilityError,
    RequirementCheck,
    check_adapter_requirements,
    normalize_adapter_requirements,
    validate_adapter_requirements,
)
from .subagents import SubagentUsageLimits
from .subagents.context import SubagentExecutionContext, current_subagent_execution
from .subagents.native import SubagentResult, SubagentScope, SubagentTask
from .tools.records import ToolCallRecord
from .tools.specs import SupportsToolRegistration, ToolSpec

if TYPE_CHECKING:
    from . import adapters
    from .adapters import (
        A2AAdapter,
        ACPAdapter,
        BrowserModelBridge,
        ClaudeAdapter,
        CodexAdapter,
        DSPyAdapter,
        LangChainAdapter,
        LazyAdapter,
        PydanticAdapter,
        PydanticCodeModeCapability,
        WebGPUAdapter,
        pydantic_approval_request_from_deferred,
        pydantic_code_mode_capability,
        pydantic_deferred_results_from_decisions,
        pydantic_tool_from_spec,
        pydantic_toolset_from_specs,
    )


_ADAPTER_EXPORTS = frozenset(
    {
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
    }
)


def __getattr__(name: str) -> Any:
    if name == "adapters" or name in _ADAPTER_EXPORTS:
        adapter_module = import_module(".adapters", __name__)
        value = adapter_module if name == "adapters" else getattr(adapter_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def kedi_procedure_to_pydantic_tool(*args: Any, **kwargs: Any) -> Any:
    from .adapters.pydantic.procedures import kedi_procedure_to_pydantic_tool as build_tool

    return build_tool(*args, **kwargs)


def kedi_program_to_pydantic_tools(*args: Any, **kwargs: Any) -> Any:
    from .adapters.pydantic.procedures import kedi_program_to_pydantic_tools as build_tools

    return build_tools(*args, **kwargs)


__all__ = [
    "A2AAdapter",
    "AgentAdapter",
    "AdapterHookSnapshot",
    "AdapterCapabilities",
    "AdapterCapability",
    "AdapterConversationContext",
    "AdapterStateSnapshot",
    "AgentEventCallback",
    "AgentEventDispatcher",
    "AgentEventSink",
    "AgentMessageEvent",
    "AgentMessagePhase",
    "AgentRunState",
    "AgentRunStateEvent",
    "AgentStreamEvent",
    "AgentAdapterKind",
    "HOOK_EVENT_NAMES",
    "HookContext",
    "HookCapabilities",
    "HookCapabilityEvent",
    "HookDecisionKind",
    "HookDeniedError",
    "HookEvent",
    "HookEventName",
    "HookExecutionError",
    "HookEventCapability",
    "HookHandler",
    "HookRegistrationMixin",
    "HookResult",
    "HookTimeoutError",
    "HistoryProcessingError",
    "HistoryOrigin",
    "HistoryProcessingValidationError",
    "HistoryProcessorCallback",
    "HistoryProcessorCondition",
    "HistoryProcessorContext",
    "HistoryProcessorState",
    "MissingAdapterCapabilityError",
    "ApprovalCancelled",
    "AsyncAgentEventQueue",
    "bind_agent_event_dispatcher",
    "ACPAdapter",
    "BrowserModelBridge",
    "ClaudeAdapter",
    "CodexAdapter",
    "CodexModel",
    "CodexModelAdapter",
    "CodexResponsesConnection",
    "CodexResponsesFallback",
    "ConversationState",
    "ConversationTurn",
    "DSPyAdapter",
    "LangChainAdapter",
    "LazyAdapter",
    "PydanticAdapter",
    "PydanticCodeModeCapability",
    "PostToolUseEvent",
    "PostToolUseFailureEvent",
    "PostToolResultFailureEvent",
    "PreToolUseDecision",
    "PreToolUseRequest",
    "RequirementCheck",
    "pydantic_approval_request_from_deferred",
    "pydantic_code_mode_capability",
    "pydantic_deferred_results_from_decisions",
    "pydantic_tool_from_spec",
    "pydantic_toolset_from_specs",
    "SupportsToolRegistration",
    "SupportsConversationState",
    "SubagentExecutionContext",
    "SubagentUsageLimits",
    "SubagentResult",
    "SubagentScope",
    "SubagentTask",
    "T",
    "ToolSpec",
    "ToolCallRecord",
    "ToolHookContext",
    "ToolOrigin",
    "UserPromptSubmitDecision",
    "UserPromptSubmitRequest",
    "WebGPUAdapter",
    "adapters",
    "check_adapter_requirements",
    "kedi_procedure_to_pydantic_tool",
    "kedi_program_to_pydantic_tools",
    "observe_agent_events",
    "codex_responses_model",
    "current_subagent_execution",
    "normalize_adapter_requirements",
    "validate_adapter_requirements",
]
