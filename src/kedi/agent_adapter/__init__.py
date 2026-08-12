from importlib import import_module
from typing import TYPE_CHECKING, Any

from .approval import ApprovalCancelled
from .conversation import (
    AdapterConversationContext,
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
    observe_agent_events,
)
from .meta import (
    AdapterCapabilities,
    AdapterCapability,
    AgentAdapter,
    AgentAdapterKind,
    T,
)
from .models import CodexModel, CodexModelAdapter, codex_responses_model
from .tool_override import SupportsToolRegistration, ToolSpec

if TYPE_CHECKING:
    from . import adapters
    from .adapters import (
        ACPAdapter,
        BrowserModelBridge,
        ClaudeAdapter,
        CodexAdapter,
        DSPyAdapter,
        LangChainAdapter,
        LazyAdapter,
        PydanticAdapter,
        WebGPUAdapter,
        pydantic_approval_request_from_deferred,
        pydantic_deferred_results_from_decisions,
        pydantic_tool_from_spec,
        pydantic_toolset_from_specs,
    )


_ADAPTER_EXPORTS = frozenset(
    {
        "ACPAdapter",
        "BrowserModelBridge",
        "ClaudeAdapter",
        "CodexAdapter",
        "DSPyAdapter",
        "LangChainAdapter",
        "LazyAdapter",
        "PydanticAdapter",
        "pydantic_approval_request_from_deferred",
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
    from .utils import kedi_procedure_to_pydantic_tool as build_tool

    return build_tool(*args, **kwargs)


def kedi_program_to_pydantic_tools(*args: Any, **kwargs: Any) -> Any:
    from .utils import kedi_program_to_pydantic_tools as build_tools

    return build_tools(*args, **kwargs)


__all__ = [
    "AgentAdapter",
    "AdapterCapabilities",
    "AdapterCapability",
    "AdapterConversationContext",
    "AgentEventCallback",
    "AgentEventDispatcher",
    "AgentEventSink",
    "AgentMessageEvent",
    "AgentMessagePhase",
    "AgentRunState",
    "AgentRunStateEvent",
    "AgentStreamEvent",
    "AgentAdapterKind",
    "ApprovalCancelled",
    "AsyncAgentEventQueue",
    "ACPAdapter",
    "BrowserModelBridge",
    "ClaudeAdapter",
    "CodexAdapter",
    "CodexModel",
    "CodexModelAdapter",
    "ConversationState",
    "ConversationTurn",
    "DSPyAdapter",
    "LangChainAdapter",
    "LazyAdapter",
    "PydanticAdapter",
    "pydantic_approval_request_from_deferred",
    "pydantic_deferred_results_from_decisions",
    "pydantic_tool_from_spec",
    "pydantic_toolset_from_specs",
    "SupportsToolRegistration",
    "SupportsConversationState",
    "T",
    "ToolSpec",
    "WebGPUAdapter",
    "adapters",
    "kedi_procedure_to_pydantic_tool",
    "kedi_program_to_pydantic_tools",
    "observe_agent_events",
    "codex_responses_model",
]
