from importlib import import_module
from typing import TYPE_CHECKING, Any

from .meta import (
    AdapterCapabilities,
    AdapterCapability,
    AgentAdapter,
    AgentAdapterKind,
    T,
)
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
    "AgentAdapterKind",
    "ACPAdapter",
    "BrowserModelBridge",
    "ClaudeAdapter",
    "CodexAdapter",
    "DSPyAdapter",
    "LangChainAdapter",
    "LazyAdapter",
    "PydanticAdapter",
    "SupportsToolRegistration",
    "T",
    "ToolSpec",
    "WebGPUAdapter",
    "adapters",
    "kedi_procedure_to_pydantic_tool",
    "kedi_program_to_pydantic_tools",
]
