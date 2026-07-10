from typing import Any

from .claude import ClaudeAdapter
from .dspy import DSPyAdapter
from .langchain import LangChainAdapter
from .meta import (
    AdapterCapabilities,
    AdapterCapability,
    AgentAdapter,
    AgentAdapterKind,
    T,
)
from .pydantic import PydanticAdapter
from .tool_override import SupportsToolRegistration, ToolSpec
from .webgpu import BrowserModelBridge, WebGPUAdapter


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
    "BrowserModelBridge",
    "ClaudeAdapter",
    "DSPyAdapter",
    "LangChainAdapter",
    "PydanticAdapter",
    "SupportsToolRegistration",
    "T",
    "ToolSpec",
    "WebGPUAdapter",
    "kedi_procedure_to_pydantic_tool",
    "kedi_program_to_pydantic_tools",
]
