from .meta import AgentAdapter, T
from .tool_override import SupportsToolRegistration


def kedi_procedure_to_pydantic_tool(*args, **kwargs):
    from .utils import kedi_procedure_to_pydantic_tool as build_tool

    return build_tool(*args, **kwargs)


def kedi_program_to_pydantic_tools(*args, **kwargs):
    from .utils import kedi_program_to_pydantic_tools as build_tools

    return build_tools(*args, **kwargs)


__all__ = [
    "AgentAdapter",
    "SupportsToolRegistration",
    "T",
    "kedi_procedure_to_pydantic_tool",
    "kedi_program_to_pydantic_tools",
]
