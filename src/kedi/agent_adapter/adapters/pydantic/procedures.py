from __future__ import annotations

from dataclasses import replace
from typing import (
    Any,
)

from pydantic_ai.tools import Tool

from kedi.agent_adapter.tools.conversion import _all_procedures, kedi_procedure_tool_spec
from kedi.core import KediRuntime
from kedi.lang.ast import ProcedureDef, Program, ProgramWithValidation


def kedi_procedure_to_pydantic_tool(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
    sequential: bool = False,
    args_validator: Any = None,
) -> Tool:
    """Build a ``pydantic_ai.Tool`` from a Kedi procedure."""
    from kedi.agent_adapter.adapters.pydantic import pydantic_tool_from_spec

    spec = kedi_procedure_tool_spec(rt, proc, local_env=local_env)
    return pydantic_tool_from_spec(
        replace(spec, sequential=sequential),
        args_validator=args_validator,
    )


def kedi_program_to_pydantic_tools(
    rt: KediRuntime,
    prog: Program | ProgramWithValidation,
    *,
    include_nested: bool = False,
    local_env: dict[str, Any] | None = None,
    sequential: bool = False,
    args_validator: Any = None,
) -> list[Tool]:
    """Build ``pydantic_ai.Tool`` objects for procedures in a parsed program."""
    from kedi.agent_adapter.adapters.pydantic import pydantic_tool_from_spec

    tools: list[Tool] = []
    for proc in _all_procedures(prog.procedures, include_nested=include_nested):
        spec = kedi_procedure_tool_spec(rt, proc, local_env=local_env)
        spec = replace(spec, sequential=sequential)
        tools.append(
            pydantic_tool_from_spec(
                spec,
                args_validator=args_validator,
            )
        )
    return tools
