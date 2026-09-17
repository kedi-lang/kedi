from __future__ import annotations

from typing import (
    Any,
)

from pydantic_ai.tools import Tool

from kedi.agent_adapter.tools.conversion import (
    _all_procedures,
    _build_tool_description,
    _make_tool_function,
    _procedure_tool_risk,
    _procedure_tool_risk_resolver,
    _tool_metadata,
    kedi_procedure_tool_spec,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_adapter.tools.validation import _procedure_tool_argument_validator
from kedi.core import KediRuntime
from kedi.lang.ast import ProcedureDef, Program, ProgramWithValidation
from kedi.lang.docstrings import parse_procedure_docstring
from kedi.lang.procedure_schema import procedure_return_json_schema, program_procedure_json_schemas


def kedi_procedure_to_pydantic_tool(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
    sequential: bool = False,
    args_validator: Any = None,
) -> Tool:
    """Build a ``pydantic_ai.Tool`` from a Kedi procedure."""
    from dataclasses import replace

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

    schemas = program_procedure_json_schemas(
        rt,
        prog,
        include_nested=include_nested,
        local_env=local_env,
    )
    procedure_map = {
        proc.name: proc for proc in _all_procedures(prog.procedures, include_nested=include_nested)
    }
    tools: list[Tool] = []
    for name, json_schema in schemas.items():
        proc = procedure_map[name]
        parsed_docstring = parse_procedure_docstring(proc.docstring)
        return_schema = procedure_return_json_schema(rt, proc, local_env=local_env)
        tool_description = _build_tool_description(
            rt,
            proc,
            parsed_docstring.description,
            local_env=local_env,
        )
        tool_fn = _make_tool_function(rt, proc)
        spec = ToolSpec(
            fn=tool_fn,
            name=proc.name,
            description=tool_description,
            json_schema=json_schema,
            return_schema=return_schema,
            metadata=_tool_metadata(
                proc,
                json_schema,
                return_schema,
                description=tool_description,
            ),
            proc=proc,
            risk=_procedure_tool_risk(proc),
            risk_resolver=_procedure_tool_risk_resolver(proc),
            argument_validator=_procedure_tool_argument_validator(
                rt,
                proc,
                tool_fn,
            ),
            sequential=sequential,
        )
        tools.append(
            pydantic_tool_from_spec(
                spec,
                args_validator=args_validator,
            )
        )
    return tools
