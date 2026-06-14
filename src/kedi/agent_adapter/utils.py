from __future__ import annotations

import asyncio
import copy
import inspect
from functools import wraps
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_ai.tools import Tool

from ..core import CallExpr, InputExpr, KediRuntime
from ..lang.ast import ProcedureDef, Program, ProgramWithValidation
from ..lang.compiler.type_system import KediType, coerce_model_value
from ..lang.docstrings import parse_procedure_docstring
from ..lang.procedure_schema import (
    build_procedure_description,
    procedure_json_schema,
    procedure_return_json_schema,
    program_procedure_json_schemas,
)
from .tool_override import ToolSpec

_DSPY_SYNC_TOOL_MARKER = "__kedi_dspy_sync_tool__"


def kedi_procedure_tool_spec(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
) -> ToolSpec:
    """Build an adapter-neutral tool spec from a Kedi procedure."""
    parsed_docstring = parse_procedure_docstring(proc.docstring)
    json_schema = procedure_json_schema(rt, proc, local_env=local_env)
    return_schema = procedure_return_json_schema(rt, proc, local_env=local_env)
    tool_description = _build_tool_description(
        rt,
        proc,
        parsed_docstring.description,
        local_env=local_env,
    )
    return ToolSpec(
        fn=_make_tool_function(rt, proc),
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
    )


def python_callable_tool_spec(name: str, fn: Callable[..., Any]) -> ToolSpec:
    """Build an adapter-neutral tool spec from a Python callable."""
    return ToolSpec(fn=fn, name=name, description=_callable_description(fn))


def kedi_procedure_to_pydantic_tool(
    rt: KediRuntime,
    proc: ProcedureDef,
    *,
    local_env: dict[str, Any] | None = None,
    sequential: bool = False,
    args_validator: Any = None,
) -> Tool:
    """Build a ``pydantic_ai.Tool`` from a Kedi procedure."""
    spec = kedi_procedure_tool_spec(rt, proc, local_env=local_env)
    assert spec.json_schema is not None
    tool = Tool.from_schema(
        function=spec.fn,
        name=spec.name,
        description=spec.description or "",
        json_schema=spec.json_schema,
        takes_ctx=False,
        sequential=sequential,
        args_validator=args_validator,
    )
    tool.metadata = spec.metadata
    return tool


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
        tool = Tool.from_schema(
            function=_make_tool_function(rt, proc),
            name=proc.name,
            description=tool_description,
            json_schema=json_schema,
            takes_ctx=False,
            sequential=sequential,
            args_validator=args_validator,
        )
        tool.metadata = _tool_metadata(
            proc,
            json_schema,
            return_schema,
            description=tool_description,
        )
        tools.append(tool)
    return tools


def dspy_sync_tool_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a sync or async callable to DSPy's sync tool-call path."""
    if getattr(fn, _DSPY_SYNC_TOOL_MARKER, False):
        return fn

    target = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else fn.__call__

    @wraps(target)
    def _sync_tool(*args: Any, **kwargs: Any) -> Any:
        result = target(*args, **kwargs)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    setattr(_sync_tool, _DSPY_SYNC_TOOL_MARKER, True)
    return _sync_tool


def dspy_tool_args(json_schema: dict[str, Any]) -> dict[str, Any]:
    defs = json_schema.get("$defs", {})
    return {
        name: _resolve_dspy_arg_schema(schema, defs)
        for name, schema in json_schema.get("properties", {}).items()
    }


def dspy_tool_arg_desc(json_schema: dict[str, Any]) -> dict[str, str]:
    properties = json_schema.get("properties", {})
    return {
        name: description
        for name, schema in properties.items()
        if isinstance(schema, dict) and isinstance((description := schema.get("description")), str)
    }


def _make_tool_function(rt: KediRuntime, proc: ProcedureDef) -> Callable[..., Any]:
    proc_fn = _get_or_compile_proc_function(rt, proc)

    @wraps(proc_fn)
    def _tool_function(**kwargs: Any) -> Any:
        return _invoke_procedure_with_kwargs(rt, proc, proc_fn, kwargs)

    _tool_function.__name__ = proc.name
    _tool_function.__doc__ = getattr(proc_fn, "__doc__", None)
    return _tool_function


def _resolve_dspy_arg_schema(
    schema: dict[str, Any],
    defs: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(schema)
    ref = resolved.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved = copy.deepcopy(defs[ref.removeprefix("#/$defs/")])
    return _strip_json_schema_titles(resolved)


def _strip_json_schema_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_json_schema_titles(item) for key, item in value.items() if key != "title"
        }
    if isinstance(value, list):
        return [_strip_json_schema_titles(item) for item in value]
    return value


def _callable_description(fn: Callable[..., Any]) -> str | None:
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)) and hasattr(fn, "__call__"):
        return inspect.getdoc(fn.__call__) or inspect.getdoc(fn)
    return inspect.getdoc(fn)


def _get_or_compile_proc_function(rt: KediRuntime, proc: ProcedureDef) -> Callable[..., Any]:
    try:
        return rt._get_proc(proc.name)
    except KeyError:
        from ..lang.compiler.procedure_compiler import ProcedureCompiler

        return ProcedureCompiler(rt)._make_procedure_impl(proc, proc.lexical_env)


def _invoke_procedure_with_kwargs(
    rt: KediRuntime,
    proc: ProcedureDef,
    fn: Any,
    kwargs: dict[str, Any],
) -> Any:
    sig = inspect.signature(fn)
    bound = sig.bind_partial(**kwargs)
    bound.apply_defaults()
    local_env = dict(bound.arguments)
    _coerce_tool_arguments(rt, proc, local_env)
    call_kwargs = {name: local_env[name] for name in bound.arguments}

    token = rt._current_env.set(local_env)
    try:
        result = fn(**call_kwargs)
    finally:
        rt._current_env.reset(token)

    return (
        rt._evaluate_value(result, local_env)
        if isinstance(result, (InputExpr, CallExpr))
        else result
    )


def _coerce_tool_arguments(
    rt: KediRuntime,
    proc: ProcedureDef,
    kwargs: dict[str, Any],
) -> None:
    if not (proc.param_type_annotations or proc.param_type_exprs):
        return
    param_exprs = proc.param_type_exprs or {}
    param_strs = proc.param_type_annotations or {}

    for param_name in proc.params:
        if param_name not in kwargs:
            raise TypeError(f"Missing parameter: {param_name}")

        kedi_type = KediType.from_node(
            param_exprs.get(param_name),
            param_strs.get(param_name),
        )
        if kedi_type.is_none():
            continue

        provided = kwargs[param_name]
        resolved = kedi_type.resolve(rt, None)
        try:
            if isinstance(resolved, type) and issubclass(resolved, BaseModel):
                kwargs[param_name] = coerce_model_value(resolved, provided)
            else:
                kwargs[param_name] = kedi_type.validate(
                    rt, None, provided, allow_str_coercion=False
                )
        except Exception as exc:
            raise TypeError(f"Parameter '{param_name}' does not match annotated type") from exc


def _tool_metadata(
    proc: ProcedureDef,
    json_schema: dict[str, Any],
    return_schema: dict[str, Any],
    *,
    description: str | None,
) -> dict[str, Any]:
    return {
        "kedi_schema": {
            "name": proc.name,
            "parameters": json_schema,
            "returns": return_schema,
            **({"description": description} if description else {}),
        },
        "kedi_input_schema": json_schema,
        "kedi_return_schema": return_schema,
    }


def _build_tool_description(
    rt: KediRuntime,
    proc: ProcedureDef,
    base_description: str | None,
    *,
    local_env: dict[str, Any] | None,
) -> str:
    return build_procedure_description(rt, proc, base_description, local_env=local_env)


def _all_procedures(procedures: list[ProcedureDef], *, include_nested: bool) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        if not include_nested:
            return
        for stmt in proc.body:
            nested = getattr(stmt, "proc", None)
            if nested is not None:
                visit(nested)

    for proc in procedures:
        visit(proc)
    return out
