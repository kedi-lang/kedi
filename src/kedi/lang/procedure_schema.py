"""JSON Schema helpers for Kedi procedures."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, TypeAdapter

from ..core import KediRuntime
from ..dynamic_models import build_dynamic_model
from .ast import ProcDefStmt, ProcedureDef, Program, ProgramWithValidation
from .compiler.environment import build_type_env
from .compiler.type_system import KediType
from .docstrings import parse_procedure_docstring

_INTROSPECTION_STUB_DOC = "This stub is generated for Python introspection."


def procedure_json_schema(
    rt: KediRuntime,
    proc: ProcedureDef,
    local_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON Schema for a procedure's input arguments."""
    parsed_docstring = parse_procedure_docstring(proc.docstring)
    default_env = build_type_env(rt, local_env or {})
    field_defs: dict[str, tuple[Any, Any]] = {}

    for param_name in proc.params:
        param_type = _resolve_param_type(rt, proc, param_name, local_env)
        default_expr = (proc.param_defaults or {}).get(param_name)
        if default_expr is None:
            field_defs[param_name] = (param_type, ...)
            continue
        default_value = rt.executor.evaluate_type_expression(rt, default_expr, default_env)
        field_defs[param_name] = (param_type, copy.deepcopy(default_value))

    model = build_dynamic_model(
        f"_{proc.name}Parameters",
        BaseModel,
        field_defs,
        __config__=ConfigDict(extra="forbid", title=proc.name),
    )
    schema = _strip_introspection_descriptions(model.model_json_schema())
    schema["description"] = build_procedure_description(
        rt,
        proc,
        parsed_docstring.description,
        local_env=local_env,
    )
    _apply_parameter_descriptions(schema, parsed_docstring.parameter_descriptions)
    return schema


def procedure_return_json_schema(
    rt: KediRuntime,
    proc: ProcedureDef,
    local_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON Schema for a procedure's return value."""
    parsed_docstring = parse_procedure_docstring(proc.docstring)
    kedi_type = KediType.from_node(proc.return_type_expr, proc.return_type_annotation)
    if kedi_type.is_none():
        schema = TypeAdapter(str).json_schema()
        if parsed_docstring.return_description:
            schema["description"] = parsed_docstring.return_description
        return schema
    resolved = kedi_type.resolve(rt, local_env)
    if resolved is None:
        raise TypeError(f"Unknown return type annotation in procedure {proc.name!r}")
    schema = _strip_introspection_descriptions(TypeAdapter(resolved).json_schema())
    if parsed_docstring.return_description:
        schema["description"] = parsed_docstring.return_description
    return schema


def program_procedure_json_schemas(
    rt: KediRuntime,
    prog: Program | ProgramWithValidation,
    *,
    include_nested: bool = False,
    local_env: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build input JSON Schemas for procedures in a parsed program."""
    procedures = _all_procedures(prog.procedures, include_nested=include_nested)
    return {proc.name: procedure_json_schema(rt, proc, local_env=local_env) for proc in procedures}


def build_procedure_description(
    rt: KediRuntime,
    proc: ProcedureDef,
    base_description: str | None,
    *,
    local_env: dict[str, Any] | None,
) -> str:
    """Build a human-readable procedure description with a return contract."""
    parts: list[str] = []
    if base_description:
        parts.append(base_description.strip())

    resolved = KediType.from_node(
        proc.return_type_expr,
        proc.return_type_annotation,
    ).resolve(rt, local_env)
    return_type_name = _return_type_name(resolved)
    parts.append(f"Returns `{return_type_name}` instance.")
    return " ".join(part for part in parts if part)


def _resolve_param_type(
    rt: KediRuntime,
    proc: ProcedureDef,
    param_name: str,
    local_env: dict[str, Any] | None,
) -> Any:
    kedi_type = KediType.from_node(
        (proc.param_type_exprs or {}).get(param_name),
        (proc.param_type_annotations or {}).get(param_name),
    )
    if kedi_type.is_none():
        return str
    resolved = kedi_type.resolve(rt, local_env)
    if resolved is None:
        raise TypeError(
            f"Unknown parameter type annotation for {param_name!r} in procedure {proc.name!r}"
        )
    return resolved


def _all_procedures(
    procedures: Iterable[ProcedureDef], *, include_nested: bool
) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        if not include_nested:
            return
        for stmt in proc.body:
            if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
                visit(stmt.proc)

    for proc in procedures:
        visit(proc)
    return out


def _strip_introspection_descriptions(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "description" and item == _INTROSPECTION_STUB_DOC:
                continue
            cleaned[key] = _strip_introspection_descriptions(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_introspection_descriptions(item) for item in value]
    return value


def _apply_parameter_descriptions(
    schema: dict[str, Any],
    descriptions: dict[str, str],
) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return

    for param_name, description in descriptions.items():
        prop = properties.get(param_name)
        if isinstance(prop, dict):
            prop.setdefault("description", description)


def _return_type_name(resolved: Any) -> str:
    if isinstance(resolved, type):
        return resolved.__name__
    if hasattr(resolved, "__name__"):
        return str(resolved.__name__)
    return str(resolved) if resolved is not None else "str"
