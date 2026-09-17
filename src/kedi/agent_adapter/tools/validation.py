from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Mapping,
    get_type_hints,
)

from pydantic import BaseModel, TypeAdapter

from kedi.core import KediRuntime
from kedi.lang.ast import ProcedureDef
from kedi.lang.compiler.type_system import KediType, coerce_model_value


def _procedure_tool_argument_validator(
    rt: KediRuntime,
    proc: ProcedureDef,
    tool_fn: Callable[..., Any],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    proc_fn = getattr(tool_fn, "__wrapped__", tool_fn)

    def validate(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return _validated_procedure_kwargs(rt, proc, proc_fn, dict(arguments))

    return validate


def _validated_procedure_kwargs(
    rt: KediRuntime,
    proc: ProcedureDef,
    fn: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    bound = sig.bind(**kwargs)
    bound.apply_defaults()
    local_env = dict(bound.arguments)
    _coerce_tool_arguments(rt, proc, local_env)
    return {name: local_env[name] for name in bound.arguments}


def _python_callable_argument_validator(
    fn: Callable[..., Any],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    signature = inspect.signature(fn)
    try:
        type_hints = get_type_hints(fn)
    except (NameError, TypeError):
        type_hints = {}

    def validate(arguments: Mapping[str, Any]) -> dict[str, Any]:
        bound = signature.bind(**arguments)
        bound.apply_defaults()
        validated: dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            value = bound.arguments.get(name)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                validated.update(value or {})
                continue
            annotation = type_hints.get(name, parameter.annotation)
            if isinstance(annotation, str):
                annotation = inspect.Parameter.empty
            if annotation is not inspect.Parameter.empty:
                value = TypeAdapter(annotation).validate_python(value)
            validated[name] = value
        return validated

    return validate


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
                kwargs[param_name] = kedi_type.validate(rt, None, provided)
        except Exception as exc:
            raise TypeError(f"Parameter '{param_name}' does not match annotated type") from exc
