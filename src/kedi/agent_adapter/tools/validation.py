from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Mapping,
    get_type_hints,
)

from pydantic import BaseModel, TypeAdapter

from kedi.agent_adapter.request.media import media_validation_context
from kedi.core import KediRuntime
from kedi.lang.ast import ProcedureDef, TypeRef
from kedi.lang.compiler.type_system import KediType, coerce_model_value
from kedi.media import media_error_detail

from .errors import ToolArgumentError


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
    bound = _bind_arguments(sig, kwargs)
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
        bound = _bind_arguments(signature, arguments)
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
                # A media argument names one of the request's media.
                value = TypeAdapter(annotation).validate_python(
                    value, context=media_validation_context()
                )
            validated[name] = value
        return validated

    return validate


def _coerce_tool_arguments(
    rt: KediRuntime,
    proc: ProcedureDef,
    kwargs: dict[str, Any],
) -> None:
    param_exprs = proc.param_type_exprs or {}
    param_strs = proc.param_type_annotations or {}

    for param_name in proc.params:
        if param_name not in kwargs:
            raise ToolArgumentError(f"Missing parameter: {param_name}")

        kedi_type = KediType.from_node(
            param_exprs.get(param_name),
            param_strs.get(param_name),
        )
        if kedi_type.is_none():
            kedi_type = KediType(TypeRef(name="str"))

        provided = kwargs[param_name]
        resolved = kedi_type.resolve(rt, None)
        # A media argument names one of the request's media.
        context = media_validation_context()
        try:
            if isinstance(resolved, type) and issubclass(resolved, BaseModel):
                kwargs[param_name] = coerce_model_value(resolved, provided, context=context)
            else:
                kwargs[param_name] = kedi_type.validate(rt, None, provided, context=context)
        except ValueError as exc:
            # The model reads this error: say which media names it can use.
            detail = media_error_detail(exc)
            raise ToolArgumentError(
                f"Parameter '{param_name}' does not match annotated type"
                + (f": {detail}" if detail else "")
            ) from exc


def _bind_arguments(
    signature: inspect.Signature, arguments: Mapping[str, Any]
) -> inspect.BoundArguments:
    try:
        return signature.bind(**arguments)
    except TypeError as exc:
        raise ToolArgumentError(str(exc)) from exc
