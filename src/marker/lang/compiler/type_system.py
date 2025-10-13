"""Type system for Marker compiler.

Handles type resolution, coercion, and validation.
"""
from __future__ import annotations

import re
import shlex
from typing import Any, List
from pydantic import BaseModel, TypeAdapter
import ast

from ...core import MarkerRuntime
from ..ast import TypeDef, TypeField
from .environment import build_type_env


class MarkerType:
    """Represents a type annotation in Marker DSL and provides strict validation."""

    def __init__(self, annotation: str | None):
        self.annotation = annotation

    def resolve(self, rt: MarkerRuntime, env: dict[str, Any] | None = None) -> Any | None:
        if not self.annotation:
            return None
        type_env = build_type_env(rt, env)
        try:
            return eval(self.annotation, type_env, type_env)
        except Exception:
            try:
                from ..types import TypeValidator
                res = TypeValidator(self.annotation).validate_names()
                return res.pytype if res.is_valid else None
            except Exception:
                return None

    def is_none(self) -> bool:
        return self.annotation is None

    def get_adapter(self, rt: MarkerRuntime, env: dict[str, Any] | None = None) -> TypeAdapter | None:
        """Build a strict TypeAdapter via eval("TypeAdapter(<annotation>, config=CONFIG)")."""
        if not self.annotation:
            return None
        ns = build_type_env(rt, env)
        ns.update({
            'TypeAdapter': TypeAdapter,
            'CONFIG': {'strict': True, 'arbitrary_types_allowed': True},
        })
        try:
            expr = f"TypeAdapter({self.annotation}, config=CONFIG)"
            return eval(expr, ns, ns)
        except Exception:
            # Fallback to resolved type
            typ = self.resolve(rt, env)
            if typ is None:
                return None
            try:
                return TypeAdapter(typ, config={'strict': True, 'arbitrary_types_allowed': True})
            except Exception:
                return None

    def validate(self, rt: MarkerRuntime, env: dict[str, Any] | None, value: Any, *, allow_str_coercion: bool = False) -> Any:
        adapter = self.get_adapter(rt, env)
        if adapter is None:
            return value
        try:
            return adapter.validate_python(value)
        except Exception:
            raise

    def __repr__(self) -> str:
        return f"MarkerType({self.annotation!r})"

    def __str__(self) -> str:
        return self.annotation or "Any"


"""Minimal helpers kept below; complex coercion/compatibility removed in favor of TypeAdapter."""


def coerce_model_value(model: type[BaseModel], value: Any) -> Any:
    """Coerce a value to a Pydantic model."""
    if isinstance(value, model):
        return value
    
    if isinstance(value, dict):
        try:
            return model.model_validate(value)
        except Exception:
            return value
    
    if isinstance(value, str):
        text = value.strip()
        # Try JSON
        try:
            if text.startswith("{") and text.endswith("}"):
                return model.model_validate_json(text)
        except Exception:
            pass
        
        # Try key=value format
        parsed = _parse_model_repr(text, model)
        if parsed is not None:
            try:
                return model.model_validate(parsed)
            except Exception:
                return value
    
    return value


def is_value_compatible(marker_type: MarkerType, value: Any, rt: MarkerRuntime, env: dict[str, Any] | None = None) -> bool:
    try:
        marker_type.validate(rt, env, value)
        return True
    except Exception:
        return False


def register_type(rt: MarkerRuntime, typedef: TypeDef, type_env: dict[str, Any]) -> None:
    """Register a custom type definition as a Pydantic model."""
    annotations: dict[str, Any] = {}
    namespace: dict[str, Any] = {}
    
    for field in typedef.fields:
        anno = field.type_annotation or "str"
        marker_type = MarkerType(anno)
        resolved = marker_type.resolve(rt, env=type_env)
        if resolved is None:
            raise TypeError(f"Unknown type annotation '{anno}' in type {typedef.name}")
        annotations[field.name] = resolved
    
    namespace['__annotations__'] = annotations
    field_order = [field.name for field in typedef.fields]
    
    # Create model class
    new_model = type(typedef.name, (BaseModel,), namespace)
    setattr(new_model, '__init__', _make_init(typedef.name, field_order, new_model))
    
    type_env[typedef.name] = new_model
    type_env[f"__{typedef.name}_fields__"] = annotations


def assign_value_from_native(native: Any, marker_type: MarkerType, rt: MarkerRuntime, env: dict[str, Any] | None, *, strict_native: bool) -> Any:
    if strict_native:
        if marker_type.is_none():
            if isinstance(native, str):
                return native
            raise TypeError("Native non-string assignment requires explicit type annotation")
        # Preserve legacy message for str vs non-str annotation mismatch
        if isinstance(native, str):
            try_typ = marker_type.resolve(rt, env)
            if try_typ is not None and try_typ is not str:
                raise TypeError("Annotation type mismatch: got str value")
        try:
            return marker_type.validate(rt, env, native)
        except Exception:
            raise TypeError("Native assignment value does not match annotated type")
    else:
        # Angle-inline: always string unless annotated
        if not marker_type.is_none():
            raise TypeError("Angle-inline pure python cannot have a non-str annotation")
        return str(native)


def detect_rhs_kind(segments: list) -> tuple[bool, bool, bool]:
    """Detect the kind of RHS expression."""
    from ..ast import TextSegment, PythonExprNativeSegment, PythonExprSegment, CallSegment
    
    non_ws = [s for s in segments if not (isinstance(s, TextSegment) and s.text.strip() == "")]
    
    is_backtick_native = len(non_ws) == 1 and isinstance(non_ws[0], PythonExprNativeSegment)
    is_angle_pure = len(non_ws) == 1 and isinstance(non_ws[0], PythonExprSegment)
    is_call_only = len(non_ws) == 1 and isinstance(non_ws[0], CallSegment)
    
    return is_backtick_native, is_angle_pure, is_call_only


def _convert_basic_literal(raw: str, typ: Any) -> Any:
    """Convert a string to a basic literal type."""
    try:
        if typ in (int, float, str):
            return typ(raw)
        if typ is bool:
            if raw.lower() in {"true", "1", "yes"}:
                return True
            if raw.lower() in {"false", "0", "no"}:
                return False
            return bool(raw)
    except Exception:
        return raw
    return raw


def _parse_model_repr(value: str, model: type[BaseModel]) -> dict[str, Any] | None:
    """Parse key=value format into dict for model."""
    text = value.strip()
    if not text:
        return None
    
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    
    data: dict[str, Any] = {}
    fields = getattr(model, "model_fields", {})
    
    for token in tokens:
        if "=" not in token:
            return None
        name, raw = token.split("=", 1)
        if name not in fields:
            return None
        field = fields[name]
        annotation = getattr(field, "annotation", str) or str
        data[name] = _convert_basic_literal(raw, annotation)
    
    return data


def _make_init(type_name: str, field_order: list[str], model_class: type[BaseModel]):
    """Create an __init__ that accepts positional args in field order."""
    def _init(self, *args: Any, **kwargs: Any) -> None:
        if args:
            if len(args) > len(field_order):
                raise TypeError(f"Too many positional arguments for {type_name}")
            for name, value in zip(field_order, args):
                if name in kwargs:
                    raise TypeError(f"Multiple values for field '{name}' in {type_name}")
                kwargs[name] = value
        super(model_class, self).__init__(**kwargs)
    return _init


# Complex list coercion/compatibility helpers removed


def push_type_env(rt: MarkerRuntime) -> dict[str, Any]:
    """Push a new type environment layer."""
    stack = getattr(rt, "_type_env_stack", None)
    if stack is None:
        stack = []
        setattr(rt, "_type_env_stack", stack)
    new_env: dict[str, Any] = {}
    stack.append(new_env)
    return new_env


def pop_type_env(rt: MarkerRuntime) -> None:
    """Pop the current type environment layer."""
    stack = getattr(rt, "_type_env_stack", None)
    if isinstance(stack, list) and stack:
        stack.pop()