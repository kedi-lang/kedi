"""Type system for Marker compiler.

Handles type resolution, coercion, and validation.
"""
from __future__ import annotations

import re
import shlex
from typing import Any, List
from pydantic import BaseModel, TypeAdapter

from ...core import MarkerRuntime
from ..ast import TypeDef, TypeField
from .environment import build_type_env


class MarkerType:
    """Represents a type annotation in Marker DSL.
    
    Preserves the annotation string and can resolve it to a Python type
    using the runtime's type environment.
    """
    
    def __init__(self, annotation: str | None):
        """Create a MarkerType from an annotation string."""
        self.annotation = annotation
    
    def resolve(self, rt: MarkerRuntime, env: dict[str, Any] | None = None) -> Any | None:
        """Resolve the annotation to a Python type using the runtime environment."""
        if not self.annotation:
            return None
        
        try:
            type_env = build_type_env(rt, env)
            return eval(self.annotation, type_env, type_env)
        except Exception:
            try:
                from ..types import TypeValidator
                res = TypeValidator(self.annotation).validate_names()
                return res.pytype if res.is_valid else None
            except Exception:
                return None
    
    def is_none(self) -> bool:
        """Check if this represents an untyped value."""
        return self.annotation is None
    
    def get_type_adapter(self, rt: MarkerRuntime, env: dict[str, Any] | None = None) -> TypeAdapter | None:
        """Create a Pydantic TypeAdapter using the annotation string."""
        if not self.annotation:
            return None
        
        try:
            # Create adapter with the annotation string directly
            return TypeAdapter(self.annotation, config={'arbitrary_types_allowed': True})
        except Exception:
            # Fallback: try resolving and creating adapter from resolved type
            resolved = self.resolve(rt, env)
            if resolved is not None:
                try:
                    return TypeAdapter(resolved, config={'arbitrary_types_allowed': True})
                except Exception:
                    return None
            return None
    
    def __repr__(self) -> str:
        return f"MarkerType({self.annotation!r})"
    
    def __str__(self) -> str:
        return self.annotation or "Any"


def coerce_value_to_type(
    value: Any,
    marker_type: MarkerType,
    rt: MarkerRuntime,
    env: dict[str, Any] | None = None
) -> Any:
    """Coerce a value to match the expected type."""
    if marker_type.is_none():
        return value
    
    pytype = marker_type.resolve(rt, env)
    if pytype is None:
        return value
    
    return _coerce_by_pytype(value, pytype)


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


def is_value_compatible(
    marker_type: MarkerType,
    value: Any,
    rt: MarkerRuntime,
    env: dict[str, Any] | None = None
) -> bool:
    """Check if a value is compatible with a type."""
    if marker_type.is_none():
        return True
    
    pytype = marker_type.resolve(rt, env)
    if pytype is None:
        return True
    
    return _is_compatible_by_pytype(value, pytype)


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


def assign_value_from_native(
    native: Any,
    marker_type: MarkerType,
    rt: MarkerRuntime,
    env: dict[str, Any] | None,
    *,
    strict_native: bool
) -> Any:
    """Assign a value from native Python, applying type rules."""
    pytype = marker_type.resolve(rt, env) if not marker_type.is_none() else None
    
    if strict_native:
        # Backtick-native: strict typing
        if isinstance(pytype, type) and issubclass(pytype, BaseModel):
            coerced = coerce_model_value(pytype, native)
            if not isinstance(coerced, pytype):
                raise TypeError("Native assignment value does not match annotated type")
            return coerced
        
        if isinstance(native, str):
            if pytype is not None and pytype is not str:
                raise TypeError("Annotation type mismatch: got str value")
            return native
        
        # Non-string requires explicit annotation
        if pytype is None:
            raise TypeError("Native non-string assignment requires explicit type annotation")
        if not is_value_compatible(marker_type, native, rt, env):
            raise TypeError("Native assignment value does not match annotated type")
        return native
    else:
        # Angle-inline: always string
        if pytype is not None:
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


# Private helpers

def _cast_item(v: Any, item_t: Any) -> Any:
    """Cast a single item to the expected type."""
    try:
        if _is_list_type(item_t):
            inner = getattr(item_t, "__args__", (Any,))[0] if getattr(item_t, "__args__", ()) else Any
            return _coerce_list(v, inner)
        if isinstance(item_t, type) and issubclass(item_t, BaseModel):
            return coerce_model_value(item_t, v)
        if item_t in (int, float, str, bool):
            return item_t(v)
        return v
    except Exception:
        return v


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


# --- Internal helpers for nested/collection types ---

def _is_list_type(typ: Any) -> bool:
    origin = getattr(typ, "__origin__", None)
    return origin in (list, List)


def _coerce_by_pytype(value: Any, pytype: Any) -> Any:
    if isinstance(pytype, type) and issubclass(pytype, BaseModel):
        return coerce_model_value(pytype, value)
    try:
        if _is_list_type(pytype):
            if not isinstance(value, list):
                return value
            item_t = getattr(pytype, "__args__", (Any,))[0] if getattr(pytype, "__args__", ()) else Any
            return [_coerce_element(v, item_t) for v in value]
        if pytype in (int, float, str, bool):
            return pytype(value)
    except Exception:
        return value
    return value


def _coerce_element(v: Any, item_t: Any) -> Any:
    if _is_list_type(item_t):
        inner = getattr(item_t, "__args__", (Any,))[0] if getattr(item_t, "__args__", ()) else Any
        return _coerce_list(v, inner)
    if isinstance(item_t, type) and issubclass(item_t, BaseModel):
        return coerce_model_value(item_t, v)
    if item_t in (int, float, str, bool):
        try:
            return item_t(v)
        except Exception:
            return v
    return v


def _coerce_list(value: Any, item_t: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [_coerce_element(v, item_t) for v in value]


def _is_compatible_by_pytype(value: Any, pytype: Any) -> bool:
    try:
        if isinstance(pytype, type) and issubclass(pytype, BaseModel):
            return isinstance(value, pytype)
        if _is_list_type(pytype):
            if not isinstance(value, list):
                return False
            item_t = getattr(pytype, "__args__", (Any,))[0] if getattr(pytype, "__args__", ()) else Any
            for v in value:
                if _is_list_type(item_t):
                    inner = getattr(item_t, "__args__", (Any,))[0] if getattr(item_t, "__args__", ()) else Any
                    if not _is_compatible_by_pytype(v, item_t):
                        return False
                elif isinstance(item_t, type) and issubclass(item_t, BaseModel):
                    if not isinstance(v, item_t):
                        return False
                elif item_t in (int, float, str, bool):
                    if not isinstance(v, item_t):
                        return False
            return True
        if pytype in (int, float, str, bool):
            return isinstance(value, pytype)
    except Exception:
        return True
    return True


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