"""Type system for the Kedi compiler.

Phase-7 refactor: ``KediType`` is now driven by structured ``TypeExpr``
nodes produced by the tree-sitter CST→AST walker. The pre-refactor
parser stored every type annotation as a raw string and ``eval``'d it
at compile time across four different paths; with structured types we
dispatch on ``TypeRef`` / ``TypeApply`` / ``TypeUnion``, and ``eval``
survives only inside the explicit ``TypePython`` (backtick) escape
hatch.

Backward-compat: ``KediType`` still accepts a raw string for callers
that have not yet migrated to the structured representation; in that
mode it falls back to the legacy ``eval``-based resolution. Phase 8
removes that path once every compiler / codegen / optimizer call site
constructs ``KediType`` from a ``TypeExpr``.
"""

from __future__ import annotations

import copy
import json
import shlex
from typing import Any, Optional, Union, cast

from pydantic import BaseModel, Field, TypeAdapter, create_model

from ...core import KediRuntime
from ..ast import (
    TypeApply,
    TypeDef,
    TypeExpr,
    TypePython,
    TypeRef,
    TypeUnion,
)
from ..type_introspection import make_type_init
from .environment import build_type_env

# ---------------------------------------------------------------------
# TypeExpr resolution
# ---------------------------------------------------------------------


def _lookup_name(rt: KediRuntime, name: str, type_env: dict[str, Any]) -> Any:
    """Resolve a bare type name against the type environment.

    Falls back to Python's ``eval`` so builtins (``int``, ``list``, …)
    and qualified names (``typing.Optional``) work without us
    enumerating every supported source.
    """
    if name in type_env:
        return type_env[name]
    try:
        return rt.executor.evaluate_type_expression(rt, name, type_env)
    except Exception:
        return None


def _resolve_expr(rt: KediRuntime, expr: TypeExpr, type_env: dict[str, Any]) -> Optional[Any]:
    """Resolve a structured ``TypeExpr`` against a type environment."""
    if isinstance(expr, TypeRef):
        return _lookup_name(rt, expr.name, type_env)

    if isinstance(expr, TypeApply):
        base = _lookup_name(rt, expr.name, type_env)
        if base is None:
            return None
        args = tuple(_resolve_expr(rt, a, type_env) for a in expr.args)
        if any(a is None for a in args):
            return None
        try:
            if len(args) == 1:
                return base[args[0]]
            return base[args]
        except Exception:
            return None

    if isinstance(expr, TypeUnion):
        left = _resolve_expr(rt, expr.left, type_env) if expr.left is not None else None
        right = _resolve_expr(rt, expr.right, type_env) if expr.right is not None else None
        if left is None or right is None:
            return None
        try:
            return left | right
        except Exception:
            return None

    if isinstance(expr, TypePython):
        # Explicit `\`code\`` escape hatch — the user opted into
        # running arbitrary Python here. A failing eval (undefined
        # name, syntax error, or a value that isn't a real type)
        # raises a structured ``TypeError`` so the caller's
        # try/except in ``assign_value_from_native`` produces the
        # legacy "Native assignment value does not match annotated
        # type" diagnostic.
        try:
            result = rt.executor.evaluate_type_expression(rt, expr.code, type_env)
        except Exception as exc:
            raise TypeError(f"Invalid type expression {expr.code!r}: {exc}") from exc
        if not _is_valid_type(result):
            raise TypeError(f"Expression {expr.code!r} does not evaluate to a valid type")
        return result

    return None


def _is_valid_type(obj: Any) -> bool:
    """Best-effort check that ``obj`` is a usable type for validation."""
    import types as _types

    return (
        isinstance(obj, type)
        or isinstance(obj, _types.GenericAlias)
        or hasattr(obj, "__origin__")
        or str(type(obj)).startswith("<class 'typing.")
    )


# ---------------------------------------------------------------------
# KediType
# ---------------------------------------------------------------------


class KediType:
    """A type annotation, either structured (``TypeExpr``) or legacy string.

    The structured form is preferred — Phase-7 call sites construct
    ``KediType`` from the AST's ``type_expr`` field whenever available.
    The legacy string path remains so unmigrated callers
    (codegen, optimizer utilities, hand-built AST in tests) still
    work; it routes through ``eval`` like the pre-refactor parser.
    """

    def __init__(self, source: Union[TypeExpr, str, None]):
        if source is None:
            self._expr: Optional[TypeExpr] = None
            self._legacy_str: Optional[str] = None
        elif isinstance(source, TypeExpr):
            self._expr = source
            self._legacy_str = None
        else:
            # Treat as legacy raw annotation string.
            self._expr = None
            self._legacy_str = source if str(source).strip() else None

    @classmethod
    def from_node(
        cls,
        type_expr: Optional[TypeExpr],
        type_annotation: Optional[str],
    ) -> "KediType":
        """Build a ``KediType`` from a pair of AST fields.

        Prefers the structured ``type_expr``; falls back to
        ``type_annotation`` so consumers that haven't migrated still
        work end-to-end during the Phase-7→8 transition.
        """
        if type_expr is not None:
            return cls(type_expr)
        return cls(type_annotation)

    # ---- public API ----

    def is_none(self) -> bool:
        return self._expr is None and self._legacy_str is None

    def resolve(self, rt: KediRuntime, env: dict[str, Any] | None = None) -> Optional[Any]:
        if self._expr is not None:
            type_env = build_type_env(rt, env)
            return _resolve_expr(rt, self._expr, type_env)
        if self._legacy_str:
            return self._legacy_resolve(rt, env)
        return None

    def get_adapter(
        self, rt: KediRuntime, env: dict[str, Any] | None = None
    ) -> Optional[TypeAdapter]:
        if self.is_none():
            return None
        resolved = self.resolve(rt, env)
        if resolved is None:
            return None
        try:
            return TypeAdapter(resolved, config={"strict": True, "arbitrary_types_allowed": True})
        except Exception:
            return None

    def validate(
        self,
        rt: KediRuntime,
        env: dict[str, Any] | None,
        value: Any,
        *,
        allow_str_coercion: bool = False,
    ) -> Any:
        adapter = self.get_adapter(rt, env)
        if adapter is None:
            return value
        return adapter.validate_python(value)

    def __repr__(self) -> str:
        if self._expr is not None:
            return f"KediType({self._expr!r})"
        return f"KediType({self._legacy_str!r})"

    def __str__(self) -> str:
        if self._expr is not None:
            return _typeexpr_to_str(self._expr)
        return self._legacy_str or "Any"

    # ---- legacy fallback ----

    def _legacy_resolve(self, rt: KediRuntime, env: dict[str, Any] | None) -> Optional[Any]:
        """Resolve a raw annotation string via ``eval`` (pre-refactor path)."""
        annotation = (self._legacy_str or "").strip()
        if not annotation:
            return None
        # Backtick-wrapped legacy strings (``[x: `Optional[int]`]``).
        if (
            len(annotation) >= 2
            and annotation.startswith("`")
            and annotation.endswith("`")
            and annotation.count("`") == 2
        ):
            expr = annotation[1:-1]
            type_env = build_type_env(rt, env)
            try:
                return rt.executor.evaluate_type_expression(rt, expr, type_env)
            except Exception:
                return None
        type_env = build_type_env(rt, env)
        try:
            return rt.executor.evaluate_type_expression(rt, annotation, type_env)
        except Exception:
            return None


def _typeexpr_to_str(e: Optional[TypeExpr]) -> str:
    """Render a structured ``TypeExpr`` back to a human-readable form."""
    if e is None:
        return "None"
    if isinstance(e, TypeRef):
        return e.name
    if isinstance(e, TypeApply):
        return f"{e.name}[{', '.join(_typeexpr_to_str(a) for a in e.args)}]"
    if isinstance(e, TypeUnion):
        return f"{_typeexpr_to_str(e.left)} | {_typeexpr_to_str(e.right)}"
    if isinstance(e, TypePython):
        return f"`{e.code}`"
    return repr(e)


# ---------------------------------------------------------------------
# Helpers below this point are unchanged from the pre-refactor module.
# ---------------------------------------------------------------------


def coerce_model_value(model: type[BaseModel], value: Any) -> Any:
    """Coerce a value to a Pydantic model."""
    if isinstance(value, model):
        return value
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, dict):
        return model(**value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return model(**loaded)
            raise ValueError(f"Cannot coerce non-object JSON to {model.__name__}")
        parsed = _parse_model_repr(text, model)
        if parsed is not None:
            return model(**parsed)
    raise ValueError(f"Cannot coerce {type(value).__name__} to {model.__name__}")


def is_value_compatible(
    kedi_type: KediType, value: Any, rt: KediRuntime, env: dict[str, Any] | None = None
) -> bool:
    try:
        kedi_type.validate(rt, env, value)
        return True
    except Exception:
        return False


def register_type(
    rt: KediRuntime,
    typedef: TypeDef,
    type_env: dict[str, Any],
    local_env: dict[str, Any] | None = None,
) -> None:
    """Register a custom type definition as a Pydantic model."""
    annotations: dict[str, Any] = {}
    field_defs: dict[str, tuple[Any, Any]] = {}
    default_values: dict[str, Any] = {}
    default_env = build_type_env(rt, local_env or {})

    for field in typedef.fields:
        kedi_type = KediType.from_node(field.type_expr, field.type_annotation)
        if kedi_type.is_none():
            if field.default_expr is not None:
                raise TypeError(
                    f"Defaulted field {field.name!r} in type {typedef.name} "
                    "requires a type annotation"
                )
            # Untyped field defaults to `str` (legacy behaviour).
            kedi_type = KediType(TypeRef(name="str"))
        resolved = kedi_type.resolve(rt, env=type_env)
        if resolved is None:
            raise TypeError(f"Unknown type annotation {kedi_type!s} in type {typedef.name}")
        annotations[field.name] = resolved
        if field.default_expr is None:
            field_defs[field.name] = (resolved, ...)
        else:
            default_value = rt.executor.evaluate_type_expression(
                rt, field.default_expr, default_env
            )
            default_values[field.name] = default_value
            field_defs[field.name] = (
                resolved,
                _pydantic_default(default_value),
            )

    new_model = create_model(
        typedef.name,
        __base__=cast(Any, BaseModel),
        **cast(Any, field_defs),
    )
    init_namespace = build_type_env(rt, local_env or {})
    init_namespace[typedef.name] = new_model
    setattr(
        new_model,
        "__init__",
        make_type_init(
            typedef,
            new_model,
            default_values=default_values,
            namespace=init_namespace,
        ),
    )

    type_env[typedef.name] = new_model
    type_env[f"__{typedef.name}_fields__"] = annotations


def _pydantic_default(value: Any) -> Any:
    if _is_immutable_default(value):
        return Field(default=value)
    return Field(default_factory=_default_factory(copy.deepcopy(value)))


def _default_factory(value: Any):
    def factory() -> Any:
        return copy.deepcopy(value)

    return factory


def _is_immutable_default(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return True
    if isinstance(value, tuple):
        return all(_is_immutable_default(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_immutable_default(item) for item in value)
    return False


def assign_value_from_native(
    native: Any,
    kedi_type: KediType,
    rt: KediRuntime,
    env: dict[str, Any] | None,
    *,
    strict_native: bool,
) -> Any:
    if strict_native:
        if kedi_type.is_none():
            if isinstance(native, str):
                return native
            raise TypeError("Native non-string assignment requires explicit type annotation")
        if isinstance(native, str):
            try_typ = kedi_type.resolve(rt, env)
            if try_typ is not None and try_typ is not str:
                raise TypeError("Annotation type mismatch: got str value")
        try:
            return kedi_type.validate(rt, env, native)
        except Exception:
            raise TypeError("Native assignment value does not match annotated type")
    else:
        if not kedi_type.is_none():
            raise TypeError("Angle-inline pure python cannot have a non-str annotation")
        return str(native)


def detect_rhs_kind(segments: list) -> tuple[bool, bool, bool]:
    """Detect the kind of RHS expression."""
    from ..ast import (
        CallSegment,
        PythonExprNativeSegment,
        PythonExprSegment,
        TextSegment,
    )

    non_ws = [s for s in segments if not (isinstance(s, TextSegment) and s.text.strip() == "")]

    is_backtick_native = len(non_ws) == 1 and isinstance(non_ws[0], PythonExprNativeSegment)
    is_angle_pure = len(non_ws) == 1 and isinstance(non_ws[0], PythonExprSegment)
    is_call_only = len(non_ws) == 1 and isinstance(non_ws[0], CallSegment)

    return is_backtick_native, is_angle_pure, is_call_only


def _convert_basic_literal(raw: str, typ: Any) -> Any:
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


def push_type_env(rt: KediRuntime) -> dict[str, Any]:
    stack = getattr(rt, "_type_env_stack", None)
    if stack is None:
        stack = []
        setattr(rt, "_type_env_stack", stack)
    new_env: dict[str, Any] = {}
    stack.append(new_env)
    return new_env


def pop_type_env(rt: KediRuntime) -> None:
    stack = getattr(rt, "_type_env_stack", None)
    if isinstance(stack, list) and stack:
        stack.pop()
