"""Shared typing aliases for resolved Kedi/Python type annotations."""

from __future__ import annotations

from types import GenericAlias, UnionType
from typing import TypeAlias, cast

from pydantic import TypeAdapter

ResolvedKediType: TypeAlias = type | GenericAlias | UnionType


def is_resolved_kedi_type(value: object) -> bool:
    if isinstance(value, (type, GenericAlias, UnionType)):
        return True
    return hasattr(value, "__origin__") or str(type(value)).startswith("<class 'typing.")


def as_resolved_kedi_type(value: object) -> ResolvedKediType | None:
    if not is_resolved_kedi_type(value):
        return None
    return cast(ResolvedKediType, value)


def runtime_isinstance(value: object, expected: ResolvedKediType) -> bool:
    # Bare class -> isinstance. A parameterized generic (GenericAlias) can pass
    # isinstance(.., type) on CPython, so route it below via __origin__.
    if isinstance(expected, type) and getattr(expected, "__origin__", None) is None:
        return isinstance(value, expected)
    # Validate generic args strictly; fail closed on mistyped/uncoercible values.
    try:
        TypeAdapter(expected).validate_python(value, strict=True)
        return True
    except Exception:
        return False


def resolved_type_name(value: ResolvedKediType) -> str:
    if isinstance(value, type):
        return value.__name__
    name = getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return str(value)


__all__ = [
    "ResolvedKediType",
    "as_resolved_kedi_type",
    "is_resolved_kedi_type",
    "resolved_type_name",
    "runtime_isinstance",
]
