"""Shared typing aliases for resolved Kedi/Python type annotations."""

from __future__ import annotations

from types import GenericAlias, UnionType
from typing import Annotated, Any, TypeAlias, cast, get_args, get_origin

from pydantic import TypeAdapter

ResolvedKediType: TypeAlias = type | GenericAlias | UnionType


def annotated_field_metadata(annotation: Any) -> tuple[Any, str | None]:
    """Extract Kedi's description while retaining native validation metadata."""
    if get_origin(annotation) is not Annotated:
        return annotation, None
    base, *metadata = get_args(annotation)
    description = metadata[0] if metadata and isinstance(metadata[0], str) else None
    retained = tuple(item for item in metadata if not isinstance(item, str))
    return (Annotated[(base, *retained)] if retained else base), description


def is_resolved_kedi_type(value: object) -> bool:
    if isinstance(value, (type, GenericAlias, UnionType)):
        return True
    return hasattr(value, "__origin__") or str(type(value)).startswith("<class 'typing.")


def as_resolved_kedi_type(value: object) -> ResolvedKediType | None:
    if not is_resolved_kedi_type(value):
        return None
    return cast(ResolvedKediType, value)


def runtime_isinstance(value: object, expected: ResolvedKediType) -> bool:
    # Bare class -> isinstance. A parameterized generic (e.g. list[str]) can also
    # pass isinstance(.., type) on CPython, so route it to the strict check below.
    if isinstance(expected, type) and not isinstance(expected, GenericAlias):
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
    "annotated_field_metadata",
    "ResolvedKediType",
    "as_resolved_kedi_type",
    "is_resolved_kedi_type",
    "resolved_type_name",
    "runtime_isinstance",
]
