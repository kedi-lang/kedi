"""Shared typing aliases for resolved Kedi/Python type annotations."""

from __future__ import annotations

from functools import reduce
from operator import or_
from types import GenericAlias, UnionType
from typing import Annotated, Any, TypeAlias, Union, cast, get_args, get_origin

from pydantic import Field, TypeAdapter
from pydantic.fields import FieldInfo

ResolvedKediType: TypeAlias = type | GenericAlias | UnionType


def annotated_field_metadata(annotation: Any) -> tuple[Any, str | None]:
    """Extract Kedi's description while retaining native validation metadata."""
    if get_origin(annotation) is not Annotated:
        return _nested_descriptions(annotation), None
    base, *metadata = get_args(annotation)
    description = _description(metadata)
    retained = tuple(item for item in metadata if not isinstance(item, str))
    base = _nested_descriptions(base)
    return (Annotated[(base, *retained)] if retained else base), description


def _nested_descriptions(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        base, *metadata = get_args(annotation)
        base = _nested_descriptions(base)
        description = _description(metadata)
        retained = tuple(item for item in metadata if not isinstance(item, str))
        if description is not None:
            retained = (*retained, Field(description=description))
        return Annotated[(base, *retained)] if retained else base

    args = get_args(annotation)
    if not args:
        return annotation
    normalized = tuple(_nested_descriptions(arg) for arg in args)
    if normalized == args:
        return annotation
    if origin in (Union, UnionType):
        return reduce(or_, normalized) if origin is UnionType else Union[normalized]
    return origin[normalized[0] if len(normalized) == 1 else normalized]


def _description(metadata: list[Any]) -> str | None:
    # Python flattens Annotated aliases, placing their schema metadata before the description.
    for index, item in enumerate(metadata):
        if isinstance(item, str):
            if all(
                isinstance(prefix, FieldInfo)
                or hasattr(prefix, "__get_pydantic_json_schema__")
                or hasattr(prefix, "__get_pydantic_core_schema__")
                for prefix in metadata[:index]
            ):
                return item
            break
    return None


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
        adapter: TypeAdapter[Any] = TypeAdapter(cast(Any, expected))
        adapter.validate_python(value, strict=True)
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
