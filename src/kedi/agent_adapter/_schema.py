from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

from pydantic import Field

from ..resolved_types import ResolvedKediType


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


def output_field_metadata(type_: ResolvedKediType) -> tuple[Any, str | None]:
    if get_origin(type_) is not Annotated:
        return type_, None

    annotation, first_metadata, *_ = get_args(type_)
    description = first_metadata if isinstance(first_metadata, str) else None
    return annotation, description


def output_field_definition(type_: ResolvedKediType) -> tuple[Any, Any]:
    annotation, description = output_field_metadata(type_)
    if description is not None:
        return annotation, Field(description=description)
    return annotation, ...
