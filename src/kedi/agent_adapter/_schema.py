from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

from pydantic import Field

from ..resolved_types import ResolvedKediType


def make_repr(cls: type) -> str:
    _type_repr = repr(cls)
    return _type_repr if not _type_repr.startswith("<class") else cls.__name__


def output_field_definition(type_: ResolvedKediType) -> tuple[Any, Any]:
    if get_origin(type_) is Annotated:
        args = get_args(type_)
        if len(args) == 0:
            return type_, ...
        description = args[1] if len(args) > 1 and isinstance(args[1], str) else None
        if description is not None:
            return args[0], Field(description=description)
    return type_, ...
