from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, TypeVar, cast

from pydantic import BaseModel, create_model

from .resolved_types import ResolvedKediType

__all__ = ["build_dynamic_model"]

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def build_dynamic_model(
    name: str,
    base: type[_ModelT],
    fields: Mapping[str, tuple[ResolvedKediType, Any]],
    **create_model_kwargs: Any,
) -> type[_ModelT]:
    field_definitions: dict[str, tuple[Any, Any]] = dict(fields)
    factory: Callable[..., type[_ModelT]] = cast(Callable[..., type[_ModelT]], create_model)
    return factory(name, __base__=base, **field_definitions, **create_model_kwargs)
