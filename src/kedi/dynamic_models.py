from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import create_model

from .resolved_types import ResolvedKediType

__all__ = ["build_dynamic_model"]


def build_dynamic_model(
    name: str,
    base: type[Any],
    fields: Mapping[str, tuple[ResolvedKediType, Any]],
    **create_model_kwargs: Any,
) -> type[Any]:
    return cast(Any, create_model)(name, __base__=base, **fields, **create_model_kwargs)
