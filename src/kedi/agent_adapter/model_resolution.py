"""Scoped model construction for embedding environments such as notebooks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import Model

_pydantic_resolver: ContextVar[Callable[[str], Model] | None] = ContextVar(
    "kedi_pydantic_model_resolver", default=None
)


@contextmanager
def pydantic_model_resolution(resolver: Callable[[str], Model]) -> Iterator[None]:
    token = _pydantic_resolver.set(resolver)
    try:
        yield
    finally:
        _pydantic_resolver.reset(token)


def resolve_pydantic_model(model: str) -> str | Model:
    resolver = _pydantic_resolver.get()
    return model if resolver is None else resolver(model)
