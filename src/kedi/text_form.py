"""The text form of a value: what ``<x>`` renders, and what ``<<x>>`` attaches for plain data.

A value has a text form when its text comes from its data alone, in one canonical form, and no
user-defined ``__str__`` or ``__repr__`` runs. One walk validates the value; ``to_json`` renders
it, without a fallback, so an unknown type never reaches ``str()``. Containers are compact JSON;
a top-level scalar is its JSON text, unquoted when it is a string.
"""

from __future__ import annotations

import json
import math
from dataclasses import is_dataclass
from datetime import date, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from pydantic_core import to_json

DEPTH_LIMIT = 64
_CONTAINERS = (list, tuple, dict)


class NoTextForm(TypeError):
    """A value, or a part of it at ``path``, has no text form."""

    def __init__(self, path: str, reason: str, hint: str | None = None) -> None:
        self.path = path
        self.reason = reason
        self.hint = hint
        super().__init__(f"{path or 'the value'} {reason}")

    def describe(self, subject: str) -> str:
        """``subject``'s offending part and why, as a sentence: ``order["items"][2] is ...``."""

        where = f"{subject}{self.path}" if self.path else subject
        sentence = f"{where} {self.reason}."
        if self.hint is not None:
            sentence += f" {self.hint[0].upper()}{self.hint[1:]}."
        return sentence


def text_form(value: Any, *, models: bool = False) -> tuple[str, bool]:
    """Return ``value``'s text and whether it is structured (a JSON container or model).

    ``models`` also accepts Pydantic models and dataclasses anywhere in the value, serialized by
    their own schema; ``<<x>>`` passes them, ``<x>`` does not.
    """

    _check(value, "", 0, set(), models)
    while isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        return str.__str__(value), False
    text = to_json(value).decode("utf-8")
    if text.startswith('"'):
        return json.loads(text), False
    return text, isinstance(value, _CONTAINERS) or _is_model(value)


def _check(value: Any, path: str, depth: int, active: set[int], models: bool) -> None:
    if depth > DEPTH_LIMIT:
        raise NoTextForm(path, f"nests deeper than {DEPTH_LIMIT} levels")
    if value is None:
        if depth:
            return
        raise NoTextForm(path, "is None")
    if isinstance(value, Enum):
        _check(value.value, path, depth, active, models)
        return
    if isinstance(value, (float, Decimal)):
        finite = value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)
        if not finite:
            raise NoTextForm(path, "is not a finite number")
        return
    if isinstance(value, (str, int, date, time, UUID)):
        return
    kind = type(value)
    if kind in _CONTAINERS:
        if id(value) in active:
            raise NoTextForm(path, "contains itself")
        active.add(id(value))
        if kind is dict:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise NoTextForm(
                        path, f"has a key of type {type(key).__name__}; dict keys must be str"
                    )
                key_path = f"{path}[{json.dumps(str.__str__(key), ensure_ascii=False)}]"
                _check(item, key_path, depth + 1, active, models)
        else:
            for index, item in enumerate(value):
                _check(item, f"{path}[{index}]", depth + 1, active, models)
        active.discard(id(value))
        return
    if models and _is_model(value):
        return
    raise NoTextForm(path, *_problem(value))


def _is_model(value: Any) -> bool:
    return isinstance(value, BaseModel) or (is_dataclass(value) and not isinstance(value, type))


def _problem(value: Any) -> tuple[str, str | None]:
    """Why ``value`` has no text form, and how to render it when there is an obvious way."""

    from kedi.media import as_media

    media = as_media(value)
    if media is not None:
        return f"is {media.label}", None
    name = type(value).__name__
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "is bytes", None
    if isinstance(value, (set, frozenset)):
        return f"is of type {name}, whose order is not stable", (
            "sort it first, for example <`sorted(...)`>"
        )
    if isinstance(value, PurePath):
        return "is a path", "render it with <`str(...)`>"
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return f"is a NamedTuple ({name}), which would lose its field names", (
            "convert it with `._asdict()`"
        )
    for base in _CONTAINERS:
        if isinstance(value, base):
            return (
                f"is of type {name}, a {base.__name__} subclass",
                f"convert it with `{base.__name__}(...)`",
            )
    if isinstance(value, (timedelta, complex)):
        return f"is of type {name}, which has no single text form", None
    return f"is of type {name}", None


__all__ = ["DEPTH_LIMIT", "NoTextForm", "text_form"]
