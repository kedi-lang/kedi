"""Invocation-local declaration and deferred-work boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ProcessingScope:
    procedures: dict[str, Callable[..., Any]]
    types: list[dict[str, Any]]
    work_owner: object = field(default_factory=object)
