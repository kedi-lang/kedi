from __future__ import annotations

from pathlib import Path
from typing import Any


def program_name(source_path: str | None, *, fallback: str = "<source>") -> str:
    """Return a trace-safe display name without exposing parent directories."""

    if not source_path:
        return fallback
    if source_path.startswith("<") and source_path.endswith(">"):
        return source_path
    return Path(source_path).name


def qualified_type_name(value: Any) -> str:
    """Return a stable result type name without serializing the result itself."""

    value_type = type(value)
    if value_type.__module__ == "builtins":
        return value_type.__qualname__
    return f"{value_type.__module__}.{value_type.__qualname__}"
