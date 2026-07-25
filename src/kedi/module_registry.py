"""Utilities for resolving Kedi module imports."""

from __future__ import annotations

import importlib.resources
import re
from functools import lru_cache
from pathlib import Path

_INTERNAL_MODULES_PACKAGE = "kedi.modules"
_MODULE_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _module_file_parts(module_name: str) -> tuple[str, ...]:
    segments = tuple(module_name.split("/"))
    if not segments or any(not _MODULE_SEGMENT_RE.fullmatch(segment) for segment in segments):
        raise ValueError(f"Invalid Kedi module path: {module_name!r}")
    return (*segments[:-1], f"{segments[-1]}.kedi")


@lru_cache(maxsize=None)
def internal_module_path(module_name: str) -> Path | None:
    """Return the bundled module path for ``module_name`` if it exists."""
    try:
        resource = importlib.resources.files(_INTERNAL_MODULES_PACKAGE)
        for part in _module_file_parts(module_name):
            resource = resource.joinpath(part)
    except Exception:
        return None

    try:
        if not resource.is_file():
            return None
    except Exception:
        return None

    return Path(str(resource)).resolve()


def module_search_paths(module_name: str, *, source_path: Path | None) -> tuple[Path, ...]:
    """Return candidate module paths in resolution order."""
    relative_path = Path(*_module_file_parts(module_name))
    candidates: list[Path] = []
    if source_path is not None:
        candidates.append((source_path.parent / relative_path).resolve())

    internal_path = internal_module_path(module_name)
    if internal_path is not None and internal_path not in candidates:
        candidates.append(internal_path)

    return tuple(candidates)


def find_module_path(module_name: str, *, source_path: Path | None) -> Path | None:
    """Return the first existing module path for ``module_name``."""
    for candidate in module_search_paths(module_name, source_path=source_path):
        if candidate.exists():
            return candidate
    return None


def describe_module_search_paths(paths: tuple[Path, ...]) -> str:
    """Render candidate paths for error and diagnostic messages."""
    if not paths:
        return "registered internal modules"
    if len(paths) == 1:
        return str(paths[0])
    return " or ".join(str(path) for path in paths)


def resolve_module_path(module_name: str, *, source_path: Path | None) -> Path:
    """Resolve ``module_name`` or raise a user-facing import error."""
    resolved = find_module_path(module_name, source_path=source_path)
    if resolved is not None:
        return resolved

    paths = module_search_paths(module_name, source_path=source_path)
    if source_path is None:
        raise RuntimeError(
            f"Cannot resolve Kedi import {module_name!r} without a source path "
            "or a registered internal module"
        )

    raise FileNotFoundError(
        f"Kedi import {module_name!r} not found at {describe_module_search_paths(paths)}"
    )
