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


def installed_module_path(module_name: str) -> Path | None:
    """Return a third-party package module from the user's Kedi registry."""
    from .package_registry import package_registry_root

    _module_file_parts(module_name)
    return _installed_module_path_cached(module_name, package_registry_root())


@lru_cache(maxsize=512)
def _installed_module_path_cached(module_name: str, registry_root: Path) -> Path | None:
    """Resolve an installed module after validating its complete package boundary."""
    from .package_manifest import PACKAGE_MANIFEST_NAME, load_package_manifest
    from .package_registry import (
        CorruptInstalledPackageError,
        installed_package_root,
        resolve_installed_package_file,
        resolve_package_source,
    )

    segments = tuple(module_name.split("/"))
    package_name = segments[0]
    root = registry_root.resolve()
    package_root = installed_package_root(package_name, registry_root=root)
    if not package_root.exists():
        return None
    if package_root.is_symlink():
        raise CorruptInstalledPackageError(
            f"Installed Kedi package {package_name!r} must not be a symlink"
        )

    resolved_package_root = package_root.resolve()
    try:
        resolved_package_root.relative_to(root)
    except ValueError as exc:
        raise CorruptInstalledPackageError(
            f"Installed Kedi package {package_name!r} escapes the package registry"
        ) from exc

    manifest_path = resolved_package_root / PACKAGE_MANIFEST_NAME
    if not manifest_path.exists():
        raise CorruptInstalledPackageError(
            f"Installed Kedi package {package_name!r} is missing {PACKAGE_MANIFEST_NAME}"
        )
    if manifest_path.is_symlink():
        raise CorruptInstalledPackageError(
            f"Installed Kedi package {package_name!r} has a symlinked package manifest"
        )
    try:
        metadata = load_package_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise CorruptInstalledPackageError(
            f"Installed Kedi package {package_name!r} has an invalid package manifest: {exc}"
        ) from exc
    if metadata.name != package_name:
        raise CorruptInstalledPackageError(
            f"Installed Kedi package directory {package_name!r} does not match "
            f"manifest package {metadata.name!r}"
        )
    try:
        source = resolve_package_source(
            resolved_package_root,
            metadata.source,
            error_type=CorruptInstalledPackageError,
        )
    except (CorruptInstalledPackageError, FileNotFoundError, ValueError) as exc:
        raise CorruptInstalledPackageError(
            f"Installed Kedi package {package_name!r} has an unsafe source: {exc}"
        ) from exc

    relative_path = (
        Path("main.kedi") if len(segments) == 1 else Path(*segments[1:-1], f"{segments[-1]}.kedi")
    )
    return resolve_installed_package_file(source, relative_path)


def clear_installed_module_cache() -> None:
    """Invalidate cached registry lookups after a package installation transaction."""
    _installed_module_path_cached.cache_clear()


def module_search_paths(module_name: str, *, source_path: Path | None) -> tuple[Path, ...]:
    """Return candidate module paths in resolution order."""
    relative_path = Path(*_module_file_parts(module_name))
    candidates: list[Path] = []
    if source_path is not None:
        candidates.append((source_path.parent / relative_path).resolve())

    internal_path = internal_module_path(module_name)
    if internal_path is not None and internal_path not in candidates:
        candidates.append(internal_path)

    # Registry validation is intentionally deferred until the higher-priority
    # sibling and bundled-module candidates miss. A corrupt installed package
    # must not prevent an otherwise valid local or built-in import from
    # resolving according to Kedi's documented precedence order.
    if any(candidate.is_file() for candidate in candidates):
        return tuple(candidates)

    installed_path = installed_module_path(module_name)
    if installed_path is not None and installed_path not in candidates:
        candidates.append(installed_path)

    return tuple(candidates)


def find_module_path(module_name: str, *, source_path: Path | None) -> Path | None:
    """Return the first existing module path for ``module_name``."""
    for candidate in module_search_paths(module_name, source_path=source_path):
        if candidate.is_file():
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
            "or a registered internal module or installed package"
        )

    raise FileNotFoundError(
        f"Kedi import {module_name!r} not found at {describe_module_search_paths(paths)}"
    )
