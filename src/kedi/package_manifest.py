"""Loading and validation helpers for ``package.kedi`` manifests."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from .lang import PackageMetadata, parse_program
from .source_loading import load_kedi_source_bundle

PACKAGE_MANIFEST_NAME = "package.kedi"
_PACKAGE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PYTHON_REQUIREMENT_RE = re.compile(r"python@(\d+)\.(\d+)(?:-(\d+)\.(\d+))?$")


def validate_package_name(package_name: str) -> str:
    """Return one canonical Kedi package name or raise a clear input error."""
    if not _PACKAGE_NAME_RE.fullmatch(package_name):
        raise ValueError(
            "Kedi package names must use lowercase letters, digits, and underscores, "
            "and must start with a letter"
        )
    return package_name


def parse_python_requirement(requirement: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse Kedi's closed Python version range syntax."""
    match = _PYTHON_REQUIREMENT_RE.fullmatch(requirement)
    if match is None:
        raise ValueError(
            "package python must be python@MAJOR.MINOR or python@MAJOR.MINOR-MAJOR.MINOR"
        )

    lower = (int(match.group(1)), int(match.group(2)))
    upper = (int(match.group(3)), int(match.group(4))) if match.group(3) is not None else lower
    if upper < lower:
        raise ValueError("package Python version range must end at or after its start")
    return lower, upper


def validate_package_metadata(
    metadata: PackageMetadata,
    *,
    enforce_python_compatibility: bool = False,
    python_version: tuple[int, int] | None = None,
) -> None:
    """Validate metadata that tree-sitter syntax alone cannot prove.

    Manifests record metadata rather than installing Python dependencies.  We do
    nevertheless parse dependency declarations eagerly so invalid package data
    never reaches the local registry.
    """
    validate_package_name(metadata.name)

    if metadata.version is not None:
        try:
            Version(metadata.version)
        except InvalidVersion as exc:
            raise ValueError(f"invalid package version {metadata.version!r}") from exc

    for dependency in metadata.python_dependencies:
        try:
            Requirement(dependency)
        except InvalidRequirement as exc:
            raise ValueError(f"invalid Python dependency {dependency!r}") from exc

    if metadata.python_requirement is None:
        return

    lower, upper = parse_python_requirement(metadata.python_requirement)
    if not enforce_python_compatibility:
        return

    current = python_version or sys.version_info[:2]
    if not lower <= current <= upper:
        raise ValueError(
            "Kedi package requires Python "
            f"{metadata.python_requirement.removeprefix('python@')}, but the installer runs "
            f"on Python {current[0]}.{current[1]}"
        )


def load_package_manifest(path: Path) -> PackageMetadata:
    """Load a package manifest and return its metadata.

    A manifest is declarative only: it must contain exactly one ``> package:``
    directive and no executable Kedi statements.
    """
    manifest_path = path.resolve()
    if manifest_path.name != PACKAGE_MANIFEST_NAME:
        raise ValueError(f"Kedi package manifest must be named {PACKAGE_MANIFEST_NAME!r}")
    bundle = load_kedi_source_bundle(manifest_path, include_val=False, include_cache=False)
    program = parse_program(
        bundle.text,
        source_path=str(manifest_path),
        source_map=bundle.source_map,
    )
    if program.package is None:
        raise ValueError(f"{manifest_path} does not define a package")
    if (
        program.imports
        or program.exports
        or program.procedures
        or program.type_defs
        or program.ordered_toplevel
    ):
        raise ValueError(f"{manifest_path} must contain package metadata only")
    validate_package_metadata(program.package)
    return program.package
