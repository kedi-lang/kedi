"""Kedi packages that depend on Kedi packages: ``kedi_dependencies``.

A package's dependents are never stored: they are the installed packages whose manifests list
it, so no record can go stale. An installation resolves its whole graph before it changes
anything: registry versions, specifiers, cycles, and the installed packages that would stop
being satisfied.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from packaging.requirements import Requirement
from packaging.version import Version

from .lang import PackageMetadata
from .package_manifest import PACKAGE_MANIFEST_NAME, load_package_manifest, parse_kedi_requirement


class _HasMetadata(Protocol):
    @property
    def metadata(self) -> PackageMetadata: ...


Fetched = TypeVar("Fetched", bound=_HasMetadata)


def satisfies(version: str | None, requirement: Requirement) -> bool:
    """Whether a package version meets ``requirement``. Installed and registry versions passed
    manifest validation, so they parse; a package without a version meets only no specifier."""
    if not requirement.specifier:
        return True
    return version is not None and requirement.specifier.contains(Version(version))


def installed_metadata(registry_root: Path, name: str) -> PackageMetadata | None:
    """An installed package's manifest, or None when it is not installed."""
    try:
        return load_package_manifest(registry_root / name / PACKAGE_MANIFEST_NAME)
    except (OSError, ValueError):
        return None


def installed_dependents(registry_root: Path, name: str) -> dict[str, Requirement]:
    """The installed packages whose manifests list ``name``, with what each needs of it."""
    dependents: dict[str, Requirement] = {}
    for root in sorted(registry_root.iterdir()) if registry_root.is_dir() else []:
        if root.name == name or root.name.startswith(".") or not root.is_dir():
            continue
        metadata = installed_metadata(registry_root, root.name)
        for line in metadata.kedi_dependencies if metadata is not None else []:
            requirement = parse_kedi_requirement(line)
            if requirement.name == name:
                dependents[root.name] = requirement
    return dependents


def describe(dependent: str, requirement: Requirement) -> str:
    return f"{dependent} needs {requirement}"


def resolve_dependencies(
    root: PackageMetadata, *, registry_root: Path, fetch: Callable[[str], Fetched]
) -> list[Fetched]:
    """The packages to install before ``root``, dependencies first.

    An installed package that satisfies every requirement on it is kept. Any other is fetched
    from the registry, whose active version must satisfy every requirement on it: those of this
    installation and those of the installed packages that stay. Raises ``ValueError``, having
    changed nothing, on a cycle or a conflict."""
    planned: dict[str, Fetched] = {}
    order: list[Fetched] = []
    kept: set[str] = set()
    wanted: dict[str, list[tuple[str, Requirement]]] = {}
    replaced = {root.name}

    def require(name: str, version: str | None) -> None:
        # A kept installed package is both a dependent this installation visited and an
        # installed dependent: it is named once.
        needs = dict(wanted[name])
        for dependent, requirement in installed_dependents(registry_root, name).items():
            if dependent not in replaced:
                needs.setdefault(dependent, requirement)
        if not all(satisfies(version, requirement) for requirement in needs.values()):
            raise ValueError(
                f"No version of Kedi package {name!r} satisfies every package that needs it: "
                + "; ".join(describe(d, r) for d, r in needs.items())
                + f" (the registry's active version is {version})"
            )

    def visit(metadata: PackageMetadata, chain: list[str]) -> None:
        for line in metadata.kedi_dependencies:
            requirement = parse_kedi_requirement(line)
            name = requirement.name
            if name in chain:
                raise ValueError(
                    "Kedi packages depend on each other in a cycle: "
                    + " → ".join([*chain[chain.index(name) :], name])
                )
            wanted.setdefault(name, []).append((metadata.name, requirement))
            if name in planned:
                require(name, planned[name].metadata.version)
                continue
            installed = installed_metadata(registry_root, name)
            if installed is not None and all(
                satisfies(installed.version, r) for _, r in wanted[name]
            ):
                if name not in kept:
                    kept.add(name)
                    visit(installed, [*chain, name])
                continue
            fetched = fetch(name)
            replaced.add(name)
            kept.discard(name)
            require(name, fetched.metadata.version)
            planned[name] = fetched
            visit(fetched.metadata, [*chain, name])
            order.append(fetched)

    visit(root, [root.name])
    unmet = [
        describe(dependent, requirement)
        for dependent, requirement in installed_dependents(registry_root, root.name).items()
        if dependent not in replaced and not satisfies(root.version, requirement)
    ]
    if unmet:
        raise ValueError(
            f"Installing {root.name} {root.version} would break installed packages: "
            + "; ".join(unmet)
        )
    return order
