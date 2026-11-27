"""Adapters that installed packages expose to the modules importing them (#232).

A package manifest's ``adapters:`` block names shortnames that ``> agent:`` or ``> adapter:``
select, in a module that imports the package and nowhere else: installing a package never
changes what other programs accept. An entry is a Python adapter class (``class:``) or a
JSON-RPC bridge to launch (``command:``).
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kedi.agent_adapter.meta import AgentAdapterKind
    from kedi.lang import Program


@dataclass(frozen=True)
class PackageAdapter:
    """One package adapter, resolved for the installed package that declares it."""

    package: str
    name: str
    kind: Literal["agent", "adapter"]
    class_path: str | None = None
    command: tuple[str, ...] | None = None
    """The bridge command, with arguments naming files of the package as absolute paths."""

    @property
    def adapter_kind(self) -> AgentAdapterKind:
        return "agent-harness" if self.kind == "agent" else "agent-framework"

    @property
    def directive(self) -> str:
        return "agent" if self.kind == "agent" else "adapter"


class PackageAdapterConflict(ValueError):
    """Two imported packages declare the same adapter shortname."""


def installed_package_name(module_name: str, module_path: Path) -> str | None:
    """The installed package an import resolved into, or None for a local or bundled module."""

    from kedi.module_registry import installed_module_path

    installed = installed_module_path(module_name)
    if installed is None or installed != module_path.resolve():
        return None
    return module_name.split("/")[0]


def package_adapters(package_name: str) -> tuple[PackageAdapter, ...]:
    """The adapters an installed package declares, resolved against its source tree."""

    from kedi.module_registry import installed_package
    from kedi.package_environment import environment_program
    from kedi.package_manifest import validate_package_adapter
    from kedi.package_registry import (
        installed_package_root,
        package_registry_root,
        resolve_installed_package_file,
    )

    installed = installed_package(package_name, package_registry_root())
    assert installed is not None, f"package {package_name!r} is not installed"
    metadata, source = installed
    resolved: list[PackageAdapter] = []
    for declaration in metadata.adapters.values():
        validate_package_adapter(declaration)
        command = None
        if declaration.command is not None:
            command = tuple(
                _package_file_argument(part, source, resolve_installed_package_file)
                for part in shlex.split(declaration.command)
            )
            if declaration.environment == "uv":
                # The program runs from the package's environment: `python` is its interpreter.
                package_root = installed_package_root(package_name)
                program = environment_program(package_root, command[0])
                if program is not None:
                    command = (program, *command[1:])
        resolved.append(
            PackageAdapter(
                package=package_name,
                name=declaration.name,
                kind="agent" if declaration.kind == "agent" else "adapter",
                class_path=declaration.class_path,
                command=command,
            )
        )
    return tuple(resolved)


def _package_file_argument(part: str, source, resolve) -> str:  # noqa: ANN001
    """``part`` as an absolute path when it names a regular file of the package's source."""

    candidate = Path(part)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        return part
    resolved = resolve(source, candidate)
    return str(resolved) if resolved is not None else part


class PackageAdapterTable:
    """The package adapters one module may select, from the packages it imports."""

    def __init__(self) -> None:
        self._adapters: dict[str, PackageAdapter] = {}

    def register_import(self, module_name: str, module_path: Path) -> None:
        """Adds the adapters of the installed package ``module_name`` resolved into."""

        package = installed_package_name(module_name, module_path)
        if package is None:
            return
        self.add(package_adapters(package))

    def add(self, adapters: Iterable[PackageAdapter]) -> None:
        for adapter in adapters:
            known = self._adapters.get(adapter.name)
            if known is not None and known.package != adapter.package:
                raise PackageAdapterConflict(
                    f"packages {known.package!r} and {adapter.package!r} both declare the "
                    f"adapter {adapter.name!r}; import only one of them in this module"
                )
            self._adapters[adapter.name] = adapter

    def get(self, name: str) -> PackageAdapter | None:
        return self._adapters.get(name)

    def names(self, kind: AgentAdapterKind) -> tuple[str, ...]:
        return tuple(
            name for name, adapter in self._adapters.items() if adapter.adapter_kind == kind
        )


def imported_package_adapters(
    module_names: Iterable[str], *, source_path: Path | None
) -> PackageAdapterTable:
    """The package adapters a module's imports expose, for static tools such as the LSP."""

    from kedi.module_registry import find_module_path

    table = PackageAdapterTable()
    for module_name in module_names:
        try:
            path = find_module_path(module_name, source_path=source_path)
        except ValueError:
            continue
        if path is not None:
            table.register_import(module_name, path)
    return table


def program_package_adapters(prog: Program | None) -> PackageAdapterTable:
    """The package adapters a parsed program's imports expose; empty when they cannot be read.

    Editors call this on every keystroke: an import that does not resolve, or a package whose
    manifest is broken, only leaves its adapters out, since the import reports its own error.
    """

    from kedi.lang.compiler.module_loader import _program_source_path

    if prog is None or not prog.imports:
        return PackageAdapterTable()
    try:
        return imported_package_adapters(
            (module_import.module for module_import in prog.imports),
            source_path=_program_source_path(prog),
        )
    except Exception:
        return PackageAdapterTable()
