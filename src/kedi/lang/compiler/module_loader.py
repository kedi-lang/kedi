"""Compilation-local loading, caching and initialization of imported modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kedi.core import KediRuntime
from kedi.lang.ast import Program
from kedi.telemetry import capture_enabled, runtime_span


@dataclass(frozen=True)
class _LoadedModule:
    """One executable imported module for one root-program compilation."""

    path: Path
    program: Program
    runtime: KediRuntime


class _ModuleLoader:
    """Compilation-local imported-module cache with deterministic cycle reporting."""

    def __init__(self) -> None:
        self._programs: dict[Path, Program] = {}
        self._loaded: dict[Path, _LoadedModule] = {}
        self._loading: list[Path] = []

    def program_for(
        self,
        module_path: Path,
        load_program: Callable[[Path], Program],
    ) -> tuple[Path, Program]:
        path = module_path.resolve()
        program = self._programs.get(path)
        if program is None:
            program = load_program(path)
            self._programs[path] = program
        return path, program

    def load(
        self,
        module_path: Path,
        program: Program,
        compile_module: Callable[[Program], KediRuntime],
    ) -> _LoadedModule:
        path = module_path.resolve()
        loaded = self._loaded.get(path)
        if loaded is not None:
            return loaded
        if path in self._loading:
            cycle_start = self._loading.index(path)
            cycle = " -> ".join(str(item) for item in (*self._loading[cycle_start:], path))
            raise RuntimeError(f"Cyclic Kedi import detected: {cycle}")

        self._loading.append(path)
        try:
            attributes: dict[str, bool | str] = {
                "kedi.module.name": path.stem,
                "kedi.compile.cache_hit": False,
            }
            if capture_enabled("source_path"):
                attributes["code.file.path"] = str(path)
            with runtime_span(
                f"compile module {path.stem}",
                "compile_module",
                attributes=attributes,
            ) as active:
                try:
                    runtime = compile_module(program)
                    # Module initialization follows Python import semantics: top-level
                    # code runs once, then every importer shares the loaded runtime.
                    runtime.run_main()
                except BaseException as exc:
                    active.set_attribute("error.type", type(exc).__name__)
                    raise
                loaded = _LoadedModule(path=path, program=program, runtime=runtime)
                self._loaded[path] = loaded
                return loaded
        finally:
            self._loading.pop()

    def loaded_for(self, module_path: Path) -> _LoadedModule:
        """Return a previously initialized module for a source-order rebinding."""
        path = module_path.resolve()
        try:
            return self._loaded[path]
        except KeyError as exc:  # pragma: no cover - defensive compiler invariant.
            raise RuntimeError(
                f"Kedi module {path} was not initialized during compilation"
            ) from exc


def _program_source_path(prog: Program) -> Path | None:
    loc = prog.loc
    if loc is not None and loc.source_path:
        return Path(loc.source_path).resolve()
    source_map = prog.source_map
    if source_map is not None and source_map.documents:
        return Path(source_map.documents[0].path).resolve()
    return None
