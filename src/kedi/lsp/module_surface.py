from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..lang.ast import (
    AssignBlockStmt,
    AssignStmt,
    ProcedureDef,
    ProfileDefStmt,
    Program,
    ProgramWithValidation,
    Span,
    TypeDef,
    TypeExpr,
)
from ..lang.parser import parse_program
from ..module_registry import find_module_path, resolve_module_path
from ..source_loading import load_kedi_source_bundle
from .ast_helpers import assignment_name_span, extract_module_docstring, is_public_export_name

__all__ = [
    "ImportedModuleInfo",
    "ImportedSurface",
    "ImportedValueInfo",
    "assignment_name_span",
    "extract_module_docstring",
    "imported_surface_for_program",
    "is_public_export_name",
    "load_imported_surface",
    "program_path",
    "resolved_import_path",
    "top_level_value_infos",
]


@dataclass
class ImportedModuleInfo:
    name: str
    path: Path
    declaration_span: Span
    docstring: Optional[str] = None


@dataclass
class ImportedValueInfo:
    name: str
    declaration_span: Span
    type_expr: Optional[TypeExpr] = None
    module_name: Optional[str] = None


@dataclass
class ImportedSurface:
    modules: dict[str, ImportedModuleInfo] = field(default_factory=dict)
    procedures: dict[str, ProcedureDef] = field(default_factory=dict)
    types: dict[str, TypeDef] = field(default_factory=dict)
    values: dict[str, ImportedValueInfo] = field(default_factory=dict)
    profiles: dict[str, ProfileDefStmt] = field(default_factory=dict)

    def merge(self, other: "ImportedSurface") -> None:
        self.modules.update(other.modules)
        self.procedures.update(other.procedures)
        self.types.update(other.types)
        self.values.update(other.values)
        self.profiles.update(other.profiles)


def program_path(prog: Program | ProgramWithValidation) -> Optional[Path]:
    loc = prog.loc
    if loc is not None and loc.source_path:
        return Path(loc.source_path).resolve()
    source_map = prog.source_map
    if source_map is not None and source_map.documents:
        return Path(source_map.documents[0].path).resolve()
    return None


def resolved_import_path(
    prog: Program | ProgramWithValidation, module_import_name: str
) -> Optional[Path]:
    try:
        return resolve_module_path(module_import_name, source_path=program_path(prog))
    except (FileNotFoundError, RuntimeError):
        return None


def _lookup_import_path(module_name: str, source_path: Path | None) -> Path | None:
    try:
        return find_module_path(module_name, source_path=source_path)
    except (FileNotFoundError, RuntimeError):
        return None


def top_level_value_infos(
    prog: Program | ProgramWithValidation,
) -> dict[str, ImportedValueInfo]:
    return {
        stmt.name: ImportedValueInfo(
            name=stmt.name,
            declaration_span=span,
            type_expr=stmt.type_expr,
        )
        for stmt in prog.toplevel_body
        if isinstance(stmt, (AssignStmt, AssignBlockStmt))
        for span in [assignment_name_span(stmt)]
        if span is not None
    }


def _merge_imports(
    surface: ImportedSurface,
    imports: list,
    *,
    source_path: Path | None,
    seen: set[Path],
) -> None:
    for module_import in imports:
        module_path = _lookup_import_path(module_import.module, source_path)
        if module_path is None:
            continue
        surface.merge(load_imported_surface(module_path, seen, module_import.module))


def imported_surface_for_program(
    prog: Program | ProgramWithValidation,
) -> ImportedSurface:
    surface = ImportedSurface()
    root_path = program_path(prog)
    seen = {root_path} if root_path is not None else set()
    _merge_imports(surface, list(prog.imports), source_path=root_path, seen=seen)
    return surface


def load_imported_surface(
    path: Path,
    seen: set[Path],
    module_name: Optional[str] = None,
) -> ImportedSurface:
    if path in seen or not path.exists():
        return ImportedSurface()

    next_seen = set(seen)
    next_seen.add(path)
    try:
        loaded = load_kedi_source_bundle(path, include_val=False, include_cache=True)
        prog = parse_program(
            loaded.text,
            source_path=str(path),
            source_map=loaded.source_map,
        )
    except Exception:
        return ImportedSurface()

    module_surface = ImportedSurface()
    if module_name is not None:
        module_surface.modules[module_name] = ImportedModuleInfo(
            name=module_name,
            path=path,
            declaration_span=Span(
                start_byte=0,
                end_byte=0,
                start_line=1,
                start_col=1,
                end_line=1,
                end_col=1,
                source_path=str(path),
            ),
            docstring=extract_module_docstring(loaded.text),
        )

    nested_surface = ImportedSurface()
    _merge_imports(
        nested_surface,
        list(prog.imports),
        source_path=path,
        seen=next_seen,
    )

    local_procedures = {proc.name: proc for proc in prog.procedures}
    local_types = {td.name: td for td in prog.type_defs}
    local_values = {
        stmt.name: ImportedValueInfo(
            name=stmt.name,
            declaration_span=span,
            type_expr=stmt.type_expr,
            module_name=module_name,
        )
        for stmt in prog.toplevel_body
        if isinstance(stmt, (AssignStmt, AssignBlockStmt))
        for span in [assignment_name_span(stmt)]
        if span is not None
    }

    combined_procedures = dict(nested_surface.procedures)
    combined_procedures.update(local_procedures)
    combined_types = dict(nested_surface.types)
    combined_types.update(local_types)
    combined_values = dict(nested_surface.values)
    combined_values.update(local_values)
    combined_profiles = dict(nested_surface.profiles)
    combined_profiles.update(prog.profile_defs)

    export_names: list[str] = []
    for export in prog.exports:
        if export.export_all:
            export_names.extend(
                name
                for name in (
                    list(combined_procedures)
                    + list(combined_types)
                    + list(combined_values)
                    + list(combined_profiles)
                )
                if is_public_export_name(name)
            )
        export_names.extend(export.names)

    resolved = ImportedSurface(modules=dict(module_surface.modules))
    for name in dict.fromkeys(export_names):
        if name in combined_procedures:
            resolved.procedures[name] = combined_procedures[name]
        elif name in combined_types:
            resolved.types[name] = combined_types[name]
        elif name in combined_values:
            resolved.values[name] = combined_values[name]
        elif name in combined_profiles:
            resolved.profiles[name] = combined_profiles[name]
    return resolved
