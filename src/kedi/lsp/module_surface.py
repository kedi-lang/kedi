from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..lang.ast import (
    AssignBlockStmt,
    AssignStmt,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    Span,
    TypeDef,
    TypeExpr,
)
from ..lang.parser import parse_program
from ..lang.parser.parser import parse as _ts_parse
from ..module_registry import find_module_path, resolve_module_path
from ..source_loading import load_kedi_source_bundle

__all__ = [
    "ImportedModuleInfo",
    "ImportedSurface",
    "ImportedSurfaceFailure",
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
class ImportedSurfaceFailure:
    module_name: Optional[str]
    path: Path
    error: Exception


@dataclass
class ImportedSurface:
    modules: dict[str, ImportedModuleInfo] = field(default_factory=dict)
    procedures: dict[str, ProcedureDef] = field(default_factory=dict)
    types: dict[str, TypeDef] = field(default_factory=dict)
    values: dict[str, ImportedValueInfo] = field(default_factory=dict)
    failures: list[ImportedSurfaceFailure] = field(default_factory=list)

    def merge(self, other: "ImportedSurface") -> None:
        self.modules.update(other.modules)
        self.procedures.update(other.procedures)
        self.types.update(other.types)
        self.values.update(other.values)
        self.failures.extend(other.failures)


def program_path(prog: Program | ProgramWithValidation) -> Optional[Path]:
    loc = getattr(prog, "loc", None)
    if loc is not None and loc.source_path:
        return Path(loc.source_path).resolve()
    source_map = getattr(prog, "source_map", None)
    documents = getattr(source_map, "documents", None)
    if documents:
        return Path(documents[0].path).resolve()
    return None


def resolved_import_path(
    prog: Program | ProgramWithValidation, module_import_name: str
) -> Optional[Path]:
    try:
        return resolve_module_path(module_import_name, source_path=program_path(prog))
    except (FileNotFoundError, RuntimeError):
        return None


def assignment_name_span(stmt: AssignStmt | AssignBlockStmt) -> Optional[Span]:
    if stmt.loc is None:
        return None
    name_start_col = stmt.loc.start_col + 1
    return Span(
        start_byte=stmt.loc.start_byte + 1,
        end_byte=stmt.loc.start_byte + 1 + len(stmt.name),
        start_line=stmt.loc.start_line,
        start_col=name_start_col,
        end_line=stmt.loc.start_line,
        end_col=name_start_col + len(stmt.name),
        source_path=stmt.loc.source_path,
    )


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


def imported_surface_for_program(
    prog: Program | ProgramWithValidation,
) -> ImportedSurface:
    surface = ImportedSurface()
    root_path = program_path(prog)
    seen = {root_path} if root_path is not None else set()
    for module_import in getattr(prog, "imports", []):
        module_path = resolved_import_path(prog, module_import.module)
        if module_path is None:
            continue
        surface.merge(load_imported_surface(module_path, seen, module_import.module))
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
    except Exception as exc:
        return ImportedSurface(failures=[ImportedSurfaceFailure(module_name, path, exc)])

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
    for module_import in getattr(prog, "imports", []):
        resolved_nested_path = find_module_path(module_import.module, source_path=path)
        if resolved_nested_path is None:
            continue
        nested_surface.merge(
            load_imported_surface(resolved_nested_path, next_seen, module_import.module)
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

    export_names: list[str] = []
    for export in getattr(prog, "exports", []):
        if export.export_all:
            export_names.extend(
                name
                for name in (
                    list(combined_procedures) + list(combined_types) + list(combined_values)
                )
                if is_public_export_name(name)
            )
        export_names.extend(export.names)

    resolved = ImportedSurface(
        modules=dict(module_surface.modules),
        failures=list(nested_surface.failures),
    )
    for name in dict.fromkeys(export_names):
        if name in combined_procedures:
            resolved.procedures[name] = combined_procedures[name]
        elif name in combined_types:
            resolved.types[name] = combined_types[name]
        elif name in combined_values:
            resolved.values[name] = combined_values[name]
    return resolved


def is_public_export_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def extract_module_docstring(source: str) -> Optional[str]:
    tree = _ts_parse(source)
    named = [child for child in tree.root_node.children if child.is_named]
    if not named:
        return None

    first = named[0]
    if first.type == "block_comment":
        lines = source[first.start_byte : first.end_byte].splitlines()
        if len(lines) < 2:
            return None
        text = "\n".join(lines[1:-1]).strip()
        return text or None

    if first.type == "line_comment":
        body: list[str] = []
        for node in named:
            if node.type != "line_comment":
                break
            line = source[node.start_byte : node.end_byte]
            body.append(line[1:].lstrip())
        text = "\n".join(body).strip()
        return text or None

    return None
