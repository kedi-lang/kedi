from __future__ import annotations

import ast as py_ast
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
from .ast_helpers import (
    assignment_name_span,
    extract_module_docstring,
    is_public_export_name,
)

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
    python_stub: Optional[str] = None


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
    prelude_values = _python_value_infos(
        prog.prelude,
        prog.prelude_loc,
        module_name=module_name,
    )
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
    local_values = {**prelude_values, **local_values}

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


def _python_value_infos(
    code: str | None,
    loc: Span | None,
    *,
    module_name: Optional[str],
) -> dict[str, ImportedValueInfo]:
    if code is None or loc is None:
        return {}
    try:
        module = py_ast.parse(code)
    except SyntaxError:
        return {}

    values: dict[str, ImportedValueInfo] = {}
    for stmt in module.body:
        for name, stub, name_line, name_col in _python_stmt_exports(stmt):
            values[name] = ImportedValueInfo(
                name=name,
                declaration_span=_python_name_span(loc, name, name_line, name_col),
                module_name=module_name,
                python_stub=stub,
            )
    return values


def _python_stmt_exports(stmt: py_ast.stmt) -> list[tuple[str, str, int, int]]:
    if isinstance(stmt, py_ast.FunctionDef):
        return [
            (
                stmt.name,
                _python_function_stub(stmt),
                stmt.lineno,
                stmt.col_offset + len("def "),
            )
        ]
    if isinstance(stmt, py_ast.AsyncFunctionDef):
        return [
            (
                stmt.name,
                _python_function_stub(stmt),
                stmt.lineno,
                stmt.col_offset + len("async def "),
            )
        ]
    if isinstance(stmt, py_ast.ClassDef):
        return [
            (
                stmt.name,
                f"class {stmt.name}: ...",
                stmt.lineno,
                stmt.col_offset + len("class "),
            )
        ]
    if isinstance(stmt, py_ast.AnnAssign) and isinstance(stmt.target, py_ast.Name):
        typ = _unparse_or_any(stmt.annotation)
        return [
            (
                stmt.target.id,
                f"{stmt.target.id}: {typ} = ...",
                stmt.target.lineno,
                stmt.target.col_offset,
            )
        ]
    if isinstance(stmt, py_ast.Assign):
        exports: list[tuple[str, str, int, int]] = []
        for target in stmt.targets:
            for name, line, col in _python_target_names(target):
                exports.append((name, f"{name}: object = ...", line, col))
        return exports
    return []


def _python_target_names(expr: py_ast.expr) -> list[tuple[str, int, int]]:
    if isinstance(expr, py_ast.Name):
        return [(expr.id, expr.lineno, expr.col_offset)]
    if isinstance(expr, (py_ast.Tuple, py_ast.List)):
        names: list[tuple[str, int, int]] = []
        for item in expr.elts:
            names.extend(_python_target_names(item))
        return names
    return []


def _python_function_stub(stmt: py_ast.FunctionDef | py_ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(stmt, py_ast.AsyncFunctionDef) else "def"
    args = py_ast.unparse(stmt.args)
    ret = f" -> {_unparse_or_any(stmt.returns)}" if stmt.returns is not None else ""
    return f"{prefix} {stmt.name}({args}){ret}: ..."


def _unparse_or_any(expr: py_ast.AST | None) -> str:
    if expr is None:
        return "Any"
    return py_ast.unparse(expr)


def _python_name_span(loc: Span, name: str, lineno: int, col_offset: int) -> Span:
    start_line = loc.start_line + lineno
    start_col = col_offset + 1
    return Span(
        start_byte=loc.start_byte,
        end_byte=loc.start_byte,
        start_line=start_line,
        start_col=start_col,
        end_line=start_line,
        end_col=start_col + len(name),
        source_path=loc.source_path,
    )
