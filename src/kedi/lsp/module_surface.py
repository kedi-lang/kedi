from __future__ import annotations

import ast as py_ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypeVar

from ..lang.ast import (
    ModuleImport,
    ProcedureDef,
    ProfileDefStmt,
    Program,
    ProgramWithValidation,
    Span,
    TypeDef,
    TypeExpr,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from ..lang.parser import parse_program
from ..module_registry import find_module_path, resolve_module_path
from ..package_registry import CorruptInstalledPackageError
from ..source_loading import load_kedi_source_bundle
from .ast_helpers import (
    binding_name_span,
    extract_module_docstring,
    is_public_export_name,
)

_SurfaceValue = TypeVar("_SurfaceValue")

__all__ = [
    "ImportedModuleInfo",
    "ImportedSurface",
    "ImportedValueInfo",
    "binding_name_span",
    "extract_module_docstring",
    "import_resolution_error",
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
        self._merge_collection(self.procedures, other.procedures)
        self._merge_collection(self.types, other.types)
        self._merge_collection(self.values, other.values)
        self._merge_collection(self.profiles, other.profiles)

    def select(self, names: list[str]) -> "ImportedSurface":
        if not names:
            return self
        selected = set(names)
        return ImportedSurface(
            modules=dict(self.modules),
            procedures={name: value for name, value in self.procedures.items() if name in selected},
            types={name: value for name, value in self.types.items() if name in selected},
            values={name: value for name, value in self.values.items() if name in selected},
            profiles={name: value for name, value in self.profiles.items() if name in selected},
        )

    def exported_names(self) -> set[str]:
        return set(self.procedures) | set(self.types) | set(self.values) | set(self.profiles)

    def _remove_binding(self, name: str) -> None:
        self.procedures.pop(name, None)
        self.types.pop(name, None)
        self.values.pop(name, None)
        self.profiles.pop(name, None)

    def _merge_collection(
        self,
        target: dict[str, _SurfaceValue],
        incoming: dict[str, _SurfaceValue],
    ) -> None:
        for name, value in incoming.items():
            self._remove_binding(name)
            target[name] = value


@dataclass
class _SurfaceBindings:
    """Track final top-level bindings and whether an import last wrote each one."""

    surface: ImportedSurface = field(default_factory=ImportedSurface)
    imported_names: set[str] = field(default_factory=set)

    def publish_import(self, imported: ImportedSurface) -> None:
        self.surface.merge(imported)
        self.imported_names.update(imported.exported_names())

    def publish_procedure(self, procedure: ProcedureDef) -> None:
        self._publish(procedure.name, self.surface.procedures, procedure)

    def publish_type(self, typedef: TypeDef) -> None:
        self._publish(typedef.name, self.surface.types, typedef)

    def publish_profile(self, profile: ProfileDefStmt) -> None:
        self._publish(profile.name, self.surface.profiles, profile)

    def publish_value(self, value: ImportedValueInfo) -> None:
        self._publish(value.name, self.surface.values, value)

    def imported_surface(self) -> ImportedSurface:
        return _retain_bindings(self.surface, self.imported_names)

    def _publish(self, name: str, target: dict[str, _SurfaceValue], value: _SurfaceValue) -> None:
        self.surface._remove_binding(name)
        target[name] = value
        self.imported_names.discard(name)


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


def import_resolution_error(
    prog: Program | ProgramWithValidation,
    module_import_name: str,
) -> str | None:
    """Return an installed-package integrity error without breaking LSP requests."""
    try:
        resolve_module_path(module_import_name, source_path=program_path(prog))
    except CorruptInstalledPackageError as exc:
        return str(exc)
    except (FileNotFoundError, RuntimeError):
        return None
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
        if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt))
        for span in [binding_name_span(stmt)]
        if span is not None
    }


def _retain_bindings(surface: ImportedSurface, names: set[str]) -> ImportedSurface:
    """Return selected bindings while retaining module navigation metadata."""
    return ImportedSurface(
        modules=dict(surface.modules),
        procedures={name: value for name, value in surface.procedures.items() if name in names},
        types={name: value for name, value in surface.types.items() if name in names},
        values={name: value for name, value in surface.values.items() if name in names},
        profiles={name: value for name, value in surface.profiles.items() if name in names},
    )


def _ordered_toplevel_items(prog: Program | ProgramWithValidation) -> list[object]:
    if prog.ordered_toplevel:
        return list(prog.ordered_toplevel)
    return [
        *prog.imports,
        *prog.type_defs,
        *prog.profile_defs.values(),
        *prog.procedures,
        *prog.toplevel_body,
    ]


def _top_level_value_info(
    stmt: VariableInitStmt | VariableInitBlockStmt,
    *,
    module_name: str | None,
) -> ImportedValueInfo | None:
    span = binding_name_span(stmt)
    if span is None:
        return None
    return ImportedValueInfo(
        name=stmt.name,
        declaration_span=span,
        type_expr=stmt.type_expr,
        module_name=module_name,
    )


def _binding_surface_for_program(
    prog: Program | ProgramWithValidation,
    *,
    source_path: Path | None,
    seen: set[Path],
    module_name: str | None,
) -> _SurfaceBindings:
    """Apply top-level writes in the same order as the runtime compiler.

    Local declarations and imports share one namespace in Kedi.  The LSP must
    therefore not simply merge all imports and then all locals: the final
    writer of a name is the declaration visible to consumers and re-exports.
    """
    bindings = _SurfaceBindings()
    for value in _python_value_infos(
        prog.prelude, prog.prelude_loc, module_name=module_name
    ).values():
        bindings.publish_value(value)

    for item in _ordered_toplevel_items(prog):
        if isinstance(item, ModuleImport):
            module_path = _lookup_import_path(item.module, source_path)
            if module_path is None:
                continue
            imported = load_imported_surface(module_path, seen, item.module)
            bindings.publish_import(imported.select(item.names))
        elif isinstance(item, ProcedureDef):
            bindings.publish_procedure(item)
        elif isinstance(item, TypeDef):
            bindings.publish_type(item)
        elif isinstance(item, ProfileDefStmt):
            bindings.publish_profile(item)
        elif isinstance(item, (VariableInitStmt, VariableInitBlockStmt)):
            value = _top_level_value_info(item, module_name=module_name)
            if value is not None:
                bindings.publish_value(value)
    return bindings


def imported_surface_for_program(
    prog: Program | ProgramWithValidation,
) -> ImportedSurface:
    root_path = program_path(prog)
    seen = {root_path} if root_path is not None else set()
    return _binding_surface_for_program(
        prog,
        source_path=root_path,
        seen=seen,
        module_name=None,
    ).imported_surface()


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

    bindings = _binding_surface_for_program(
        prog,
        source_path=path,
        seen=next_seen,
        module_name=module_name,
    ).surface

    export_names: list[str] = []
    for export in prog.exports:
        if export.export_all:
            export_names.extend(
                name for name in bindings.exported_names() if is_public_export_name(name)
            )
        export_names.extend(export.names)

    resolved = ImportedSurface(modules=dict(module_surface.modules))
    for name in dict.fromkeys(export_names):
        if name in bindings.procedures:
            resolved.procedures[name] = bindings.procedures[name]
        elif name in bindings.types:
            resolved.types[name] = bindings.types[name]
        elif name in bindings.values:
            resolved.values[name] = bindings.values[name]
        elif name in bindings.profiles:
            resolved.profiles[name] = bindings.profiles[name]
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
                _python_class_stub(stmt),
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


def _python_class_stub(stmt: py_ast.ClassDef) -> str:
    lines = [f"class {stmt.name}:"]
    for item in stmt.body:
        if isinstance(item, py_ast.AnnAssign) and isinstance(item.target, py_ast.Name):
            lines.append(f"    {item.target.id}: {_unparse_or_any(item.annotation)}")
        elif isinstance(item, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
            for decorator in item.decorator_list:
                decorator_name = py_ast.unparse(decorator)
                if decorator_name in {"property", "classmethod", "staticmethod"}:
                    lines.append(f"    @{decorator_name}")
            lines.append(f"    {_python_function_stub(item)}")
    if len(lines) == 1:
        lines.append("    pass")
    return "\n".join(lines)


def _unparse_or_any(expr: py_ast.AST | None) -> str:
    if expr is None:
        return "Any"
    return py_ast.unparse(expr)


def _python_name_span(loc: Span, name: str, lineno: int, col_offset: int) -> Span:
    start_line = loc.start_line + lineno
    start_col = col_offset + 1
    start_byte = loc.start_byte
    end_byte = loc.end_byte
    if loc.source_path:
        try:
            source = Path(loc.source_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            source = None
        if source is not None:
            source_lines = source.splitlines(keepends=True)
            if 0 < start_line <= len(source_lines):
                line_start = len("".join(source_lines[: start_line - 1]).encode("utf-8"))
                start_byte = line_start + col_offset
                end_byte = start_byte + len(name.encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=start_line,
        start_col=start_col,
        end_line=start_line,
        end_col=start_col + len(name),
        source_path=loc.source_path,
    )
