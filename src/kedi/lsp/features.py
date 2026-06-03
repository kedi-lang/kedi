"""LSP feature implementations for the Kedi language server.

The split keeps :mod:`kedi.lsp.server` thin — it owns the protocol
surface and dispatches to the functions in this module. Each function
takes raw source plus an LSP position / range and returns the LSP
response shape.

Features implemented here:

  - :func:`compute_diagnostics`
  - :func:`compute_semantic_tokens_full`  (B.1)
  - :func:`compute_document_symbols`      (B.2)
  - :func:`compute_references`            (B.3)
  - :func:`compute_prepare_rename`        (B.4)
  - :func:`compute_rename`                (B.4)
  - :func:`compute_formatting`            (B.5)
  - :func:`compute_signature_help`        (B.6)
  - :func:`compute_inlay_hints`           (B.7)
  - :func:`compute_definition`            (B.8 — extended)
  - :func:`compute_hover`
  - :func:`compute_python_ranges`         (B.9 — custom request)

Plus the constants ``SEMANTIC_TOKEN_TYPES`` / ``SEMANTIC_TOKEN_MODIFIERS``
the server publishes as a semantic-tokens legend.
"""

from __future__ import annotations

import ast as _py_ast
import builtins as _builtins
import importlib.resources
import typing as _typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from lsprotocol import types as lsp
from tree_sitter import Language as _TSLanguage
from tree_sitter import (
    Node,
    Tree,
)
from tree_sitter import Parser as _TSParser
from tree_sitter import Query as _TSQuery
from tree_sitter import QueryCursor as _TSQueryCursor

try:
    import tree_sitter_python as _ts_python_mod

    _PY_LANGUAGE = _TSLanguage(_ts_python_mod.language())
    _PY_PARSER = _TSParser(_PY_LANGUAGE)
    _PY_HIGHLIGHTS_SRC = (
        importlib.resources.files("tree_sitter_python") / "queries" / "highlights.scm"
    ).read_text()
    _PY_HIGHLIGHTS_QUERY = _TSQuery(_PY_LANGUAGE, _PY_HIGHLIGHTS_SRC)
except Exception:  # pragma: no cover — degraded mode if the wheel is missing
    _PY_PARSER = None
    _PY_LANGUAGE = None
    _PY_HIGHLIGHTS_QUERY = None

from ..lang.ast import (
    AssignBlockStmt,
    AssignStmt,
    AutoSpec,
    BacktickLineStmt,
    CallSegment,
    EvalDataset,
    EvalMetric,
    EvalSuite,
    InputSegment,
    OptimizeSpan,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    PythonBlock,
    PythonExprSegment,
    ReturnBlockStmt,
    ReturnStmt,
    Span,
    TemplateLine,
    TestSuite,
    TextSegment,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypeRef,
    TypeUnion,
    VarArg,
)
from ..lang.parser import KediDiagnostic, KediParseError, parse_program
from ..lang.parser.parser import get_language as _get_kedi_language
from ..lang.parser.parser import parse as _ts_parse
from ..module_registry import (
    describe_module_search_paths,
    find_module_path,
    module_search_paths,
    resolve_module_path,
)
from ..source_loading import load_kedi_source_bundle

# Kedi highlights + injection queries, loaded from the tree-sitter-kedi
# wheel's bundled ``queries/`` directory. These are the single source of
# truth for what the editor highlights — features.py just maps capture
# names to LSP token types.
try:
    _KEDI_LANGUAGE = _get_kedi_language()
    _KEDI_HIGHLIGHTS_SRC = (
        importlib.resources.files("tree_sitter_kedi") / "queries" / "highlights.scm"
    ).read_text()
    _KEDI_INJECTIONS_SRC = (
        importlib.resources.files("tree_sitter_kedi") / "queries" / "injections.scm"
    ).read_text()
    _KEDI_HIGHLIGHTS_QUERY = _TSQuery(_KEDI_LANGUAGE, _KEDI_HIGHLIGHTS_SRC)
    _KEDI_INJECTIONS_QUERY = _TSQuery(_KEDI_LANGUAGE, _KEDI_INJECTIONS_SRC)
except Exception:  # pragma: no cover
    _KEDI_HIGHLIGHTS_QUERY = None
    _KEDI_INJECTIONS_QUERY = None

try:
    import typing_extensions as _typing_extensions
except Exception:  # pragma: no cover - optional runtime dependency in degraded envs
    _typing_extensions = None


def _build_lsp_type_registry() -> dict[str, str]:
    registry: dict[str, str] = {}
    for name in dir(_builtins):
        value = getattr(_builtins, name, None)
        if isinstance(value, type) and not name.startswith("_"):
            registry[name] = "builtins"
    for module, origin in (
        (_typing, "typing"),
        (_typing_extensions, "typing_extensions"),
    ):
        if module is None:
            continue
        for name in getattr(module, "__all__", ()):
            if not name.startswith("_") and hasattr(module, name):
                registry.setdefault(name, origin)
    return registry


_LSP_TYPE_REGISTRY = _build_lsp_type_registry()


def _known_lsp_type_origin(name: str) -> Optional[str]:
    return _LSP_TYPE_REGISTRY.get(name)


# =====================================================================
# Position / range helpers
# =====================================================================


def _span_to_range(span: Span) -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=span.start_line - 1, character=span.start_col - 1),
        end=lsp.Position(line=span.end_line - 1, character=max(span.end_col - 1, 0)),
    )


def _position_inside(pos: lsp.Position, span: Span) -> bool:
    """Return True if a 0-based LSP position falls inside a 1-based ``Span``."""
    if pos.line + 1 < span.start_line or pos.line + 1 > span.end_line:
        return False
    if pos.line + 1 == span.start_line and pos.character + 1 < span.start_col:
        return False
    if pos.line + 1 == span.end_line and pos.character + 1 >= span.end_col:
        return False
    return True


def _segments_of(stmt) -> list:
    """All segment children of a statement (``segments`` or ``expr``)."""
    return list(getattr(stmt, "segments", []) or []) + list(getattr(stmt, "expr", []) or [])


# =====================================================================
# Diagnostics
# =====================================================================


def compute_diagnostics(source: str, source_path: Optional[str]) -> List[lsp.Diagnostic]:
    """Parse ``source`` and surface every ``KediDiagnostic`` as LSP."""
    try:
        prog = parse_program(source, source_path=source_path)
        return _missing_import_diagnostics(prog)
    except KediParseError as exc:
        return [_diag_to_lsp(d) for d in exc.diagnostics]


def _diag_to_lsp(d: KediDiagnostic) -> lsp.Diagnostic:
    severity = (
        lsp.DiagnosticSeverity.Error if d.severity == "error" else lsp.DiagnosticSeverity.Warning
    )
    return lsp.Diagnostic(
        range=_span_to_range(d.span),
        severity=severity,
        code=d.code,
        source="kedi",
        message=d.message,
    )


# =====================================================================
# Symbol resolution at a cursor position
# =====================================================================


@dataclass
class SymbolRef:
    """The symbol under the cursor, used by references / rename / hover."""

    kind: str  # "module" | "procedure" | "type" | "type_field" | "parameter" | "local" | "param_decl" | "output_decl"
    name: str
    declaration_span: Optional[Span]
    scope_proc: Optional[ProcedureDef]
    # The span of the identifier token under the cursor (for prepareRename).
    cursor_span: Span


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

    def merge(self, other: "ImportedSurface") -> None:
        self.modules.update(other.modules)
        self.procedures.update(other.procedures)
        self.types.update(other.types)
        self.values.update(other.values)


def _program_path(prog: Program | ProgramWithValidation) -> Optional[Path]:
    loc = getattr(prog, "loc", None)
    if loc is not None and loc.source_path:
        return Path(loc.source_path).resolve()
    source_map = getattr(prog, "source_map", None)
    documents = getattr(source_map, "documents", None)
    if documents:
        return Path(documents[0].path).resolve()
    return None


def _resolved_import_path(
    prog: Program | ProgramWithValidation, module_import_name: str
) -> Optional[Path]:
    try:
        return resolve_module_path(module_import_name, source_path=_program_path(prog))
    except (FileNotFoundError, RuntimeError):
        return None


def _assignment_name_span(stmt: AssignStmt | AssignBlockStmt) -> Optional[Span]:
    if stmt.loc is None:
        return None
    name_start_col = stmt.loc.start_col + 1  # skip '['
    return Span(
        start_byte=stmt.loc.start_byte + 1,
        end_byte=stmt.loc.start_byte + 1 + len(stmt.name),
        start_line=stmt.loc.start_line,
        start_col=name_start_col,
        end_line=stmt.loc.start_line,
        end_col=name_start_col + len(stmt.name),
        source_path=stmt.loc.source_path,
    )


def _top_level_value_infos(prog: Program | ProgramWithValidation) -> dict[str, ImportedValueInfo]:
    return {
        stmt.name: ImportedValueInfo(
            name=stmt.name,
            declaration_span=span,
            type_expr=stmt.type_expr,
        )
        for stmt in prog.toplevel_body
        if isinstance(stmt, (AssignStmt, AssignBlockStmt))
        for span in [(_assignment_name_span(stmt))]
        if span is not None
    }


def _is_public_export_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _program_line_text(
    prog: Program | ProgramWithValidation,
    line_number: int,
    source_path: Optional[str] = None,
) -> Optional[str]:
    if prog.source_text:
        lines = prog.source_text.splitlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
    source_map = getattr(prog, "source_map", None)
    if source_map is not None:
        path = source_path or getattr(getattr(prog, "loc", None), "source_path", None)
        if path:
            return source_map.get_line(path, line_number)
    return None


def _extract_module_docstring(source: str) -> Optional[str]:
    tree = _ts_parse(source.encode("utf-8"))
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


def _import_name_span(
    prog: Program | ProgramWithValidation,
    module_import,
) -> Optional[Span]:
    if module_import.loc is None:
        return None
    line = _program_line_text(prog, module_import.loc.start_line, module_import.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    name_index = line.find(module_import.module, search_start)
    if name_index < 0:
        return None
    prefix = line[:name_index]
    start_col = name_index + 1
    start_byte = module_import.loc.start_byte + len(prefix.encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(module_import.module.encode("utf-8")),
        start_line=module_import.loc.start_line,
        start_col=start_col,
        end_line=module_import.loc.start_line,
        end_col=start_col + len(module_import.module),
        source_path=module_import.loc.source_path,
    )


def _exported_name_at_position(
    prog: Program | ProgramWithValidation,
    pos: lsp.Position,
) -> Optional[tuple[str, Span]]:
    for export in getattr(prog, "exports", []):
        if export.loc is None or not _position_inside(pos, export.loc):
            continue
        if pos.line + 1 <= export.loc.start_line:
            continue
        line = _program_line_text(prog, pos.line + 1, export.loc.source_path)
        if line is None:
            continue
        export_name = line.split("#", 1)[0].strip()
        if not export_name or export_name == "*" or not _is_identifier(export_name):
            continue
        name_index = line.find(export_name)
        if name_index < 0:
            continue
        start_col = name_index + 1
        start_byte = export.loc.start_byte + len(line[:name_index].encode("utf-8"))
        span = Span(
            start_byte=start_byte,
            end_byte=start_byte + len(export_name.encode("utf-8")),
            start_line=pos.line + 1,
            start_col=start_col,
            end_line=pos.line + 1,
            end_col=start_col + len(export_name),
            source_path=export.loc.source_path,
        )
        if _position_inside(pos, span):
            return export_name, span
    return None


def _imported_surface_for_program(prog: Program | ProgramWithValidation) -> ImportedSurface:
    surface = ImportedSurface()
    root_path = _program_path(prog)
    seen = {root_path} if root_path is not None else set()
    for module_import in getattr(prog, "imports", []):
        module_path = _resolved_import_path(prog, module_import.module)
        if module_path is None:
            continue
        surface.merge(_load_imported_surface(module_path, seen, module_import.module))
    return surface


def _load_imported_surface(
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
            docstring=_extract_module_docstring(loaded.text),
        )

    nested_surface = ImportedSurface()
    for module_import in getattr(prog, "imports", []):
        resolved_nested_path = find_module_path(module_import.module, source_path=path)
        if resolved_nested_path is None:
            continue
        nested_surface.merge(
            _load_imported_surface(resolved_nested_path, next_seen, module_import.module)
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
        for span in [(_assignment_name_span(stmt))]
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
                if _is_public_export_name(name)
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
    return resolved


def _missing_import_diagnostics(
    prog: Program | ProgramWithValidation,
) -> List[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for module_import in getattr(prog, "imports", []):
        module_path = _resolved_import_path(prog, module_import.module)
        if module_path is not None:
            continue
        span = _import_name_span(prog, module_import) or module_import.loc
        if span is None:
            continue
        paths = module_search_paths(module_import.module, source_path=_program_path(prog))
        diagnostics.append(
            lsp.Diagnostic(
                range=_span_to_range(span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-IMPORT-NOT-FOUND",
                source="kedi",
                message=(
                    f"imported module '{module_import.module}' not found: "
                    f"{describe_module_search_paths(paths)}"
                ),
            )
        )
    return diagnostics


def _resolve_symbol_at(
    prog: Union[Program, ProgramWithValidation], pos: lsp.Position
) -> Optional[SymbolRef]:
    """Identify what the cursor is on. Returns ``None`` if nothing useful."""
    imported_surface = _imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in _all_procedures(prog)}
    proc_by_name.update(imported_surface.procedures)
    type_by_name = {t.name: t for t in prog.type_defs}
    type_by_name.update(imported_surface.types)
    top_level_values = _top_level_value_infos(prog)

    for module_import in getattr(prog, "imports", []):
        name_span = _import_name_span(prog, module_import)
        if name_span is None or not _position_inside(pos, name_span):
            continue
        module_info = imported_surface.modules.get(module_import.module)
        return SymbolRef(
            kind="module",
            name=module_import.module,
            declaration_span=module_info.declaration_span if module_info is not None else None,
            scope_proc=None,
            cursor_span=name_span,
        )

    export_name_hit = _exported_name_at_position(prog, pos)
    if export_name_hit is not None:
        export_name, export_span = export_name_hit
        if export_name in proc_by_name:
            target = proc_by_name[export_name]
            return SymbolRef(
                kind="procedure",
                name=export_name,
                declaration_span=target.loc,
                scope_proc=None,
                cursor_span=export_span,
            )
        if export_name in type_by_name:
            target = type_by_name[export_name]
            return SymbolRef(
                kind="type",
                name=export_name,
                declaration_span=target.loc,
                scope_proc=None,
                cursor_span=export_span,
            )
        if export_name in top_level_values:
            target = top_level_values[export_name]
            return SymbolRef(
                kind="module_value",
                name=export_name,
                declaration_span=target.declaration_span,
                scope_proc=None,
                cursor_span=export_span,
            )
        if export_name in imported_surface.values:
            target = imported_surface.values[export_name]
            return SymbolRef(
                kind="imported_value",
                name=export_name,
                declaration_span=target.declaration_span,
                scope_proc=None,
                cursor_span=export_span,
            )

    # Walk procedures first — they own the deepest spans.
    for proc in prog.procedures:
        r = _resolve_in_procedure(proc, pos, proc_by_name, type_by_name, imported_surface.values)
        if r is not None:
            return r

    # Top-level
    for stmt in prog.toplevel_body:
        r = _resolve_in_stmt(
            stmt,
            pos,
            None,
            proc_by_name,
            type_by_name,
            imported_surface.values,
        )
        if r is not None:
            return r
    if prog.toplevel_return is not None:
        r = _resolve_in_stmt(
            prog.toplevel_return,
            pos,
            None,
            proc_by_name,
            type_by_name,
            imported_surface.values,
        )
        if r is not None:
            return r
    # Type defs (top-level).
    for td in prog.type_defs:
        r = _resolve_in_type_def(td, pos, type_by_name)
        if r is not None:
            return r
    return None


def _resolve_in_procedure(
    proc: ProcedureDef,
    pos: lsp.Position,
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
) -> Optional[SymbolRef]:
    if proc.loc is None or not _position_inside(pos, proc.loc):
        return None
    for child in _nested_procedures(proc):
        r = _resolve_in_procedure(child, pos, proc_by_name, type_by_name, imported_values)
        if r is not None:
            return r
    if pos.line + 1 == proc.loc.start_line and _position_in_proc_name(proc, pos):
        return SymbolRef(
            kind="procedure",
            name=proc.name,
            declaration_span=proc.loc,
            scope_proc=None,
            cursor_span=_proc_name_span(proc),
        )
    for pname, pspan in _param_decls(proc):
        if _position_inside(pos, pspan):
            return SymbolRef(
                kind="param_decl",
                name=pname,
                declaration_span=pspan,
                scope_proc=proc,
                cursor_span=pspan,
            )
    r = _resolve_in_proc_signature_types(proc, pos, type_by_name)
    if r is not None:
        return r
    for stmt in proc.body:
        r = _resolve_in_stmt(stmt, pos, proc, proc_by_name, type_by_name, imported_values)
        if r is not None:
            return r
    return None


def _position_in_proc_name(proc: ProcedureDef, pos: lsp.Position) -> bool:
    """True iff the cursor is exactly on the procedure's name token.

    The header `@<name>(...)` starts at column 1 in 1-based terms. The
    name occupies columns 2..2+len(name).
    """
    if pos.line + 1 != proc.loc.start_line:
        return False
    name_start_col = proc.loc.start_col + 1  # skip '@'
    return name_start_col <= pos.character + 1 < name_start_col + len(proc.name)


def _proc_name_span(proc: ProcedureDef) -> Span:
    name_start_col = proc.loc.start_col + 1
    return Span(
        start_byte=proc.loc.start_byte + 1,
        end_byte=proc.loc.start_byte + 1 + len(proc.name),
        start_line=proc.loc.start_line,
        start_col=name_start_col,
        end_line=proc.loc.start_line,
        end_col=name_start_col + len(proc.name),
        source_path=proc.loc.source_path,
    )


def _param_decls(proc: ProcedureDef) -> List[Tuple[str, Span]]:
    """Best-effort span for each parameter declaration in the header.

    We don't carry parameter spans in the AST (the walker captures the
    procedure-level Span only), so we reconstruct them by searching
    the procedure's first physical line. Cheap and good enough for
    cursor hit-testing on a single line.
    """
    if proc.loc is None or not proc.params:
        return []
    return []  # Conservative for now: skip param-decl hit-testing in symbol
    # resolution. The header-level hover branch covers the common case.


def _resolve_in_proc_signature_types(
    proc: ProcedureDef, pos: lsp.Position, type_by_name: dict
) -> Optional[SymbolRef]:
    """Resolve type names used in a procedure signature."""
    for expr in (proc.param_type_exprs or {}).values():
        r = _resolve_in_type_expr(expr, pos, type_by_name)
        if r is not None:
            return r
    return _resolve_in_type_expr(proc.return_type_expr, pos, type_by_name)


def _resolve_in_stmt(
    stmt,
    pos: lsp.Position,
    proc: Optional[ProcedureDef],
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
) -> Optional[SymbolRef]:
    if not hasattr(stmt, "loc") or stmt.loc is None or not _position_inside(pos, stmt.loc):
        return None
    # AssignStmt / AssignBlockStmt target name
    if isinstance(stmt, (AssignStmt, AssignBlockStmt)) and stmt.loc is not None:
        # The target identifier lives just inside the `[name` literal.
        # Without an explicit span on the name we approximate.
        # Skip cursor hit-test on the target for now; references work
        # via the segment search.
        pass
    if isinstance(stmt, AssignStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name)
        if r is not None:
            return r
    if isinstance(stmt, AssignBlockStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name)
        if r is not None:
            return r
    for seg in _segments_of(stmt):
        r = _resolve_in_segment(seg, pos, proc, proc_by_name, type_by_name, imported_values)
        if r is not None:
            return r
    return None


def _resolve_in_segment(
    seg,
    pos: lsp.Position,
    proc: Optional[ProcedureDef],
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
) -> Optional[SymbolRef]:
    if not hasattr(seg, "loc") or seg.loc is None or not _position_inside(pos, seg.loc):
        return None
    if isinstance(seg, InputSegment):
        # Could be a parameter, an output, or an assign-target reference.
        if proc is not None and proc.params and seg.name in proc.params:
            return SymbolRef(
                kind="parameter",
                name=seg.name,
                declaration_span=proc.loc,  # imprecise but useful
                scope_proc=proc,
                cursor_span=seg.loc,
            )
        decl = _find_local_declaration(proc, seg.name) if proc else None
        if decl is None and seg.name in imported_values:
            return SymbolRef(
                kind="imported_value",
                name=seg.name,
                declaration_span=imported_values[seg.name].declaration_span,
                scope_proc=proc,
                cursor_span=seg.loc,
            )
        return SymbolRef(
            kind="local",
            name=seg.name,
            declaration_span=decl,
            scope_proc=proc,
            cursor_span=seg.loc,
        )
    if isinstance(seg, CallSegment):
        # Cursor on the call NAME? Identifier sits at start_col+1..+len(name).
        name_start = seg.loc.start_col + 1  # skip '<'
        if pos.line + 1 == seg.loc.start_line and (
            name_start <= pos.character + 1 < name_start + len(seg.name)
        ):
            target = proc_by_name.get(seg.name)
            return SymbolRef(
                kind="procedure",
                name=seg.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=Span(
                    start_byte=seg.loc.start_byte + 1,
                    end_byte=seg.loc.start_byte + 1 + len(seg.name),
                    start_line=seg.loc.start_line,
                    start_col=name_start,
                    end_line=seg.loc.start_line,
                    end_col=name_start + len(seg.name),
                    source_path=seg.loc.source_path,
                ),
            )
        # Recurse into args.

        for arg in seg.args or []:
            for inner_seg_attr in (
                "__call_segment__",
                "__python_segment__",
            ):
                inner = getattr(arg, inner_seg_attr, None)
                if inner is not None:
                    r = _resolve_in_segment(
                        inner,
                        pos,
                        proc,
                        proc_by_name,
                        type_by_name,
                        imported_values,
                    )
                    if r is not None:
                        return r
            rendered = getattr(arg, "__render_segments__", None) or []
            for inner in rendered:
                r = _resolve_in_segment(
                    inner,
                    pos,
                    proc,
                    proc_by_name,
                    type_by_name,
                    imported_values,
                )
                if r is not None:
                    return r
    if isinstance(seg, OutputSegment):
        # The name occupies the first identifier after `[`.
        name_start = seg.loc.start_col + 1
        if pos.line + 1 == seg.loc.start_line and (
            name_start <= pos.character + 1 < name_start + len(seg.name)
        ):
            return SymbolRef(
                kind="output_decl",
                name=seg.name,
                declaration_span=seg.loc,
                scope_proc=proc,
                cursor_span=Span(
                    start_byte=seg.loc.start_byte + 1,
                    end_byte=seg.loc.start_byte + 1 + len(seg.name),
                    start_line=seg.loc.start_line,
                    start_col=name_start,
                    end_line=seg.loc.start_line,
                    end_col=name_start + len(seg.name),
                    source_path=seg.loc.source_path,
                ),
            )
        # type annotation
        if seg.type_expr is not None:
            r = _resolve_in_type_expr(seg.type_expr, pos, type_by_name)
            if r is not None:
                return r
    return None


def _resolve_in_type_def(td: TypeDef, pos: lsp.Position, type_by_name: dict) -> Optional[SymbolRef]:
    if td.loc is None or not _position_inside(pos, td.loc):
        return None
    # `~Name(...)` — the name starts at start_col + 1.
    name_start = td.loc.start_col + 1
    if pos.line + 1 == td.loc.start_line and (
        name_start <= pos.character + 1 < name_start + len(td.name)
    ):
        return SymbolRef(
            kind="type",
            name=td.name,
            declaration_span=td.loc,
            scope_proc=None,
            cursor_span=Span(
                start_byte=td.loc.start_byte + 1,
                end_byte=td.loc.start_byte + 1 + len(td.name),
                start_line=td.loc.start_line,
                start_col=name_start,
                end_line=td.loc.start_line,
                end_col=name_start + len(td.name),
                source_path=td.loc.source_path,
            ),
        )
    for f in td.fields:
        r = _resolve_in_type_expr(f.type_expr, pos, type_by_name)
        if r is not None:
            return r
        if f.loc is not None and _position_inside(pos, f.loc):
            return SymbolRef(
                kind="type_field",
                name=f.name,
                declaration_span=f.loc,
                scope_proc=None,
                cursor_span=f.loc,
            )
    return None


def _resolve_in_type_expr(
    expr: Optional[TypeExpr], pos: lsp.Position, type_by_name: dict
) -> Optional[SymbolRef]:
    if expr is None or expr.loc is None or not _position_inside(pos, expr.loc):
        return None
    if isinstance(expr, TypeRef):
        target = type_by_name.get(expr.name)
        return SymbolRef(
            kind="type",
            name=expr.name,
            declaration_span=target.loc if target is not None else None,
            scope_proc=None,
            cursor_span=expr.loc,
        )
    if isinstance(expr, TypeApply):
        # `Name[arg,...]` — cursor on Name?
        name_end_col = expr.loc.start_col + len(expr.name)
        on_name = (
            pos.line + 1 == expr.loc.start_line
            and expr.loc.start_col <= pos.character + 1 < name_end_col
        )
        if on_name:
            target = type_by_name.get(expr.name)
            return SymbolRef(
                kind="type",
                name=expr.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=Span(
                    start_byte=expr.loc.start_byte,
                    end_byte=expr.loc.start_byte + len(expr.name),
                    start_line=expr.loc.start_line,
                    start_col=expr.loc.start_col,
                    end_line=expr.loc.start_line,
                    end_col=name_end_col,
                    source_path=expr.loc.source_path,
                ),
            )
        for arg in expr.args:
            r = _resolve_in_type_expr(arg, pos, type_by_name)
            if r is not None:
                return r
    if isinstance(expr, TypeUnion):
        for child in (expr.left, expr.right):
            r = _resolve_in_type_expr(child, pos, type_by_name)
            if r is not None:
                return r
    return None


def _find_local_declaration(proc: Optional[ProcedureDef], name: str) -> Optional[Span]:
    """Find an output/assign-target named ``name`` in a procedure body."""
    if proc is None:
        return None
    for stmt in proc.body:
        if isinstance(stmt, (AssignStmt, AssignBlockStmt)) and stmt.name == name:
            return stmt.loc
        for seg in _segments_of(stmt):
            if isinstance(seg, OutputSegment) and seg.name == name:
                return seg.loc
    return None


def _nested_procedures(proc: ProcedureDef) -> list[ProcedureDef]:
    from ..lang.ast import ProcDefStmt

    return [
        stmt.proc for stmt in proc.body if isinstance(stmt, ProcDefStmt) and stmt.proc is not None
    ]


def _all_procedures(prog: Union[Program, ProgramWithValidation]) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        for child in _nested_procedures(proc):
            visit(child)

    for proc in prog.procedures:
        visit(proc)
    return out


# =====================================================================
# Hover
# =====================================================================


def compute_hover(
    source: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Hover]:
    """Markdown hover for the symbol under the cursor."""
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    if _position_in_header_inline_python(source, pos):
        return None
    if (doc_hover := _docstring_hover(prog, pos)) is not None:
        return doc_hover
    if _position_in_comment(source, pos):
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        # Fall back to "generic statement" hover so the cursor inside
        # known regions at least returns something descriptive.
        return _generic_hover(prog, pos)

    range_ = _span_to_range(sym.cursor_span)
    imported_surface = _imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in _all_procedures(prog)}
    proc_by_name.update(imported_surface.procedures)
    type_by_name = {t.name: t for t in prog.type_defs}
    type_by_name.update(imported_surface.types)

    if sym.kind == "module":
        module_info = imported_surface.modules.get(sym.name)
        if module_info is None:
            text = f"**Module** `{sym.name}` — import target not found."
        else:
            text = f"**Module** `{sym.name}`"
            if module_info.docstring:
                text += f"\n\n{module_info.docstring}"
    elif sym.kind == "procedure":
        target = proc_by_name.get(sym.name)
        if target is not None:
            text = _proc_signature_markdown(target)
        else:
            text = f"**{sym.name}** — procedure not found in this module."
    elif sym.kind == "type":
        target = type_by_name.get(sym.name)
        if target is not None:
            text = _type_signature_markdown(target)
        elif (origin := _known_lsp_type_origin(sym.name)) is not None:
            text = _known_type_markdown(sym.name, origin)
        elif sym.name in _prelude_defined_names(prog.prelude):
            text = _prelude_type_markdown(sym.name)
        elif sym.name in _module_scope_python_names(prog):
            text = _python_module_scope_type_markdown(sym.name)
        else:
            text = f"**{sym.name}** — type not defined in this module (resolved at runtime)."
    elif sym.kind == "parameter":
        proc_name = f" of `@{sym.scope_proc.name}`" if sym.scope_proc is not None else ""
        if sym.scope_proc is not None:
            text = f"**Parameter** `{sym.name}: {_proc_param_annotation(sym.scope_proc, sym.name)}`{proc_name}"
        else:
            text = f"**Parameter** `{sym.name}`{proc_name}"
    elif sym.kind == "local":
        text = f"**Local** `{sym.name}` — output or assignment in this procedure."
    elif sym.kind == "module_value":
        text = f"**Module value** `{sym.name}` — top-level assignment exported by this module."
    elif sym.kind == "imported_value":
        imported_value = imported_surface.values.get(sym.name)
        if imported_value is not None and imported_value.type_expr is not None:
            text = f"**Imported value** `{sym.name}: {_type_expr_to_str(imported_value.type_expr)}`"
        else:
            text = f"**Imported value** `{sym.name}`"
        if imported_value is not None and imported_value.module_name:
            text += f" — exported from `{imported_value.module_name}`."
        else:
            text += " — exported from another Kedi module."
    elif sym.kind == "output_decl":
        text = (
            f"**Output** `{sym.name}` — LLM-filled placeholder; "
            "the value becomes available as a local variable below."
        )
    elif sym.kind == "type_field":
        target_field = next(
            (
                field
                for typedef in prog.type_defs
                for field in typedef.fields
                if field.loc == sym.declaration_span and field.name == sym.name
            ),
            None,
        )
        if target_field is not None:
            text = f"**Field** `{sym.name}: {_type_field_annotation(target_field)}` — type definition member."
        else:
            text = f"**Field** `{sym.name}` — type definition member."
    else:
        text = f"**{sym.kind}** `{sym.name}`"
    return lsp.Hover(
        contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text),
        range=range_,
    )


def _generic_hover(prog, pos: lsp.Position) -> Optional[lsp.Hover]:
    """Fallback hover when no named symbol is under the cursor.

    Drill into the smallest enclosing AST node so the hover range
    matches the statement / segment the cursor is on — *not* the
    entire enclosing procedure. Multi-line statements that end lines
    with ``\\`` are handled naturally because the parser already
    records their ``loc.end_line`` past the first physical line.
    """
    target = _smallest_enclosing(prog, pos)
    if target is None or target.loc is None:
        return None
    text = _describe_ast_node(target)
    if text is None:
        return None
    return lsp.Hover(
        contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text),
        range=_span_to_range(target.loc),
    )


def _docstring_hover(
    prog: Program | ProgramWithValidation, pos: lsp.Position
) -> Optional[lsp.Hover]:
    for proc in _all_procedures(prog):
        if proc.docstring_loc is None or not _position_inside(pos, proc.docstring_loc):
            continue
        return lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value=_proc_signature_markdown(proc),
            ),
            range=_span_to_range(proc.docstring_loc),
        )
    return None


def _position_in_comment(source: str, pos: lsp.Position) -> bool:
    tree = _ts_parse(source.encode("utf-8"))

    def visit(node: Node) -> bool:
        if node.type in {"line_comment", "block_comment"}:
            span = Span(
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0] + 1,
                start_col=node.start_point[1] + 1,
                end_line=node.end_point[0] + 1,
                end_col=node.end_point[1] + 1,
            )
            return _position_inside(pos, span)
        for child in node.children:
            if child.is_named and visit(child):
                return True
        return False

    return visit(tree.root_node)


def _smallest_enclosing(prog, pos: lsp.Position):
    """Walk the AST and return the deepest node whose ``loc`` contains ``pos``.

    Used by both ``compute_hover`` (fallback) and any future feature
    that wants "what's under the cursor" granularity. The walk is
    explicit per AST class to stay fast on large files — no generic
    dataclass reflection.
    """
    best = None

    def visit(node) -> None:
        nonlocal best
        if node is None:
            return
        loc = getattr(node, "loc", None)
        if loc is None or not _position_inside(pos, loc):
            return
        # Provisionally accept this node, then try children for a
        # tighter match.
        best = node
        # Procedure → body + optimize spans + auto spec
        if isinstance(node, ProcedureDef):
            for stmt in node.body:
                visit(stmt)
            if node.auto_spec is not None:
                visit(node.auto_spec)
            for span in node.optimize_spans or []:
                visit(span)
            return
        if isinstance(node, ProcDefStmt) and node.proc is not None:
            visit(node.proc)
            return
        if isinstance(node, OptimizeSpan):
            for tl in node.template_lines or []:
                visit(tl)
            return
        # Statements with segment children
        if isinstance(node, (TemplateLine, ReturnStmt)):
            for seg in getattr(node, "segments", []) or []:
                visit(seg)
            return
        if isinstance(node, AssignStmt):
            for seg in getattr(node, "expr", []) or []:
                visit(seg)
            return
        # Type defs descend into fields
        if isinstance(node, TypeDef):
            for f in node.fields:
                visit(f)
            return
        # Validation suites (only present on ProgramWithValidation)
        if isinstance(node, TestSuite):
            for case in getattr(node, "cases", []) or []:
                visit(case)
            return
        if isinstance(node, EvalSuite):
            for ds in getattr(node, "datasets", []) or []:
                visit(ds)
            for ds in getattr(node, "test_datasets", []) or []:
                visit(ds)
            for m in getattr(node, "metrics", []) or []:
                visit(m)
            return
        # Segment containers: CallSegment may have nested args
        if isinstance(node, CallSegment):
            for arg in getattr(node, "args", []) or []:
                if isinstance(arg, VarArg):
                    for inner in arg.segments or []:
                        visit(inner)
            return

    # Top-level walk
    for proc in prog.procedures or []:
        visit(proc)
    for td in getattr(prog, "type_defs", []) or []:
        visit(td)
    for stmt in getattr(prog, "top_level_statements", []) or []:
        visit(stmt)
    if isinstance(prog, ProgramWithValidation):
        for s in prog.tests or []:
            visit(s)
        for s in prog.evals or []:
            visit(s)
    return best


def _describe_ast_node(node) -> Optional[str]:
    """Per-node hover description used by :func:`_generic_hover`."""
    if isinstance(node, ProcedureDef):
        return _proc_signature_markdown(node)
    if isinstance(node, TypeDef):
        return _type_signature_markdown(node)
    if isinstance(node, TypeField):
        ann = node.type_annotation or "str"
        default = f" = `{node.default_expr}`" if node.default_expr is not None else ""
        return f"**Type field** `{node.name}: {ann}{default}` — member of a Kedi type."
    if isinstance(node, TemplateLine):
        return (
            "**Template line** — text the LLM sees as a prompt segment. "
            "Inputs `<x>`, outputs `[x: T]`, calls `<f(...)>`, inline Python `` `...` ``, "
            "and literal text are stitched into one prompt."
        )
    if isinstance(node, AssignStmt):
        ann = node.type_annotation or "str"
        return (
            f"**Assignment** `[{node.name}: {ann}] = …` — binds the value to "
            f"a local for use later in this procedure."
        )
    if isinstance(node, AssignBlockStmt):
        ann = node.type_annotation or "str"
        return (
            f"**Python block assignment** `[{node.name}: {ann}] = ```…``` ` — "
            f"runs Python and binds the return value."
        )
    if isinstance(node, ReturnStmt):
        return "**Return** `= …` — final value emitted by this procedure."
    if isinstance(node, ReturnBlockStmt):
        return "**Python block return** `= ```…``` ` — runs Python and returns its value."
    if isinstance(node, BacktickLineStmt):
        return "**Inline Python statement** `` `…` `` — side-effecting Python line."
    if isinstance(node, PythonBlock):
        return "**Python block** — top-level ```python``` runs once as the module prelude."
    if isinstance(node, AutoSpec):
        return "**`> auto:`** — AI-generated procedure prompt; body is the spec."
    if isinstance(node, OptimizeSpan):
        return f"**`> optimize: {node.name}:`** — GEPA / DSPy optimisation target."
    if isinstance(node, TextSegment):
        text = (node.text or "").strip()
        preview = text[:40] + ("…" if len(text) > 40 else "")
        return f"**Text segment** — literal text in the prompt: `{preview}`."
    if isinstance(node, InputSegment):
        return f"**Input segment** `<{node.name}>` — substituted from a parameter or local."
    if isinstance(node, OutputSegment):
        ann = node.type_annotation or "str"
        return (
            f"**Output segment** `[{node.name}: {ann}]` — LLM-filled placeholder, "
            f"available as a local below this line."
        )
    if isinstance(node, CallSegment):
        return f"**Call segment** `<{node.name}(...)>` — invokes another procedure inline."
    if isinstance(node, PythonExprSegment):
        return "**Inline Python expression** `` `…` `` — evaluated and substituted into the prompt."
    if isinstance(node, EvalSuite):
        return f"**`@eval: {node.procedure_name}:`** — evaluation block."
    if isinstance(node, TestSuite):
        return f"**`@test: {node.procedure_name}:`** — test block."
    if isinstance(node, EvalDataset):
        return f"**Dataset** `> data: {node.name}` — input rows for evaluation."
    if isinstance(node, EvalMetric):
        return f"**Metric** `> metric: {node.name}` — scoring function for `@eval:`."
    return None


def _proc_signature_markdown(proc: ProcedureDef) -> str:
    params = ", ".join(_format_proc_param(proc, p) for p in proc.params)
    sig = f"`@{proc.name}({params})`"
    if proc.return_type_annotation:
        sig += f" → `{proc.return_type_annotation}`"
    desc = "Kedi procedure."
    if proc.auto_spec is not None:
        desc = "AI-generated procedure (`> auto:`)."
    elif proc.optimize_spans:
        desc = (
            "Procedure with optimization spans: "
            + ", ".join(s.name for s in proc.optimize_spans)
            + "."
        )
    if proc.docstring:
        return f"{sig}\n\n{desc}\n\n{proc.docstring}"
    return f"{sig}\n\n{desc}"


def _type_signature_markdown(td: TypeDef) -> str:
    fields = ", ".join(_format_type_field(f) for f in td.fields)
    return f"`~{td.name}({fields})`\n\nKedi type definition (compiled to a Pydantic model)."


def _type_expr_to_str(expr: TypeExpr) -> str:
    if isinstance(expr, TypeRef):
        return expr.name
    if isinstance(expr, TypeApply):
        return f"{expr.name}[{', '.join(_type_expr_to_str(arg) for arg in expr.args)}]"
    if isinstance(expr, TypeUnion):
        return f"{_type_expr_to_str(expr.left)} | {_type_expr_to_str(expr.right)}"
    return repr(expr)


def _format_proc_param(proc: ProcedureDef, name: str) -> str:
    piece = f"{name}: {_proc_param_annotation(proc, name)}"
    default = (proc.param_defaults or {}).get(name)
    if default is not None:
        piece += f" = `{default}`"
    return piece


def _format_type_field(field: TypeField) -> str:
    piece = f"{field.name}: {_type_field_annotation(field)}"
    if field.default_expr is not None:
        piece += f" = `{field.default_expr}`"
    return piece


def _proc_param_annotation(proc: ProcedureDef, name: str) -> str:
    return ((proc.param_type_annotations or {}).get(name) or "str").strip()


def _type_field_annotation(field: TypeField) -> str:
    return (field.type_annotation or "str").strip()


def _known_type_markdown(name: str, origin: str) -> str:
    return f"`{name}`\n\nKedi type resolved to Python's {origin}.{name}."


def _prelude_type_markdown(name: str) -> str:
    return f"`{name}`\n\nKedi type defined in the Python prelude."


def _python_module_scope_type_markdown(name: str) -> str:
    return f"`{name}`\n\nKedi type defined in a top-level Python block."


def _prelude_defined_names(code: str | None) -> set[str]:
    if not code:
        return set()
    try:
        module = _py_ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for stmt in module.body:
        if isinstance(stmt, (_py_ast.ClassDef, _py_ast.FunctionDef, _py_ast.AsyncFunctionDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (_py_ast.Import, _py_ast.ImportFrom)):
            for alias in stmt.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(stmt, (_py_ast.Assign, _py_ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, _py_ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, _py_ast.Name):
                    names.add(target.id)
    return names


def _module_scope_python_names(prog: Program | ProgramWithValidation) -> set[str]:
    names: set[str] = set()
    for stmt in prog.toplevel_body:
        if not isinstance(stmt, PythonBlock):
            break
        names.update(_prelude_defined_names(stmt.code))
    return names


# =====================================================================
# Definition
# =====================================================================


def compute_definition(
    source: str, uri: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Location]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None or sym.declaration_span is None:
        return None
    target_uri = uri
    if sym.declaration_span.source_path:
        decl_path = Path(sym.declaration_span.source_path).resolve()
        if source_path is None or decl_path != Path(source_path).resolve():
            target_uri = decl_path.as_uri()
    return lsp.Location(uri=target_uri, range=_span_to_range(sym.declaration_span))


# =====================================================================
# References
# =====================================================================


def compute_references(
    source: str,
    uri: str,
    source_path: Optional[str],
    pos: lsp.Position,
    include_declaration: bool,
) -> List[lsp.Location]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return []
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        return []
    spans = _find_all_references(prog, sym, include_declaration=include_declaration)
    return [lsp.Location(uri=uri, range=_span_to_range(s)) for s in spans]


def _find_all_references(
    prog: Union[Program, ProgramWithValidation],
    sym: SymbolRef,
    *,
    include_declaration: bool,
) -> List[Span]:
    out: List[Span] = []
    if include_declaration and sym.declaration_span is not None:
        out.append(sym.declaration_span)

    if sym.kind == "procedure":
        for p in prog.procedures:
            for stmt in p.body:
                for seg in _segments_of(stmt):
                    _collect_call_refs(seg, sym.name, out)
        for stmt in prog.toplevel_body:
            for seg in _segments_of(stmt):
                _collect_call_refs(seg, sym.name, out)
        if prog.toplevel_return is not None:
            for seg in _segments_of(prog.toplevel_return):
                _collect_call_refs(seg, sym.name, out)
    elif sym.kind == "type":
        for td in prog.type_defs:
            _collect_type_refs_in_type_def(td, sym.name, out)
        for stmt in prog.toplevel_body:
            _collect_type_refs_in_stmt(stmt, sym.name, out)
        for p in prog.procedures:
            if p.return_type_expr is not None:
                _collect_type_refs_in_expr(p.return_type_expr, sym.name, out)
            for v in (p.param_type_exprs or {}).values():
                if v is not None:
                    _collect_type_refs_in_expr(v, sym.name, out)
            for stmt in p.body:
                _collect_type_refs_in_stmt(stmt, sym.name, out)
    elif sym.kind in ("parameter", "param_decl", "local", "output_decl"):
        if sym.scope_proc is not None:
            for stmt in sym.scope_proc.body:
                for seg in _segments_of(stmt):
                    _collect_input_refs(seg, sym.name, out)
    # Deduplicate spans by (start_byte, end_byte) — extra-safe.
    seen: set = set()
    uniq: List[Span] = []
    for s in out:
        key = (s.start_byte, s.end_byte, s.start_line, s.start_col)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def _collect_call_refs(seg, name: str, out: List[Span]) -> None:
    if isinstance(seg, CallSegment):
        if seg.name == name and seg.loc is not None:
            name_start_col = seg.loc.start_col + 1
            out.append(
                Span(
                    start_byte=seg.loc.start_byte + 1,
                    end_byte=seg.loc.start_byte + 1 + len(seg.name),
                    start_line=seg.loc.start_line,
                    start_col=name_start_col,
                    end_line=seg.loc.start_line,
                    end_col=name_start_col + len(seg.name),
                    source_path=seg.loc.source_path,
                )
            )
        # Recurse args
        for arg in seg.args or []:
            for inner_attr in ("__call_segment__", "__python_segment__"):
                inner = getattr(arg, inner_attr, None)
                if inner is not None:
                    _collect_call_refs(inner, name, out)
            for inner in getattr(arg, "__render_segments__", None) or []:
                _collect_call_refs(inner, name, out)


def _collect_input_refs(seg, name: str, out: List[Span]) -> None:
    if isinstance(seg, InputSegment) and seg.name == name and seg.loc is not None:
        name_start_col = seg.loc.start_col + 1
        out.append(
            Span(
                start_byte=seg.loc.start_byte + 1,
                end_byte=seg.loc.start_byte + 1 + len(seg.name),
                start_line=seg.loc.start_line,
                start_col=name_start_col,
                end_line=seg.loc.start_line,
                end_col=name_start_col + len(seg.name),
                source_path=seg.loc.source_path,
            )
        )
    elif isinstance(seg, CallSegment):
        for arg in seg.args or []:
            for inner_attr in ("__call_segment__", "__python_segment__"):
                inner = getattr(arg, inner_attr, None)
                if inner is not None:
                    _collect_input_refs(inner, name, out)
            for inner in getattr(arg, "__render_segments__", None) or []:
                _collect_input_refs(inner, name, out)


def _collect_type_refs_in_stmt(stmt, name: str, out: List[Span]) -> None:
    for attr in ("type_expr",):
        v = getattr(stmt, attr, None)
        if v is not None:
            _collect_type_refs_in_expr(v, name, out)
    for seg in _segments_of(stmt):
        if isinstance(seg, OutputSegment) and seg.type_expr is not None:
            _collect_type_refs_in_expr(seg.type_expr, name, out)


def _collect_type_refs_in_type_def(td: TypeDef, name: str, out: List[Span]) -> None:
    for type_field in td.fields:
        _collect_type_refs_in_expr(type_field.type_expr, name, out)


def _collect_type_refs_in_expr(expr: Optional[TypeExpr], name: str, out: List[Span]) -> None:
    if expr is None:
        return
    if isinstance(expr, TypeRef) and expr.name == name and expr.loc is not None:
        out.append(expr.loc)
    elif isinstance(expr, TypeApply):
        if expr.name == name and expr.loc is not None:
            name_end_col = expr.loc.start_col + len(expr.name)
            out.append(
                Span(
                    start_byte=expr.loc.start_byte,
                    end_byte=expr.loc.start_byte + len(expr.name),
                    start_line=expr.loc.start_line,
                    start_col=expr.loc.start_col,
                    end_line=expr.loc.start_line,
                    end_col=name_end_col,
                    source_path=expr.loc.source_path,
                )
            )
        for arg in expr.args:
            _collect_type_refs_in_expr(arg, name, out)
    elif isinstance(expr, TypeUnion):
        _collect_type_refs_in_expr(expr.left, name, out)
        _collect_type_refs_in_expr(expr.right, name, out)


# =====================================================================
# Rename
# =====================================================================


def compute_prepare_rename(
    source: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Range]:
    """Return the precise identifier range for the editor's rename popup.

    Only renameable kinds are returned; everything else is ``None``
    (which tells the editor "can't rename here").
    """
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        return None
    if (
        sym.kind == "type"
        and sym.declaration_span is None
        and _known_lsp_type_origin(sym.name) is not None
    ):
        return None
    if (
        source_path is not None
        and sym.declaration_span is not None
        and sym.declaration_span.source_path is not None
        and Path(sym.declaration_span.source_path).resolve() != Path(source_path).resolve()
    ):
        return None
    if sym.kind not in (
        "procedure",
        "type",
        "type_field",
        "parameter",
        "param_decl",
        "local",
        "output_decl",
        "imported_value",
    ):
        return None
    return _span_to_range(sym.cursor_span)


def compute_rename(
    source: str,
    uri: str,
    source_path: Optional[str],
    pos: lsp.Position,
    new_name: str,
) -> Optional[lsp.WorkspaceEdit]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        return None
    if (
        sym.kind == "type"
        and sym.declaration_span is None
        and _known_lsp_type_origin(sym.name) is not None
    ):
        return None
    if (
        source_path is not None
        and sym.declaration_span is not None
        and sym.declaration_span.source_path is not None
        and Path(sym.declaration_span.source_path).resolve() != Path(source_path).resolve()
    ):
        return None
    # Reject non-identifier new names.
    if not _is_identifier(new_name):
        return None
    spans = _find_all_references(prog, sym, include_declaration=True)
    # For procedure / type / output declarations we also need to
    # rewrite the declaration identifier itself, not the whole declaring
    # node — synthesize a name-only span if needed.
    edits: List[lsp.TextEdit] = []
    for s in spans:
        edits.append(lsp.TextEdit(range=_span_to_range(s), new_text=new_name))
    # Always include the declaration identifier explicitly.
    if sym.cursor_span is not None:
        decl_edit = lsp.TextEdit(range=_span_to_range(sym.cursor_span), new_text=new_name)
        # de-dupe
        if not any(e.range == decl_edit.range and e.new_text == decl_edit.new_text for e in edits):
            edits.append(decl_edit)
    return lsp.WorkspaceEdit(changes={uri: edits})


def _is_identifier(s: str) -> bool:
    import re

    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s))


# =====================================================================
# Formatting
# =====================================================================


def compute_formatting(source: str) -> List[lsp.TextEdit]:
    """Conservative: trim trailing WS per line, single trailing newline."""
    lines = source.splitlines(keepends=False)
    cleaned = [ln.rstrip(" \t") for ln in lines]
    # Drop trailing empty lines, then ensure exactly one final newline.
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    formatted = "\n".join(cleaned) + "\n" if cleaned else ""

    if formatted == source:
        return []

    # Compute end position of the original document.
    if not source:
        return [
            lsp.TextEdit(
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=0),
                ),
                new_text=formatted,
            )
        ]
    end_line = source.count("\n")
    last_line_start = source.rfind("\n") + 1
    end_char = len(source) - last_line_start
    return [
        lsp.TextEdit(
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=end_line, character=end_char),
            ),
            new_text=formatted,
        )
    ]


# =====================================================================
# Signature help
# =====================================================================


def compute_signature_help(
    source: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.SignatureHelp]:
    """Provide parameter-list info while typing inside ``<func(...)>``."""
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    imported_surface = _imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in _all_procedures(prog)}
    proc_by_name.update(imported_surface.procedures)

    # Find the innermost call_segment containing the cursor.
    target_call: Optional[CallSegment] = None
    for proc in _all_procedures(prog):
        if proc.loc is None or not _position_inside(pos, proc.loc):
            continue
        for stmt in proc.body:
            call = _innermost_call_at(stmt, pos)
            if call is not None:
                target_call = call
    for stmt in prog.toplevel_body:
        call = _innermost_call_at(stmt, pos)
        if call is not None:
            target_call = call
    if prog.toplevel_return is not None:
        call = _innermost_call_at(prog.toplevel_return, pos)
        if call is not None:
            target_call = call

    if target_call is None:
        return None
    target = proc_by_name.get(target_call.name)
    if target is None or target.loc is None:
        return None

    # Build signature label and parameter ranges.
    params_label_parts: List[str] = []
    param_infos: List[lsp.ParameterInformation] = []
    cursor_byte_offset = 0  # within the label string
    label_prefix = f"@{target.name}("
    cursor_byte_offset = len(label_prefix)
    for i, pname in enumerate(target.params):
        piece = _format_proc_param(target, pname)
        params_label_parts.append(piece)
        start = cursor_byte_offset
        end = cursor_byte_offset + len(piece)
        param_infos.append(lsp.ParameterInformation(label=(start, end)))
        cursor_byte_offset = end + 2  # ", "
    label = f"{label_prefix}{', '.join(params_label_parts)})"
    if target.return_type_annotation:
        label += f" -> {target.return_type_annotation}"

    active_param = _active_param_index(target_call, pos)
    return lsp.SignatureHelp(
        signatures=[
            lsp.SignatureInformation(
                label=label,
                parameters=param_infos,
                active_parameter=active_param,
            )
        ],
        active_signature=0,
        active_parameter=active_param,
    )


def _innermost_call_at(stmt, pos: lsp.Position) -> Optional[CallSegment]:
    if not hasattr(stmt, "loc") or stmt.loc is None or not _position_inside(pos, stmt.loc):
        return None
    found: Optional[CallSegment] = None
    for seg in _segments_of(stmt):
        c = _innermost_call_in_segment(seg, pos)
        if c is not None:
            found = c
    return found


def _innermost_call_in_segment(seg, pos: lsp.Position) -> Optional[CallSegment]:
    if not hasattr(seg, "loc") or seg.loc is None or not _position_inside(pos, seg.loc):
        return None
    if isinstance(seg, CallSegment):
        # Recurse into nested calls first to find the innermost.
        deepest: Optional[CallSegment] = seg
        for arg in seg.args or []:
            for attr in ("__call_segment__", "__python_segment__"):
                inner = getattr(arg, attr, None)
                if inner is not None:
                    r = _innermost_call_in_segment(inner, pos)
                    if r is not None:
                        deepest = r
            for inner in getattr(arg, "__render_segments__", None) or []:
                r = _innermost_call_in_segment(inner, pos)
                if r is not None:
                    deepest = r
        return deepest
    return None


def _active_param_index(call: CallSegment, pos: lsp.Position) -> int:
    """Approximate: count commas BEFORE the cursor between the call's parens."""
    # The call spans <name(arg1, arg2)>. We don't have per-arg fine-
    # grained spans, so count commas in the source slice from after the
    # opening `(` to the cursor.
    # The position byte we'd need to map; we just clamp at param count.
    if call.args is None:
        return 0
    # Best effort: index of the arg whose span contains the cursor.
    for i, arg in enumerate(call.args):
        loc = getattr(arg, "loc", None)
        if loc is not None and _position_inside(pos, loc):
            return i
    # Past the end — last param.
    return max(0, len(call.args) - 1)


# =====================================================================
# Inlay hints
# =====================================================================


def compute_inlay_hints(
    source: str, source_path: Optional[str], rng: lsp.Range
) -> List[lsp.InlayHint]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return []
    out: List[lsp.InlayHint] = []
    proc_by_name = {p.name: p for p in _all_procedures(prog)}

    def visit_stmt(stmt, enclosing_proc: Optional[ProcedureDef]) -> None:
        for seg in _segments_of(stmt):
            visit_segment(seg, enclosing_proc)

    def visit_segment(seg, enclosing_proc: Optional[ProcedureDef]) -> None:
        if isinstance(seg, CallSegment):
            target = proc_by_name.get(seg.name)
            if target is not None and seg.args and target.params:
                for i, arg in enumerate(seg.args):
                    if i >= len(target.params):
                        break
                    loc = getattr(arg, "loc", None)
                    if loc is None or not _in_range(loc, rng):
                        continue
                    out.append(
                        lsp.InlayHint(
                            position=lsp.Position(
                                line=loc.start_line - 1, character=loc.start_col - 1
                            ),
                            label=f"{target.params[i]}: ",
                            kind=lsp.InlayHintKind.Parameter,
                            padding_right=False,
                            padding_left=False,
                        )
                    )
            for arg in seg.args or []:
                for attr in ("__call_segment__", "__python_segment__"):
                    inner = getattr(arg, attr, None)
                    if inner is not None:
                        visit_segment(inner, enclosing_proc)
                for inner in getattr(arg, "__render_segments__", None) or []:
                    visit_segment(inner, enclosing_proc)
        elif isinstance(seg, OutputSegment):
            if seg.type_expr is None and seg.type_annotation is None and seg.loc is not None:
                if not _in_range(seg.loc, rng):
                    return
                # Position the hint AT the closing `]` so it reads as
                # `[name: str]` visually.
                out.append(
                    lsp.InlayHint(
                        position=lsp.Position(
                            line=seg.loc.end_line - 1,
                            character=max(seg.loc.end_col - 2, 0),
                        ),
                        label=": str",
                        kind=lsp.InlayHintKind.Type,
                        padding_right=False,
                        padding_left=False,
                    )
                )

    for proc in _all_procedures(prog):
        for stmt in proc.body:
            visit_stmt(stmt, proc)
    for stmt in prog.toplevel_body:
        visit_stmt(stmt, None)
    if prog.toplevel_return is not None:
        visit_stmt(prog.toplevel_return, None)
    return out


def _in_range(loc: Span, rng: lsp.Range) -> bool:
    """Loose intersection check between an AST span and an LSP range."""
    if loc.end_line - 1 < rng.start.line:
        return False
    if loc.start_line - 1 > rng.end.line:
        return False
    return True


# =====================================================================
# Document symbols
# =====================================================================


def compute_document_symbols(source: str, source_path: Optional[str]) -> List[lsp.DocumentSymbol]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return []
    symbols: List[lsp.DocumentSymbol] = []
    for td in prog.type_defs:
        symbols.append(_type_def_symbol(td))
    for proc in prog.procedures:
        symbols.append(_proc_symbol(proc))
    if isinstance(prog, ProgramWithValidation):
        for suite in prog.tests:
            symbols.append(_test_suite_symbol(suite))
        for suite in prog.evals:
            symbols.append(_eval_suite_symbol(suite))
    return symbols


def _proc_symbol(proc: ProcedureDef) -> lsp.DocumentSymbol:
    children: List[lsp.DocumentSymbol] = []
    for stmt in proc.body:
        from ..lang.ast import ProcDefStmt

        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            children.append(_proc_symbol(stmt.proc))
    for span in proc.optimize_spans:
        if span.loc is not None:
            children.append(
                lsp.DocumentSymbol(
                    name=f"> optimize: {span.name}",
                    detail="optimize span",
                    kind=lsp.SymbolKind.Constructor,
                    range=_span_to_range(span.loc),
                    selection_range=_span_to_range(span.loc),
                    children=[],
                )
            )
    detail_parts = []
    if proc.params:
        detail_parts.append(
            "(" + ", ".join(f"{p}: {_proc_param_annotation(proc, p)}" for p in proc.params) + ")"
        )
    if proc.return_type_annotation:
        detail_parts.append(f"-> {proc.return_type_annotation}")
    return lsp.DocumentSymbol(
        name=proc.name,
        detail=" ".join(detail_parts),
        kind=lsp.SymbolKind.Function,
        range=_span_to_range(proc.loc) if proc.loc else _empty_range(),
        selection_range=_span_to_range(_proc_name_span(proc)) if proc.loc else _empty_range(),
        children=children,
    )


def _type_def_symbol(td: TypeDef) -> lsp.DocumentSymbol:
    children = [
        lsp.DocumentSymbol(
            name=f.name,
            detail=_type_field_annotation(f),
            kind=lsp.SymbolKind.Field,
            range=_span_to_range(f.loc) if f.loc else _empty_range(),
            selection_range=_span_to_range(f.loc) if f.loc else _empty_range(),
            children=[],
        )
        for f in td.fields
    ]
    return lsp.DocumentSymbol(
        name=td.name,
        detail=f"({len(td.fields)} fields)",
        kind=lsp.SymbolKind.Struct,
        range=_span_to_range(td.loc) if td.loc else _empty_range(),
        selection_range=_span_to_range(td.loc) if td.loc else _empty_range(),
        children=children,
    )


def _test_suite_symbol(suite: TestSuite) -> lsp.DocumentSymbol:
    children = [
        lsp.DocumentSymbol(
            name=c.name,
            detail="test case",
            kind=lsp.SymbolKind.Method,
            range=_span_to_range(c.loc) if c.loc else _empty_range(),
            selection_range=_span_to_range(c.loc) if c.loc else _empty_range(),
            children=[],
        )
        for c in suite.cases
    ]
    return lsp.DocumentSymbol(
        name=f"@test: {suite.procedure_name}",
        detail=f"{len(suite.cases)} cases",
        kind=lsp.SymbolKind.Namespace,
        range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        selection_range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        children=children,
    )


def _eval_suite_symbol(suite: EvalSuite) -> lsp.DocumentSymbol:
    children: List[lsp.DocumentSymbol] = []
    for name, ds in suite.datasets.items():
        children.append(
            lsp.DocumentSymbol(
                name=f"> data: {name}",
                detail="dataset",
                kind=lsp.SymbolKind.Variable,
                range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                selection_range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                children=[],
            )
        )
    for name, ds in suite.test_datasets.items():
        children.append(
            lsp.DocumentSymbol(
                name=f"> test_data: {name}",
                detail="test dataset",
                kind=lsp.SymbolKind.Variable,
                range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                selection_range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                children=[],
            )
        )
    for m in suite.metrics:
        children.append(
            lsp.DocumentSymbol(
                name=f"> metric: {m.name}",
                detail=f"(dataset={m.dataset_name})" if m.dataset_name else "metric",
                kind=lsp.SymbolKind.Function,
                range=_span_to_range(m.loc) if m.loc else _empty_range(),
                selection_range=_span_to_range(m.loc) if m.loc else _empty_range(),
                children=[],
            )
        )
    return lsp.DocumentSymbol(
        name=f"@eval: {suite.procedure_name}",
        detail=f"{len(suite.metrics)} metrics, {len(suite.datasets)} data",
        kind=lsp.SymbolKind.Namespace,
        range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        selection_range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        children=children,
    )


def _empty_range() -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=0, character=0),
        end=lsp.Position(line=0, character=0),
    )


# =====================================================================
# Semantic tokens
# =====================================================================


# Standard LSP semantic-token type list. The ORDER matters: clients
# reference token types by their index in the legend the server returns
# at initialization.
SEMANTIC_TOKEN_TYPES: List[str] = [
    "namespace",  # 0
    "type",  # 1
    "class",  # 2
    "enum",  # 3
    "interface",  # 4
    "struct",  # 5
    "typeParameter",  # 6
    "parameter",  # 7
    "variable",  # 8
    "property",  # 9
    "enumMember",  # 10
    "event",  # 11
    "function",  # 12
    "method",  # 13
    "macro",  # 14
    "keyword",  # 15
    "modifier",  # 16
    "comment",  # 17
    "string",  # 18
    "number",  # 19
    "regexp",  # 20
    "operator",  # 21
    "decorator",  # 22
]
SEMANTIC_TOKEN_MODIFIERS: List[str] = [
    "declaration",  # bit 0  = 1
    "definition",  # bit 1  = 2
    "readonly",  # bit 2  = 4
    "static",  # bit 3  = 8
    "deprecated",  # bit 4  = 16
    "abstract",  # bit 5  = 32
    "async",  # bit 6  = 64
    "modification",  # bit 7  = 128
    "documentation",  # bit 8  = 256
    "defaultLibrary",  # bit 9  = 512
]

_TT = {name: i for i, name in enumerate(SEMANTIC_TOKEN_TYPES)}
_TM = {name: 1 << i for i, name in enumerate(SEMANTIC_TOKEN_MODIFIERS)}


# Map tree-sitter highlight-capture names → (LSP token type, modifiers).
#
# This is the *single* place feature semantics turn into LSP wire
# encoding. Both the Kedi grammar's ``highlights.scm`` and
# ``tree-sitter-python``'s upstream ``highlights.scm`` are run as
# tree-sitter queries; their captures share the standard nvim-treesitter
# / Helix / Zed naming convention (``@function.call``,
# ``@variable.parameter``, …), so one mapping covers both.
#
# Lookup is **longest-prefix** in capture-name space: a node tagged
# ``@function.method`` will match the ``function.method`` entry; if no
# entry exists for the exact name, we walk down dotted segments
# (``function.method`` → ``function``) until we hit one. That way new
# refinements in upstream highlights.scm files degrade gracefully.
_CAPTURE_TO_LSP: dict = {
    # Comments
    "comment": ("comment", 0),
    "comment.line": ("comment", 0),
    "comment.block": ("comment", 0),
    # Strings
    "string": ("string", 0),
    "string.special": ("string", 0),
    "string.escape": ("string", 0),
    "escape": ("string", 0),
    # Numbers / literals
    "number": ("number", 0),
    "boolean": ("keyword", 0),
    "constant": ("variable", 0),
    "constant.builtin": ("keyword", 0),
    # Operators / punctuation
    "operator": ("operator", 0),
    # We deliberately leave ``punctuation.*`` out — colouring every
    # paren / colon / bracket looks noisy. Re-enable here if desired.
    # Keywords
    "keyword": ("keyword", 0),
    "keyword.return": ("keyword", 0),
    "keyword.function": ("keyword", 0),
    "keyword.operator": ("keyword", 0),
    "keyword.import": ("keyword", 0),
    "keyword.control": ("keyword", 0),
    # Functions / methods
    "function": ("function", 0),
    "function.call": ("function", 0),
    "function.method": ("method", 0),
    "function.builtin": ("function", 0),
    "function.macro": ("macro", 0),
    "method": ("method", 0),
    # Types
    "type": ("type", 0),
    "type.builtin": ("type", 0),
    "type.definition": ("type", 0),
    "constructor": ("class", 0),
    "class": ("class", 0),
    # Variables / params / properties
    "variable": ("variable", 0),
    "variable.parameter": ("parameter", 0),
    "variable.builtin": ("variable", 0),
    "parameter": ("parameter", 0),
    "property": ("property", 0),
    "attribute": ("property", 0),
    "field": ("property", 0),
    # Misc identifiers
    "decorator": ("decorator", 0),
    "namespace": ("namespace", 0),
    "module": ("namespace", 0),
    "label": ("enumMember", 0),
    "tag": ("variable", 0),
    "embedded": ("operator", 0),
}


def _capture_to_token(name: str) -> Optional[Tuple[int, int]]:
    """Resolve a tree-sitter highlight capture to (token_type_idx, mods).

    Falls back through dotted prefixes (``a.b.c`` → ``a.b`` → ``a``)
    when the exact capture name isn't mapped.
    """
    parts = name.split(".")
    while parts:
        key = ".".join(parts)
        hit = _CAPTURE_TO_LSP.get(key)
        if hit is not None:
            tt, mods = hit
            return _TT[tt], mods
        parts.pop()
    return None


def _procedure_docstring_nodes(root: Node) -> List[Node]:
    out: List[Node] = []

    def visit(node: Node) -> None:
        if node.type == "procedure_def":
            body = node.child_by_field_name("body")
            if body is not None:
                named = [child for child in body.children if child.is_named]
                if named and named[0].type == "block_comment":
                    out.append(named[0])
        for child in node.children:
            if child.is_named:
                visit(child)

    visit(root)
    return out


def compute_semantic_tokens_full(source: str, source_path: Optional[str]) -> lsp.SemanticTokens:
    """Run tree-sitter highlight queries and emit LSP semantic tokens.

    The pipeline mirrors what editors like Neovim/Helix do natively
    with tree-sitter:

      1. Parse the document with tree-sitter-kedi.
      2. Run ``tree-sitter-kedi/queries/highlights.scm`` → Kedi tokens.
      3. Run ``tree-sitter-kedi/queries/injections.scm`` → list of
         (range, embedded language) injection sites.
      4. For each Python injection, parse the region's raw bytes with
         tree-sitter-python and run *its own*
         ``queries/highlights.scm`` against the sub-tree.
      5. Translate sub-tree positions back into the Kedi document's
         coordinate space; merge with the Kedi tokens.
      6. Sort + delta-encode per the LSP spec.

    There is no hand-rolled Python (or Kedi) classifier — the SCM
    query files are the single source of truth.
    """
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    tree: Tree = _ts_parse(src_bytes)
    src_lines = source.split("\n")
    raw: List[Tuple[int, int, int, int, int]] = []

    # Collect Python-injection byte ranges *first* so any Kedi-level
    # token that falls inside them (e.g. the fallback ``(python_code)
    # @string.special`` capture in kedi's highlights.scm) can be
    # punched out — the upstream tree-sitter-python tokens are finer
    # and the LSP spec forbids overlapping semantic tokens.
    python_byte_ranges: List[Tuple[int, int]] = []
    if _KEDI_INJECTIONS_QUERY is not None:
        cursor = _TSQueryCursor(_KEDI_INJECTIONS_QUERY)
        for _pat_idx, caps in cursor.matches(tree.root_node):
            for region_node in caps.get("injection.content", []):
                python_byte_ranges.append((region_node.start_byte, region_node.end_byte))
    docstring_nodes = _procedure_docstring_nodes(tree.root_node)
    docstring_byte_ranges = [(node.start_byte, node.end_byte) for node in docstring_nodes]

    def in_python_region(node: Node) -> bool:
        for s, e in python_byte_ranges:
            if node.start_byte >= s and node.end_byte <= e:
                return True
        return False

    def in_docstring_region(node: Node) -> bool:
        for s, e in docstring_byte_ranges:
            if node.start_byte >= s and node.end_byte <= e:
                return True
        return False

    def emit(line: int, col: int, length: int, ttype: int, mods: int) -> None:
        if length <= 0:
            return
        raw.append((line, col, length, ttype, mods))

    def emit_node(node: Node, ttype: int, mods: int, base_line: int = 0, base_col: int = 0) -> None:
        sl, sc = node.start_point
        el, ec = node.end_point
        # Translate from the embedded sub-tree's coordinates back to
        # Kedi-document coordinates. The Python sub-tree was parsed in
        # isolation, so its line 0 col 0 corresponds to (base_line,
        # base_col) in the Kedi document. Lines past line 0 in the
        # sub-tree always start at col 0 (newlines are 1:1).
        kedi_sl = base_line + sl
        kedi_sc = (base_col + sc) if sl == 0 else sc
        kedi_el = base_line + el
        kedi_ec = (base_col + ec) if el == 0 else ec
        if kedi_sl == kedi_el:
            emit(kedi_sl, kedi_sc, kedi_ec - kedi_sc, ttype, mods)
            return
        # Multi-line: split per line so the LSP spec is satisfied.
        if kedi_sl < len(src_lines):
            emit(
                kedi_sl,
                kedi_sc,
                max(0, len(src_lines[kedi_sl]) - kedi_sc),
                ttype,
                mods,
            )
        for li in range(kedi_sl + 1, kedi_el):
            if li < len(src_lines):
                emit(li, 0, len(src_lines[li]), ttype, mods)
        if kedi_ec > 0 and kedi_el < len(src_lines):
            emit(kedi_el, 0, kedi_ec, ttype, mods)

    # ---- 1. Kedi tokens via highlights.scm ----------------------------
    if _KEDI_HIGHLIGHTS_QUERY is not None:
        _emit_query_captures(
            _KEDI_HIGHLIGHTS_QUERY,
            tree.root_node,
            emit_node,
            base_line=0,
            base_col=0,
            skip_if=lambda node: in_python_region(node) or in_docstring_region(node),
        )
    docstring_token = _capture_to_token("variable")
    if docstring_token is not None:
        ttype, mods = docstring_token
        for node in docstring_nodes:
            emit_node(node, ttype, mods)

    # ---- 2. Embedded Python regions via injections.scm ---------------
    if python_byte_ranges and _PY_PARSER is not None and _PY_HIGHLIGHTS_QUERY is not None:
        cursor = _TSQueryCursor(_KEDI_INJECTIONS_QUERY)
        for _pat_idx, caps in cursor.matches(tree.root_node):
            for region_node in caps.get("injection.content", []):
                py_bytes = region_node.text
                if not py_bytes:
                    continue
                try:
                    py_tree = _PY_PARSER.parse(py_bytes)
                except Exception:  # pragma: no cover — defensive
                    continue
                py_base_line, py_base_col = region_node.start_point
                _emit_query_captures(
                    _PY_HIGHLIGHTS_QUERY,
                    py_tree.root_node,
                    emit_node,
                    base_line=py_base_line,
                    base_col=py_base_col,
                )

    # ---- 3. Resolve overlaps + LSP delta-encode ----------------------
    # The LSP spec forbids overlapping semantic tokens. Highlights
    # queries routinely produce overlap — e.g. Python's f-string
    # ``(string)`` covers the whole literal while ``(interpolation)``
    # and identifier captures sit inside it, and pattern refinement
    # (``(identifier) @variable`` vs ``(call ... @function)``) emits
    # two tokens on the same node. Resolve by "paint with specificity":
    # for each character column, the *smallest* covering token wins.
    resolved = _resolve_overlaps(raw)
    resolved.sort(key=lambda t: (t[0], t[1]))

    data: List[int] = []
    prev_line = 0
    prev_col = 0
    for line, col, length, ttype, mods in resolved:
        d_line = line - prev_line
        d_col = col - prev_col if d_line == 0 else col
        data.extend([d_line, d_col, length, ttype, mods])
        prev_line = line
        prev_col = col
    return lsp.SemanticTokens(data=data)


def _resolve_overlaps(
    tokens: List[Tuple[int, int, int, int, int]],
) -> List[Tuple[int, int, int, int, int]]:
    """Collapse overlapping tokens into non-overlapping runs.

    For each character position, the token with the *smallest* length
    that covers it wins (smaller = more specific). Ties go to whichever
    was emitted first. Output runs of equal classification are merged
    so we don't fragment unnecessarily.
    """
    by_line: dict = {}
    for line, col, length, ttype, mods in tokens:
        by_line.setdefault(line, []).append((col, length, ttype, mods))

    out: List[Tuple[int, int, int, int, int]] = []
    for line, line_tokens in by_line.items():
        if not line_tokens:
            continue
        max_col = max(c + n for c, n, _, _ in line_tokens)
        # `paint[i]` = (length, ttype, mods) where length acts as the
        # specificity tag. Smaller length wins on tie-break.
        paint: List[Optional[Tuple[int, int, int]]] = [None] * max_col
        # Apply tokens longest-first so shorter ones overwrite.
        for col, length, ttype, mods in sorted(line_tokens, key=lambda t: -t[1]):
            for i in range(col, col + length):
                if 0 <= i < max_col:
                    paint[i] = (length, ttype, mods)
        # Collapse contiguous runs of identical (ttype, mods).
        i = 0
        while i < max_col:
            slot = paint[i]
            if slot is None:
                i += 1
                continue
            _, ttype, mods = slot
            j = i
            while j < max_col and paint[j] is not None and paint[j][1:] == (ttype, mods):
                j += 1
            out.append((line, i, j - i, ttype, mods))
            i = j
    return out


def _emit_query_captures(
    query: _TSQuery,
    root: Node,
    emit_node,
    *,
    base_line: int,
    base_col: int,
    skip_if=None,
) -> None:
    """Run a highlights.scm-style query and emit tokens for each capture.

    Pattern order matters: tree-sitter returns matches in the order
    patterns appear in the .scm file, and the convention is that
    later patterns refine earlier ones (``(identifier) @variable``
    early, ``(function_definition name: (identifier) @function)``
    later). We feed all captures into the emit pipeline and let the
    dedup pass in :func:`compute_semantic_tokens_full` keep the last
    one for any identical (line, col, length).

    ``skip_if(node) -> bool`` lets the caller drop captures that fall
    inside an injection region (so an outer-language fallback
    capture doesn't overlap the inner-language injection tokens).
    """
    cursor = _TSQueryCursor(query)
    for _pat_idx, caps in cursor.matches(root):
        for cap_name, nodes_list in caps.items():
            mapped = _capture_to_token(cap_name)
            if mapped is None:
                continue
            ttype, mods = mapped
            for n in nodes_list:
                if skip_if is not None and skip_if(n):
                    continue
                emit_node(n, ttype, mods, base_line, base_col)


# =====================================================================
# kedi/pythonRanges — custom request
# =====================================================================


def compute_python_ranges(source: str) -> List[dict]:
    """Return byte / line ranges of every Python region in the document.

    Each entry is::

        {
          "kind": "fenced" | "inline",
          "range": {"start": {"line": ..., "character": ...},
                     "end":   {"line": ..., "character": ...}},
          "text":  "<the Python source (dedented for fenced, raw for inline)>"
        }

    Consumed by the VS Code extension to assemble a virtual Python
    document that mirrors the Kedi file's coordinate space so requests
    can be forwarded to a Python language server without translation.
    """
    return _collect_python_ranges(source, include_header_inline=False)


def compute_header_python_ranges(source: str) -> List[dict]:
    return _collect_python_ranges(source, include_regular_inline=False)


def _collect_python_ranges(
    source: str,
    *,
    include_header_inline: bool = True,
    include_regular_inline: bool = True,
) -> List[dict]:
    tree: Tree = _ts_parse(source.encode("utf-8") if isinstance(source, str) else source)
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    out: List[dict] = []

    def visit(node: Node, parents: tuple[Node, ...] = ()) -> None:
        if node.type == "python_code":
            text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")
            from ..lang.parser.escapes import dedent_block

            lines = text.split("\n")
            if lines and lines[-1] == "":
                lines = lines[:-1]
            dedented = "\n".join(dedent_block(lines))
            out.append(
                {
                    "kind": "fenced",
                    "range": _node_range(node),
                    "text": dedented,
                }
            )
        elif node.type == "python_inline_body":
            header_owner = _header_inline_owner(parents)
            if header_owner is not None and not include_header_inline:
                return
            if header_owner is None and not include_regular_inline:
                return
            text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")
            item = {
                "kind": "inline",
                "range": _node_range(node),
                "text": text,
            }
            if header_owner is not None:
                item.update(header_owner)
            out.append(item)
        for c in node.children:
            visit(c, parents + (node,))

    visit(tree.root_node)
    return out


def _position_in_header_inline_python(source: str, pos: lsp.Position) -> bool:
    for region in compute_header_python_ranges(source):
        if _position_in_range(pos, region["range"]):
            return True
    return False


def _position_in_range(pos: lsp.Position, range_: dict) -> bool:
    start = range_["start"]
    end = range_["end"]
    if pos.line < start["line"] or pos.line > end["line"]:
        return False
    if pos.line == start["line"] and pos.character < start["character"]:
        return False
    if pos.line == end["line"] and pos.character >= end["character"]:
        return False
    return True


def _node_range(node: Node) -> dict:
    return {
        "start": {"line": node.start_point[0], "character": node.start_point[1]},
        "end": {"line": node.end_point[0], "character": node.end_point[1]},
    }


def _header_inline_owner(parents: tuple[Node, ...]) -> Optional[dict]:
    owner = next(
        (parent for parent in reversed(parents) if parent.type in {"param", "type_field"}),
        None,
    )
    if owner is None:
        return None
    container = next(
        (parent for parent in reversed(parents) if parent.type in {"procedure_def", "type_def"}),
        None,
    )
    return {
        "ownerKind": owner.type,
        "containerKind": container.type if container is not None else None,
        "containerLine": container.start_point[0] if container is not None else None,
    }
