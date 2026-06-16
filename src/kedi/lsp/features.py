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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Tuple, Union

from lsprotocol import types as lsp
from tree_sitter import Language as _TSLanguage
from tree_sitter import Node
from tree_sitter import Parser as _TSParser
from tree_sitter import Query as _TSQuery
from tree_sitter import QueryCursor as _TSQueryCursor
from tree_sitter import Tree

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
    EffortDirectiveStmt,
    InputSegment,
    McpDirectiveStmt,
    McpField,
    ModelDirectiveStmt,
    OptimizeSpan,
    ProfileDefStmt,
    ProfileEffortEntry,
    ProfileMcpEntry,
    ProfileSystemEntry,
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
    SystemDirectiveStmt,
    TemplateBlock,
    TemplateLine,
    TestSuite,
    TextSegment,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypeRef,
    TypeUnion,
    UseDirectiveStmt,
)
from ..lang.parser import KediDiagnostic, KediParseError, parse_program
from ..lang.parser.parser import get_language as _get_kedi_language
from ..lang.parser.parser import parse as _ts_parse
from ..module_registry import describe_module_search_paths, module_search_paths
from .ast_helpers import assignment_name_span
from .ast_spans import name_token_span
from .module_surface import (
    ImportedSurface,
    ImportedValueInfo,
    imported_surface_for_program,
    program_path,
    resolved_import_path,
    top_level_value_infos,
)

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
    _typing_extensions: ModuleType | None = None


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
    if isinstance(stmt, TemplateBlock):
        segs: list = []
        for line in stmt.lines:
            segs.extend(line.segments)
        return segs
    return list(getattr(stmt, "segments", []) or []) + list(getattr(stmt, "expr", []) or [])


# =====================================================================
# Diagnostics
# =====================================================================


def compute_diagnostics(source: str, source_path: Optional[str]) -> List[lsp.Diagnostic]:
    """Parse ``source`` and surface every ``KediDiagnostic`` as LSP."""
    try:
        prog = parse_program(source, source_path=source_path)
        out = _missing_import_diagnostics(prog)
        for warning in prog.parser_warnings:
            out.append(_diag_to_lsp(warning))
        return out
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

    kind: str
    name: str
    declaration_span: Optional[Span]
    scope_proc: Optional[ProcedureDef]
    # The span of the identifier token under the cursor (for prepareRename).
    cursor_span: Span


def _merged_profile_defs(
    prog: Program | ProgramWithValidation,
    imported_surface: ImportedSurface,
) -> dict[str, ProfileDefStmt]:
    merged = dict(getattr(prog, "profile_defs", {}))
    merged.update(imported_surface.profiles)
    return merged


def _output_name_span(seg: OutputSegment) -> Optional[Span]:
    if seg.loc is None:
        return None
    return name_token_span(seg.loc, seg.name, prefix_len=1)


def _input_name_span(seg: InputSegment) -> Optional[Span]:
    if seg.loc is None:
        return None
    return name_token_span(seg.loc, seg.name, prefix_len=1)


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


def _line_start_byte(
    prog: Program | ProgramWithValidation,
    line_number: int,
) -> Optional[int]:
    if not prog.source_text:
        return None
    offset = 0
    for index, line in enumerate(prog.source_text.splitlines(keepends=True), start=1):
        if index == line_number:
            return offset
        offset += len(line.encode("utf-8"))
    return None


def _name_token_span_on_line(
    prog: Program | ProgramWithValidation,
    *,
    line_number: int,
    source_path: Optional[str],
    name: str,
    search_start: int = 0,
) -> Optional[Span]:
    line = _program_line_text(prog, line_number, source_path)
    line_start = _line_start_byte(prog, line_number)
    if line is None or line_start is None:
        return None
    for candidate in (f"`{name}`", name):
        name_index = line.find(candidate, search_start)
        if name_index < 0:
            continue
        start_col = name_index + 1
        start_byte = line_start + len(line[:name_index].encode("utf-8"))
        return Span(
            start_byte=start_byte,
            end_byte=start_byte + len(candidate.encode("utf-8")),
            start_line=line_number,
            start_col=start_col,
            end_line=line_number,
            end_col=start_col + len(candidate),
            source_path=source_path,
        )
    return None


def _use_tool_entry_spans(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
) -> list[tuple[str, Span]]:
    if stmt.loc is None or not stmt.tool_names:
        return []
    spans: list[tuple[str, Span]] = []
    seen: set[tuple[str, int]] = set()
    for line_number in range(stmt.loc.start_line, stmt.loc.end_line + 1):
        for tool_name in stmt.tool_names:
            name_span = _name_token_span_on_line(
                prog,
                line_number=line_number,
                source_path=stmt.loc.source_path,
                name=tool_name,
            )
            if name_span is None:
                continue
            key = (tool_name, name_span.start_line)
            if key in seen:
                continue
            seen.add(key)
            spans.append((tool_name, name_span))
    return spans


def _use_single_target_span(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
) -> Optional[Span]:
    if stmt.loc is None or not stmt.name:
        return None
    line = _program_line_text(prog, stmt.loc.start_line, stmt.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    return _name_token_span_on_line(
        prog,
        line_number=stmt.loc.start_line,
        source_path=stmt.loc.source_path,
        name=stmt.name,
        search_start=search_start,
    )


def _directive_name_span(
    prog: Program | ProgramWithValidation,
    loc: Span,
    name: str,
) -> Optional[Span]:
    if loc is None:
        return None
    line = _program_line_text(prog, loc.start_line, loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    name_index = line.find(name, search_start)
    if name_index < 0:
        return None
    prefix = line[:name_index]
    start_col = name_index + 1
    start_byte = loc.start_byte + len(prefix.encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(name.encode("utf-8")),
        start_line=loc.start_line,
        start_col=start_col,
        end_line=loc.start_line,
        end_col=start_col + len(name),
        source_path=loc.source_path,
    )


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


def _missing_import_diagnostics(
    prog: Program | ProgramWithValidation,
) -> List[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for module_import in getattr(prog, "imports", []):
        module_path = resolved_import_path(prog, module_import.module)
        if module_path is not None:
            continue
        span = _import_name_span(prog, module_import) or module_import.loc
        if span is None:
            continue
        paths = module_search_paths(module_import.module, source_path=program_path(prog))
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
    imported_surface = imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in _all_procedures(prog)}
    proc_by_name.update(imported_surface.procedures)
    type_by_name = {t.name: t for t in prog.type_defs}
    type_by_name.update(imported_surface.types)
    top_level_values = top_level_value_infos(prog)

    for module_import in getattr(prog, "imports", []):
        name_span = _import_name_span(prog, module_import)
        if name_span is None or not _position_inside(pos, name_span):
            continue
        module_info = imported_surface.modules.get(module_import.module)
        return SymbolRef(
            kind="module",
            name=module_import.module,
            declaration_span=(module_info.declaration_span if module_info is not None else None),
            scope_proc=None,
            cursor_span=name_span,
        )

    profile_defs = _merged_profile_defs(prog, imported_surface)
    for profile_name, profile_def in profile_defs.items():
        name_span = (
            _directive_name_span(prog, profile_def.loc, profile_name) if profile_def.loc else None
        )
        if name_span is not None and _position_inside(pos, name_span):
            return SymbolRef(
                kind="agent_profile",
                name=profile_name,
                declaration_span=profile_def.loc,
                scope_proc=None,
                cursor_span=name_span,
            )

    for item in getattr(prog, "ordered_toplevel", None) or []:
        if isinstance(item, UseDirectiveStmt) and item.loc is not None:
            if item.tool_names:
                for tool_name, name_span in _use_tool_entry_spans(prog, item):
                    if not _position_inside(pos, name_span):
                        continue
                    target = proc_by_name.get(tool_name)
                    if target is not None:
                        return SymbolRef(
                            kind="agent_tool",
                            name=tool_name,
                            declaration_span=target.loc,
                            scope_proc=None,
                            cursor_span=name_span,
                        )
                    return SymbolRef(
                        kind="agent_tool",
                        name=tool_name,
                        declaration_span=None,
                        scope_proc=None,
                        cursor_span=name_span,
                    )
            elif item.name:
                name_span = _use_single_target_span(prog, item)
                if name_span is not None and _position_inside(pos, name_span):
                    if item.name in proc_by_name:
                        target = proc_by_name[item.name]
                        return SymbolRef(
                            kind="agent_tool",
                            name=item.name,
                            declaration_span=target.loc,
                            scope_proc=None,
                            cursor_span=name_span,
                        )
                    profile_def = profile_defs.get(item.name)
                    return SymbolRef(
                        kind="agent_profile",
                        name=item.name,
                        declaration_span=(profile_def.loc if profile_def is not None else None),
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
        if export_name in profile_defs:
            profile_def = profile_defs[export_name]
            return SymbolRef(
                kind="agent_profile",
                name=export_name,
                declaration_span=profile_def.loc,
                scope_proc=None,
                cursor_span=export_span,
            )

    # Walk procedures first — they own the deepest spans.
    for proc in prog.procedures:
        r = _resolve_in_procedure(
            prog, proc, pos, proc_by_name, type_by_name, imported_surface.values
        )
        if r is not None:
            return r

    # Top-level
    for stmt in prog.toplevel_body:
        r = _resolve_in_stmt(
            prog,
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
            prog,
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
    prog: Program | ProgramWithValidation,
    proc: ProcedureDef,
    pos: lsp.Position,
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
) -> Optional[SymbolRef]:
    if proc.loc is None or not _position_inside(pos, proc.loc):
        return None
    for child in _nested_procedures(proc):
        r = _resolve_in_procedure(prog, child, pos, proc_by_name, type_by_name, imported_values)
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
        r = _resolve_in_stmt(prog, stmt, pos, proc, proc_by_name, type_by_name, imported_values)
        if r is not None:
            return r
    return None


def _position_in_proc_name(proc: ProcedureDef, pos: lsp.Position) -> bool:
    """True iff the cursor is exactly on the procedure's name token.

    The header `@<name>(...)` starts at column 1 in 1-based terms. The
    name occupies columns 2..2+len(name).
    """
    assert proc.loc is not None
    if pos.line + 1 != proc.loc.start_line:
        return False
    name_start_col = proc.loc.start_col + 1  # skip '@'
    return name_start_col <= pos.character + 1 < name_start_col + len(proc.name)


def _proc_name_span(proc: ProcedureDef) -> Span:
    assert proc.loc is not None
    return name_token_span(proc.loc, proc.name, prefix_len=1)


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
    prog: Program | ProgramWithValidation,
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
        name_span = assignment_name_span(stmt)
        if name_span is not None and _position_inside(pos, name_span):
            return SymbolRef(
                kind="local",
                name=stmt.name,
                declaration_span=name_span,
                scope_proc=proc,
                cursor_span=name_span,
            )
    if isinstance(stmt, AssignStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name)
        if r is not None:
            return r
    if isinstance(stmt, AssignBlockStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name)
        if r is not None:
            return r
    if isinstance(stmt, UseDirectiveStmt) and stmt.loc is not None:
        if stmt.tool_names:
            for tool_name, name_span in _use_tool_entry_spans(prog, stmt):
                if not _position_inside(pos, name_span):
                    continue
                target = proc_by_name.get(tool_name)
                if target is not None:
                    return SymbolRef(
                        kind="agent_tool",
                        name=tool_name,
                        declaration_span=target.loc,
                        scope_proc=proc,
                        cursor_span=name_span,
                    )
                return SymbolRef(
                    kind="agent_tool",
                    name=tool_name,
                    declaration_span=None,
                    scope_proc=proc,
                    cursor_span=name_span,
                )
        elif stmt.name:
            name_span = _use_single_target_span(prog, stmt)
            if name_span is not None and _position_inside(pos, name_span):
                if stmt.name in proc_by_name:
                    target = proc_by_name[stmt.name]
                    return SymbolRef(
                        kind="agent_tool",
                        name=stmt.name,
                        declaration_span=target.loc,
                        scope_proc=proc,
                        cursor_span=name_span,
                    )
                imported_surface = imported_surface_for_program(prog)
                profile_def = _merged_profile_defs(prog, imported_surface).get(stmt.name)
                return SymbolRef(
                    kind="agent_profile",
                    name=stmt.name,
                    declaration_span=(profile_def.loc if profile_def is not None else None),
                    scope_proc=proc,
                    cursor_span=name_span,
                )
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
        name_span = _input_name_span(seg) or seg.loc
        if proc is not None and proc.params and seg.name in proc.params:
            return SymbolRef(
                kind="parameter",
                name=seg.name,
                declaration_span=proc.loc,  # imprecise but useful
                scope_proc=proc,
                cursor_span=name_span,
            )
        decl = _find_local_declaration(proc, seg.name) if proc else None
        if decl is None and seg.name in imported_values:
            return SymbolRef(
                kind="imported_value",
                name=seg.name,
                declaration_span=imported_values[seg.name].declaration_span,
                scope_proc=proc,
                cursor_span=name_span,
            )
        if decl is None and seg.name == "args":
            return SymbolRef(
                kind="runtime_value",
                name=seg.name,
                declaration_span=None,
                scope_proc=proc,
                cursor_span=name_span,
            )
        return SymbolRef(
            kind="local",
            name=seg.name,
            declaration_span=decl,
            scope_proc=proc,
            cursor_span=name_span,
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
                cursor_span=name_token_span(seg.loc, seg.name, prefix_len=1),
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
        name_span = _output_name_span(seg)
        if name_span is not None and _position_inside(pos, name_span):
            return SymbolRef(
                kind="output_decl",
                name=seg.name,
                declaration_span=name_span,
                scope_proc=proc,
                cursor_span=name_span,
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
            cursor_span=name_token_span(td.loc, td.name, prefix_len=1),
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
            return assignment_name_span(stmt)
        for seg in _segments_of(stmt):
            if isinstance(seg, OutputSegment) and seg.name == name:
                return _output_name_span(seg)
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
    imported_surface = imported_surface_for_program(prog)
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
    elif sym.kind == "agent_tool":
        target = proc_by_name.get(sym.name)
        if target is not None:
            text = _agent_tool_markdown(sym.name, target)
        else:
            text = f"**Agent tool** `{sym.name}` — procedure not found in this module."
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
    elif sym.kind == "runtime_value":
        text = (
            "**Runtime value** `args: AttrDict[str, ArgType]` — CLI arguments passed after "
            "the Kedi source path. Named flags are available as attributes; missing flags "
            "resolve to `None`."
        )
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
    elif sym.kind == "agent_profile":
        profile_def = _merged_profile_defs(prog, imported_surface).get(sym.name)
        if profile_def is not None:
            text = _profile_signature_markdown(profile_def)
        else:
            text = _missing_agent_profile_hover(sym.name, imported_surface)
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
    for profile in getattr(prog, "profile_defs", {}).values():
        if profile.docstring_loc is None or not _position_inside(pos, profile.docstring_loc):
            continue
        return lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value=_profile_signature_markdown(profile),
            ),
            range=_span_to_range(profile.docstring_loc),
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
        if isinstance(node, ProfileDefStmt):
            for member in node.model_members:
                visit(member)
            for member in node.effort_members:
                visit(member)
            for member in node.system_members:
                visit(member)
            for member in node.mcp_members:
                visit(member)
            return
        if isinstance(node, SystemDirectiveStmt):
            for line in node.template_lines:
                visit(line)
            return
        if isinstance(node, McpDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileMcpEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileSystemEntry):
            for line in node.template_lines:
                visit(line)
            return
        if isinstance(node, ProcDefStmt) and node.proc is not None:
            visit(node.proc)
            return
        if isinstance(node, OptimizeSpan):
            for tl in node.template_lines or []:
                visit(tl)
            return
        # Statements with segment children
        if isinstance(node, TemplateBlock):
            for line in node.lines:
                visit(line)
            return
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
                for inner in getattr(arg, "__render_segments__", []) or []:
                    visit(inner)
            return

    if getattr(prog, "ordered_toplevel", None):
        for item in prog.ordered_toplevel:
            visit(item)
    else:
        for proc in prog.procedures or []:
            visit(proc)
        for stmt in (
            getattr(prog, "toplevel_body", None)
            or getattr(prog, "top_level_statements", None)
            or []
        ):
            visit(stmt)
    for td in getattr(prog, "type_defs", []) or []:
        visit(td)
    toplevel_return = getattr(prog, "toplevel_return", None)
    if toplevel_return is not None:
        visit(toplevel_return)
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
    if isinstance(node, TemplateBlock):
        line_count = len(node.lines)
        return (
            "**Template block** `>> …` — multiline LLM prompt run. "
            f"{line_count} line(s) are newline-joined into **one** agent call; "
            "`<name>` on later rows refers to `[name]` outputs declared earlier in the block."
        )
    if isinstance(node, TemplateLine):
        return (
            "**Template line** — one row in a template block (or a legacy bare line). "
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
    if isinstance(node, ModelDirectiveStmt):
        return _model_directive_markdown(node)
    if isinstance(node, EffortDirectiveStmt):
        return _effort_directive_markdown(node)
    if isinstance(node, SystemDirectiveStmt):
        return _system_directive_markdown(node)
    if isinstance(node, McpDirectiveStmt):
        return _mcp_directive_markdown(node)
    if isinstance(node, McpField):
        return _mcp_field_markdown(node)
    if isinstance(node, ProfileEffortEntry):
        return _profile_effort_entry_markdown(node)
    if isinstance(node, ProfileSystemEntry):
        return _profile_system_entry_markdown(node)
    if isinstance(node, ProfileMcpEntry):
        return _profile_mcp_entry_markdown(node)
    if isinstance(node, UseDirectiveStmt):
        return _use_directive_markdown(node)
    if isinstance(node, ProfileDefStmt):
        return _profile_signature_markdown(node)
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


def _agent_tool_markdown(name: str, proc: ProcedureDef) -> str:
    signature = _proc_signature_markdown(proc)
    headline = signature.splitlines()[0]
    return (
        f"**Agent tool** `{name}` — exposes {headline} "
        "to the active agent adapter for this block.\n\n"
        f"{signature}"
    )


def _use_directive_markdown(stmt: UseDirectiveStmt) -> str:
    if stmt.tool_names:
        tools = ", ".join(f"`{name}`" for name in stmt.tool_names)
        return (
            f"**`> use:`** — register Kedi procedures as agent tools: {tools}.\n\n"
            "Multiline form always lists tools. Inner `> use:` overrides outer tools "
            "for the current indentation block."
        )
    return (
        f"**`> use: {stmt.name}`** — when `{stmt.name}` names a procedure, register it "
        f"as an agent tool; otherwise merge agent profile `{stmt.name}` into the active "
        "block profile for subsequent statements."
    )


def _model_directive_markdown(stmt: ModelDirectiveStmt) -> str:
    if stmt.backtick_code is not None:
        value = f"`{stmt.backtick_code}`"
    else:
        value = f"`{stmt.plain_value or ''}`"
    return (
        f"**`> model:`** — set active agent model to {value} for subsequent "
        "statements in this indentation block."
    )


def _effort_directive_markdown(stmt: EffortDirectiveStmt) -> str:
    value = _effort_value_preview(stmt)
    return (
        f"**`> effort:`** — set active model reasoning effort to {value} for subsequent "
        "statements in this indentation block."
    )


def _system_directive_markdown(stmt: SystemDirectiveStmt) -> str:
    value = _system_value_preview(stmt)
    return (
        f"**`> system:`** — set active agent instructions to {value} for subsequent "
        "statements in this indentation block."
    )


def _mcp_directive_markdown(stmt: McpDirectiveStmt) -> str:
    fields = ", ".join(f"`{field.name}`" for field in stmt.fields) or "`<empty>`"
    return (
        f"**`> mcp:`** — load tools from an MCP server for the active agent scope. "
        f"Configured fields: {fields}."
    )


def _mcp_field_markdown(field: McpField) -> str:
    value = _mcp_field_value_preview(field)
    descriptions = {
        "transport": (
            "MCP transport. Accepted values: `stdio`, `sse`, `http`, "
            "or `streamable-http`; `http` is an alias for `streamable-http`."
        ),
        "command": "Command used to start a stdio MCP server.",
        "args": "String argument list passed to the stdio MCP command.",
        "env": "String environment dictionary passed to the stdio MCP process.",
        "url": "HTTP/SSE MCP server URL.",
        "headers": "String header dictionary sent to HTTP/SSE MCP servers.",
    }
    desc = descriptions.get(field.name, "MCP server configuration field.")
    return f"**MCP field** `{field.name}` = {value} — {desc}"


def _profile_mcp_entry_markdown(entry: ProfileMcpEntry) -> str:
    fields = ", ".join(f"`{field.name}`" for field in entry.fields) or "`<empty>`"
    return f"**`> mcp:`** — reusable profile MCP server configuration. Fields: {fields}."


def _profile_system_entry_markdown(entry: ProfileSystemEntry) -> str:
    value = _system_value_preview(entry)
    return f"**`> system:`** — reusable agent profile instructions: {value}."


def _profile_effort_entry_markdown(entry: ProfileEffortEntry) -> str:
    value = _effort_value_preview(entry)
    return f"**`> effort:`** — reusable agent profile reasoning effort: {value}."


def _effort_value_preview(stmt: EffortDirectiveStmt | ProfileEffortEntry) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _system_value_preview(stmt: SystemDirectiveStmt | ProfileSystemEntry) -> str:
    if getattr(stmt, "template_lines", None):
        parts: list[str] = []
        for index, line in enumerate(stmt.template_lines):
            if index:
                parts.append("\\n")
            for seg in line.segments:
                if isinstance(seg, TextSegment):
                    parts.append(seg.text)
                elif isinstance(seg, InputSegment):
                    parts.append(f"<{seg.name}>")
        text = "".join(parts).strip()
        return f"`{text}`" if text else "`<empty>`"
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _mcp_field_value_preview(field: McpField) -> str:
    if field.backtick_code is not None:
        return f"`{field.backtick_code}`"
    return f"`{field.plain_value or ''}`"


def _missing_agent_profile_hover(name: str, imported_surface: ImportedSurface) -> str:
    module_info = imported_surface.modules.get(name)
    if module_info is not None:
        from .module_surface import load_imported_surface

        exported = sorted(load_imported_surface(module_info.path, set(), name).profiles.keys())
        if exported:
            options = ", ".join(f"`> use: {profile}`" for profile in exported)
            return (
                f"**Agent profile** `{name}` — not found. "
                f"`{name}` is an imported **module**, not a profile name. "
                f"Exported profiles from that module: {options}."
            )
        return (
            f"**Agent profile** `{name}` — not found. "
            f"`{name}` is an imported module with no exported agent profiles."
        )
    return f"**Agent profile** `{name}` — referenced by `> use:`; definition not found."


def _profile_signature_markdown(profile: ProfileDefStmt) -> str:
    if profile.docstring:
        return f"**Agent profile** `> profile: {profile.name}:`\n\n{profile.docstring}"
    model_parts: list[str] = []
    for member in profile.model_members:
        if member.backtick_code is not None:
            model_parts.append(f"`{member.backtick_code}`")
        elif member.plain_value is not None:
            model_parts.append(f"`{member.plain_value}`")
    model_summary = ", ".join(model_parts) if model_parts else "(no model members)"
    effort_parts: list[str] = []
    for member in profile.effort_members:
        effort_parts.append(_effort_value_preview(member))
    effort_summary = ", ".join(effort_parts) if effort_parts else "(no effort members)"
    system_parts: list[str] = []
    for member in profile.system_members:
        value = _system_value_preview(member)
        if value != "``":
            system_parts.append(value)
    system_summary = ", ".join(system_parts) if system_parts else "(no system members)"
    tool_summary = (
        ", ".join(f"`{name}`" for name in profile.tool_names)
        if profile.tool_names
        else "(no tool members)"
    )
    return (
        f"**Agent profile** `> profile: {profile.name}:`\n\n"
        f"Reusable agent settings. `> model:` members: {model_summary}.\n\n"
        f"`> effort:` members: {effort_summary}.\n\n"
        f"`> system:` members: {system_summary}.\n\n"
        f"`> use:` tool members: {tool_summary}.\n\n"
        "Profile body values are evaluated when `> use:` runs."
    )


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
    elif getattr(proc, "captured_agent_profile", None) is not None:
        captured = proc.captured_agent_profile
        if getattr(captured, "model", None):
            desc = f"Kedi procedure (captured model: `{captured.model}`)."
        elif getattr(captured, "effort", None):
            desc = f"Kedi procedure (captured effort: `{captured.effort}`)."
    if proc.docstring:
        return f"{sig}\n\n{desc}\n\n{proc.docstring}"
    return f"{sig}\n\n{desc}"


def _type_signature_markdown(td: TypeDef) -> str:
    fields = ", ".join(_format_type_field(f) for f in td.fields)
    return f"`~{td.name}({fields})`\n\nKedi type definition (compiled to a Pydantic model)."


def _type_expr_to_str(expr: TypeExpr | None) -> str:
    if expr is None:
        return "None"
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
    elif sym.kind in (
        "parameter",
        "param_decl",
        "local",
        "output_decl",
    ):
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
            out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
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
        out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
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
    imported_surface = imported_surface_for_program(prog)
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
    for profile in getattr(prog, "profile_defs", {}).values():
        symbols.append(_profile_def_symbol(profile))
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
    for stmt in proc.body:
        if isinstance(stmt, ModelDirectiveStmt) and stmt.loc is not None:
            value = stmt.plain_value or (f"`{stmt.backtick_code}`" if stmt.backtick_code else "")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> model: {value}",
                    detail="agent model override",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, EffortDirectiveStmt) and stmt.loc is not None:
            value = stmt.plain_value or (f"`{stmt.backtick_code}`" if stmt.backtick_code else "")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> effort: {value}",
                    detail="agent reasoning effort",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, SystemDirectiveStmt) and stmt.loc is not None:
            value = _system_value_preview(stmt).strip("`")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> system: {value}",
                    detail="agent instructions",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, UseDirectiveStmt) and stmt.loc is not None:
            if stmt.tool_names:
                use_label = "> use: " + ", ".join(stmt.tool_names)
                use_detail = "agent tools"
            else:
                use_label = f"> use: {stmt.name}"
                use_detail = "agent profile or tool"
            children.append(
                lsp.DocumentSymbol(
                    name=use_label,
                    detail=use_detail,
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
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
        selection_range=(_span_to_range(_proc_name_span(proc)) if proc.loc else _empty_range()),
        children=children,
    )


def _profile_def_symbol(profile: ProfileDefStmt) -> lsp.DocumentSymbol:
    children: list[lsp.DocumentSymbol] = []
    for member in profile.model_members:
        if member.loc is None:
            continue
        value = member.plain_value or (
            f"`{member.backtick_code}`" if member.backtick_code is not None else ""
        )
        children.append(
            lsp.DocumentSymbol(
                name=f"> model: {value}",
                detail="agent model override",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.effort_members:
        if member.loc is None:
            continue
        value = member.plain_value or (
            f"`{member.backtick_code}`" if member.backtick_code is not None else ""
        )
        children.append(
            lsp.DocumentSymbol(
                name=f"> effort: {value}",
                detail="agent reasoning effort",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.system_members:
        if member.loc is None:
            continue
        value = _system_value_preview(member).strip("`")
        children.append(
            lsp.DocumentSymbol(
                name=f"> system: {value}",
                detail="agent instructions",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    if profile.tool_names and profile.loc is not None:
        children.append(
            lsp.DocumentSymbol(
                name="> use: " + ", ".join(profile.tool_names),
                detail="agent tools",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(profile.loc),
                selection_range=_span_to_range(profile.loc),
                children=[],
            )
        )
    return lsp.DocumentSymbol(
        name=f"> profile: {profile.name}",
        detail="agent profile",
        kind=lsp.SymbolKind.Interface,
        range=_span_to_range(profile.loc) if profile.loc else _empty_range(),
        selection_range=(_span_to_range(profile.loc) if profile.loc else _empty_range()),
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


def _python_code_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _emit_token_spans_for_range(
    *,
    src_lines: Sequence[str],
    emit: Callable[[int, int, int, int, int], None],
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    token_type: int,
    token_mods: int,
) -> None:
    """Emit one or more LSP semantic-token spans for a possibly multi-line range."""
    if start_line == end_line:
        emit(start_line, start_col, end_col - start_col, token_type, token_mods)
        return
    if start_line < len(src_lines):
        emit(
            start_line,
            start_col,
            max(0, len(src_lines[start_line]) - start_col),
            token_type,
            token_mods,
        )
    for line_idx in range(start_line + 1, end_line):
        if line_idx < len(src_lines):
            emit(line_idx, 0, len(src_lines[line_idx]), token_type, token_mods)
    if end_col > 0 and end_line < len(src_lines):
        emit(end_line, 0, end_col, token_type, token_mods)


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
        if length <= 0:  # pragma: no cover - tree-sitter captures are non-empty.
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
        _emit_token_spans_for_range(
            src_lines=src_lines,
            emit=emit,
            start_line=kedi_sl,
            start_col=kedi_sc,
            end_line=kedi_el,
            end_col=kedi_ec,
            token_type=ttype,
            token_mods=mods,
        )

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
        assert _KEDI_INJECTIONS_QUERY is not None
        cursor = _TSQueryCursor(_KEDI_INJECTIONS_QUERY)
        for _pat_idx, caps in cursor.matches(tree.root_node):
            for region_node in caps.get("injection.content", []):
                py_bytes = region_node.text
                if not py_bytes:  # pragma: no cover - injection captures are non-empty.
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
            while j < max_col:
                next_slot = paint[j]
                if next_slot is None or next_slot[1:] != (ttype, mods):
                    break
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

            dedented = "\n".join(dedent_block(_python_code_lines(text)))
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
