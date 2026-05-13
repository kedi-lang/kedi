"""LSP feature implementations for the Kedi language server.

Each function here turns a Kedi AST (or raw source) plus an LSP
position into the LSP response shape (``Diagnostic``, ``Hover``,
``Location``). The split keeps :mod:`kedi.lsp.server` thin — it owns
the protocol surface and dispatches to functions in this module.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from lsprotocol import types as lsp

from ..lang.ast import (
    AssignStmt,
    CallSegment,
    InputSegment,
    OutputSegment,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    Span,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypePython,
    TypeRef,
    TypeUnion,
)
from ..lang.parser import KediDiagnostic, KediParseError, parse_program


# ---------------------------------------------------------------------
# Source ↔ position helpers
# ---------------------------------------------------------------------


def _span_to_range(span: Span) -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=span.start_line - 1, character=span.start_col - 1),
        end=lsp.Position(line=span.end_line - 1, character=max(span.end_col - 1, 0)),
    )


def _position_inside(pos: lsp.Position, span: Span) -> bool:
    """Return True if ``pos`` (0-based LSP) falls inside ``span`` (1-based)."""
    if pos.line + 1 < span.start_line or pos.line + 1 > span.end_line:
        return False
    if pos.line + 1 == span.start_line and pos.character + 1 < span.start_col:
        return False
    if pos.line + 1 == span.end_line and pos.character + 1 >= span.end_col:
        return False
    return True


def _byte_offset_of(text: str, pos: lsp.Position) -> int:
    """Translate a 0-based LSP ``Position`` to a UTF-8 byte offset."""
    lines = text.split("\n")
    if pos.line >= len(lines):
        return len(text.encode("utf-8"))
    pre = "\n".join(lines[: pos.line])
    if pre:
        pre += "\n"
    line_prefix = lines[pos.line][: pos.character]
    return len((pre + line_prefix).encode("utf-8"))


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def compute_diagnostics(source: str, source_path: Optional[str]) -> List[lsp.Diagnostic]:
    """Parse ``source`` and translate any ``KediDiagnostic`` to LSP form."""
    try:
        parse_program(source, source_path=source_path)
        return []
    except KediParseError as exc:
        return [_diag_to_lsp(d) for d in exc.diagnostics]


def _diag_to_lsp(d: KediDiagnostic) -> lsp.Diagnostic:
    severity = (
        lsp.DiagnosticSeverity.Error
        if d.severity == "error"
        else lsp.DiagnosticSeverity.Warning
    )
    return lsp.Diagnostic(
        range=_span_to_range(d.span),
        severity=severity,
        code=d.code,
        source="kedi",
        message=d.message,
    )


# ---------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------


def compute_hover(
    source: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Hover]:
    """Best-effort hover description for the symbol under the cursor.

    Walks the AST top-down, locating the smallest node whose ``Span``
    contains ``pos`` and rendering an appropriate short summary. If
    the parse fails, returns ``None`` (diagnostics will tell the user
    what's wrong).
    """
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None

    proc_by_name = {p.name: p for p in prog.procedures}
    type_by_name = {t.name: t for t in prog.type_defs}

    summary = _hover_for_position(prog, pos, proc_by_name, type_by_name)
    if summary is None:
        return None
    text, range_ = summary
    return lsp.Hover(
        contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text),
        range=range_,
    )


def _hover_for_position(
    prog: Union[Program, ProgramWithValidation],
    pos: lsp.Position,
    proc_by_name: dict[str, ProcedureDef],
    type_by_name: dict[str, TypeDef],
) -> Optional[Tuple[str, lsp.Range]]:
    """Locate the cursor in the AST and return (markdown, range)."""
    for proc in prog.procedures:
        result = _hover_in_procedure(proc, pos, proc_by_name, type_by_name)
        if result is not None:
            return result
    # Top-level body, return.
    for stmt in prog.toplevel_body:
        result = _hover_in_stmt(stmt, pos, proc_by_name, type_by_name)
        if result is not None:
            return result
    if prog.toplevel_return is not None:
        result = _hover_in_stmt(prog.toplevel_return, pos, proc_by_name, type_by_name)
        if result is not None:
            return result
    return None


def _hover_in_procedure(
    proc: ProcedureDef,
    pos: lsp.Position,
    proc_by_name: dict[str, ProcedureDef],
    type_by_name: dict[str, TypeDef],
) -> Optional[Tuple[str, lsp.Range]]:
    if proc.loc is None or not _position_inside(pos, proc.loc):
        return None
    for stmt in proc.body:
        r = _hover_in_stmt(stmt, pos, proc_by_name, type_by_name)
        if r is not None:
            return r
    # Cursor is inside the procedure header (name / params / return type)
    # but not on a specific sub-node — show the procedure signature.
    return _proc_summary(proc, proc.loc)


def _proc_summary(proc: ProcedureDef, range_span: Span) -> Tuple[str, lsp.Range]:
    parts = [f"**@{proc.name}**("]
    if proc.params:
        annos = proc.param_type_annotations or {}
        items = []
        for p in proc.params:
            anno = annos.get(p)
            items.append(f"{p}: {anno}" if anno else p)
        parts.append(", ".join(items))
    parts.append(")")
    if proc.return_type_annotation:
        parts.append(f" → {proc.return_type_annotation}")
    return "\n".join([" ".join(parts), "", "Kedi procedure."]), _span_to_range(range_span)


def _hover_in_stmt(
    stmt, pos, proc_by_name, type_by_name
) -> Optional[Tuple[str, lsp.Range]]:
    if not hasattr(stmt, "loc") or stmt.loc is None or not _position_inside(pos, stmt.loc):
        return None
    for seg in _segments_of(stmt):
        r = _hover_in_segment(seg, pos, proc_by_name, type_by_name)
        if r is not None:
            return r
    return f"`{type(stmt).__name__}`", _span_to_range(stmt.loc)


def _segments_of(stmt) -> list:
    """All segment children of a statement (``segments`` or ``expr``)."""
    return list(getattr(stmt, "segments", []) or []) + list(getattr(stmt, "expr", []) or [])


def _hover_in_segment(
    seg, pos, proc_by_name, type_by_name
) -> Optional[Tuple[str, lsp.Range]]:
    if not hasattr(seg, "loc") or seg.loc is None or not _position_inside(pos, seg.loc):
        return None
    if isinstance(seg, InputSegment):
        return (
            f"**Input** `{seg.name}` — value substituted from a parameter or assignment.",
            _span_to_range(seg.loc),
        )
    if isinstance(seg, OutputSegment):
        anno = seg.type_annotation or "str"
        return (
            f"**Output** `{seg.name}: {anno}` — LLM-filled placeholder.",
            _span_to_range(seg.loc),
        )
    if isinstance(seg, CallSegment):
        target = proc_by_name.get(seg.name)
        if target is not None:
            return _proc_summary(target, seg.loc)
        return (
            f"**Call** `{seg.name}(...)` — procedure not found in this module.",
            _span_to_range(seg.loc),
        )
    return None


# ---------------------------------------------------------------------
# Go-to-definition
# ---------------------------------------------------------------------


def compute_definition(
    source: str, uri: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Location]:
    """Return the definition site of the symbol at ``pos``.

    Currently resolves procedure-call names to their ``ProcedureDef``
    and type identifiers (in type_expr / output annotations) to the
    matching ``TypeDef``. Variable / parameter references are deferred.
    """
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None

    proc_by_name = {p.name: p for p in prog.procedures}
    type_by_name = {t.name: t for t in prog.type_defs}

    target_span = _find_definition(prog, pos, proc_by_name, type_by_name)
    if target_span is None:
        return None
    return lsp.Location(uri=uri, range=_span_to_range(target_span))


def _find_definition(
    prog,
    pos: lsp.Position,
    proc_by_name: dict[str, ProcedureDef],
    type_by_name: dict[str, TypeDef],
) -> Optional[Span]:
    for proc in prog.procedures:
        if proc.loc is None or not _position_inside(pos, proc.loc):
            continue
        for stmt in proc.body:
            t = _def_in_stmt(stmt, pos, proc_by_name, type_by_name)
            if t is not None:
                return t
    for stmt in prog.toplevel_body:
        t = _def_in_stmt(stmt, pos, proc_by_name, type_by_name)
        if t is not None:
            return t
    if prog.toplevel_return is not None:
        t = _def_in_stmt(prog.toplevel_return, pos, proc_by_name, type_by_name)
        if t is not None:
            return t
    return None


def _def_in_stmt(stmt, pos, proc_by_name, type_by_name) -> Optional[Span]:
    if not hasattr(stmt, "loc") or stmt.loc is None or not _position_inside(pos, stmt.loc):
        return None
    for seg in _segments_of(stmt):
        if not hasattr(seg, "loc") or seg.loc is None or not _position_inside(pos, seg.loc):
            continue
        if isinstance(seg, CallSegment):
            target = proc_by_name.get(seg.name)
            if target is not None and target.loc is not None:
                return target.loc
        if isinstance(seg, OutputSegment) and seg.type_expr is not None:
            t = _def_for_type_expr(seg.type_expr, pos, type_by_name)
            if t is not None:
                return t
    if isinstance(stmt, AssignStmt) and stmt.type_expr is not None:
        t = _def_for_type_expr(stmt.type_expr, pos, type_by_name)
        if t is not None:
            return t
    return None


def _def_for_type_expr(
    expr: Optional[TypeExpr],
    pos: lsp.Position,
    type_by_name: dict[str, TypeDef],
) -> Optional[Span]:
    if expr is None or expr.loc is None or not _position_inside(pos, expr.loc):
        return None
    if isinstance(expr, TypeRef):
        target = type_by_name.get(expr.name)
        if target is not None and target.loc is not None:
            return target.loc
    if isinstance(expr, TypeApply):
        target = type_by_name.get(expr.name)
        if target is not None and target.loc is not None:
            return target.loc
        for arg in expr.args:
            t = _def_for_type_expr(arg, pos, type_by_name)
            if t is not None:
                return t
    if isinstance(expr, TypeUnion):
        for child in (expr.left, expr.right):
            t = _def_for_type_expr(child, pos, type_by_name)
            if t is not None:
                return t
    if isinstance(expr, TypePython):
        # Backtick-wrapped Python is opaque to the LSP for now.
        return None
    return None
