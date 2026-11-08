from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Union

from lsprotocol import types as lsp

from kedi.lang.ast import (
    AssignmentBlockStmt,
    AssignmentStmt,
    AwaitStmt,
    CallArg,
    CallSegment,
    ConditionalLoopStmt,
    IfStmt,
    InputSegment,
    LoopStmt,
    OutputSegment,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    SendStmt,
    Span,
    TaskGroupStmt,
    TaskStmt,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeRef,
    TypeUnion,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
    iter_profile_bodies,
)
from kedi.lang.call_arg_access import iter_call_arg_nested
from kedi.lang.parser import KediParseError
from kedi.lsp.ast_helpers import binding_name_span
from kedi.lsp.ast_spans import name_token_span
from kedi.lsp.coordinates import current_source, with_source_coordinates
from kedi.lsp.coordinates import span_to_range as _span_to_range
from kedi.lsp.features.resolution import _resolve_symbol_at
from kedi.lsp.features.scope import (
    _all_procedures,
    _all_type_defs,
    _control_body_statements,
    _iter_control_statements,
    _proc_name_span,
    _procedure_for_declaration,
    _resolve_lexical_name,
    _segments_of,
)
from kedi.lsp.features.symbols import (
    SymbolRef,
    _is_identifier,
    _location_for_span,
    _profile_use_entry_spans,
    _use_single_target_span,
    _use_tool_entry_spans,
    _validation_procedure_span,
    _validation_suites,
)
from kedi.lsp.features.type_support import _known_lsp_type_origin
from kedi.lsp.parsing import parse_program


@with_source_coordinates(2)
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
    return _location_for_span(uri, source_path, sym.declaration_span)


@with_source_coordinates(2)
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
    return [_location_for_span(uri, source_path, span) for span in spans]


def _find_all_references(
    prog: Union[Program, ProgramWithValidation],
    sym: SymbolRef,
    *,
    include_declaration: bool,
) -> List[Span]:
    out: List[Span] = []
    if include_declaration and sym.declaration_span is not None:
        out.append(_declaration_identifier_span(prog, sym))

    if sym.kind in ("procedure", "agent_tool"):
        procedure_candidates: List[Span] = []
        for profile in prog.profile_defs.values():
            for tool_name, tool_span in _profile_use_entry_spans(prog, profile):
                if tool_name == sym.name:
                    procedure_candidates.append(tool_span)
        for suite in _validation_suites(prog):
            if suite.procedure_name == sym.name:
                suite_span = _validation_procedure_span(prog, suite)
                if suite_span is not None:
                    procedure_candidates.append(suite_span)
        for p in _all_procedures(prog):
            for stmt in _iter_control_statements(p.body):
                _collect_use_refs(prog, stmt, sym.name, procedure_candidates)
                for seg in _segments_of(stmt):
                    _collect_call_refs(seg, sym.name, procedure_candidates)
        for stmt in _iter_control_statements(prog.ordered_toplevel or prog.toplevel_body):
            _collect_use_refs(prog, stmt, sym.name, procedure_candidates)
            for seg in _segments_of(stmt):
                _collect_call_refs(seg, sym.name, procedure_candidates)
        if prog.toplevel_return is not None:
            for seg in _segments_of(prog.toplevel_return):
                _collect_call_refs(seg, sym.name, procedure_candidates)
        out.extend(
            candidate
            for candidate in procedure_candidates
            if _reference_resolves_to(prog, candidate, sym)
        )
    elif sym.kind == "type":
        type_candidates: List[Span] = []
        for td in _all_type_defs(prog):
            _collect_type_refs_in_type_def(td, sym.name, type_candidates)
        for profile in prog.profile_defs.values():
            for member in (
                entry for body in iter_profile_bodies(profile) for entry in body.output_members
            ):
                _collect_type_refs_in_expr(member.type_expr, sym.name, type_candidates)
        for stmt in prog.toplevel_body:
            _collect_type_refs_in_stmt(stmt, sym.name, type_candidates)
        for p in _all_procedures(prog):
            if p.return_type_expr is not None:
                _collect_type_refs_in_expr(p.return_type_expr, sym.name, type_candidates)
            for v in (p.param_type_exprs or {}).values():
                if v is not None:
                    _collect_type_refs_in_expr(v, sym.name, type_candidates)
            for stmt in p.body:
                _collect_type_refs_in_stmt(stmt, sym.name, type_candidates)
        out.extend(
            candidate
            for candidate in type_candidates
            if _reference_resolves_to(prog, candidate, sym)
        )
    elif sym.kind == "agent_profile":
        for profile in prog.profile_defs.values():
            for target_name, target_span in _profile_use_entry_spans(prog, profile):
                if target_name == sym.name:
                    out.append(target_span)
            for member in (
                entry for body in iter_profile_bodies(profile) for entry in body.subagent_members
            ):
                if member.name == sym.name and member.loc is not None:
                    out.append(member.loc)
        for stmt in prog.ordered_toplevel:
            if isinstance(stmt, UseDirectiveStmt) and stmt.name == sym.name and not stmt.tool_names:
                target_span = _use_single_target_span(prog, stmt)
                if target_span is not None:
                    out.append(target_span)
        for stmt in _iter_control_statements(prog.toplevel_body):
            if isinstance(stmt, TaskStmt) and stmt.subagent == sym.name:
                if stmt.subagent_loc is not None:
                    out.append(stmt.subagent_loc)
        for procedure in _all_procedures(prog):
            for stmt in _iter_control_statements(procedure.body):
                if isinstance(stmt, TaskStmt) and stmt.subagent == sym.name:
                    if stmt.subagent_loc is not None:
                        out.append(stmt.subagent_loc)
    elif sym.kind in {
        "parameter",
        "param_decl",
        "local",
        "output_decl",
        "loop_binder",
        "module_value",
        "imported_value",
    }:
        value_candidates: list[Span] = []
        _collect_value_reference_candidates(prog.toplevel_body, sym.name, value_candidates)
        if prog.toplevel_return is not None:
            for segment in _segments_of(prog.toplevel_return):
                _collect_input_refs(segment, sym.name, value_candidates)
        for procedure in _all_procedures(prog):
            _collect_value_reference_candidates(procedure.body, sym.name, value_candidates)
        out.extend(
            candidate
            for candidate in value_candidates
            if _reference_resolves_to(prog, candidate, sym)
        )
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


def _declaration_identifier_span(
    prog: Program | ProgramWithValidation,
    sym: SymbolRef,
) -> Span:
    assert sym.declaration_span is not None
    if current_source() is None:
        return sym.declaration_span
    if sym.kind in ("procedure", "agent_tool"):
        proc = _procedure_for_declaration(prog, sym.declaration_span)
        if proc is not None:
            return _proc_name_span(proc)
        return name_token_span(sym.declaration_span, sym.name, prefix_len=1)
    if sym.kind == "type":
        return name_token_span(sym.declaration_span, sym.name, prefix_len=1)
    if sym.kind == "type_field":
        return name_token_span(sym.declaration_span, sym.name, prefix_len=0)
    return sym.declaration_span


def _reference_resolves_to(
    prog: Program | ProgramWithValidation,
    candidate: Span,
    target: SymbolRef,
) -> bool:
    if current_source() is None:
        return True
    resolved = _resolve_symbol_at(
        prog,
        lsp.Position(line=candidate.start_line - 1, character=candidate.start_col - 1),
    )
    if resolved is None:  # pragma: no cover - collectors only emit resolvable syntax spans
        return False
    procedure_kinds = {"procedure", "agent_tool"}
    parameter_kinds = {"parameter", "param_decl"}
    same_kind = (
        resolved.kind == target.kind
        or resolved.kind in procedure_kinds
        and target.kind in procedure_kinds
        or resolved.kind in parameter_kinds
        and target.kind in parameter_kinds
    )
    return same_kind and resolved.declaration_span == target.declaration_span


def _collect_use_refs(
    prog: Program | ProgramWithValidation, stmt: object, name: str, out: list[Span]
) -> None:
    if not isinstance(stmt, UseDirectiveStmt):
        return
    if stmt.tool_names:
        out.extend(span for target, span in _use_tool_entry_spans(prog, stmt) if target == name)
    elif stmt.name == name:
        span = _use_single_target_span(prog, stmt)
        if span is not None:
            out.append(span)


def _collect_literal_arg_refs(arg: CallArg, name: str, out: List[Span], collector) -> None:
    for nested in iter_call_arg_nested(arg):
        collector(nested, name, out)


def _collect_call_refs(seg, name: str, out: List[Span]) -> None:
    if isinstance(seg, CallSegment):
        if seg.name == name and seg.loc is not None:
            out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
        for arg in seg.args:
            _collect_literal_arg_refs(arg, name, out, _collect_call_refs)


def _collect_input_refs(seg, name: str, out: List[Span]) -> None:
    """Collect matching template substitutions, including nested call arguments."""
    if isinstance(seg, InputSegment) and seg.name == name and seg.loc is not None:
        out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
    elif isinstance(seg, CallSegment):
        for arg in seg.args:
            _collect_literal_arg_refs(arg, name, out, _collect_input_refs)


def _collect_value_reference_candidates(
    statements: Sequence[object],
    name: str,
    out: List[Span],
) -> None:
    for stmt in _iter_control_statements(statements):
        if isinstance(stmt, (AwaitStmt, SendStmt)) and stmt.handle == name:
            if stmt.handle_loc is not None:
                out.append(stmt.handle_loc)
        if isinstance(stmt, (AssignmentStmt, AssignmentBlockStmt)) and stmt.name == name:
            target_span = binding_name_span(stmt)
            if target_span is not None:
                out.append(target_span)
        for segment in _segments_of(stmt):
            _collect_input_refs(segment, name, out)


def _collect_lexical_input_refs(
    seg,
    name: str,
    proc: ProcedureDef,
    ancestors: tuple[ProcedureDef, ...],
    owner: ProcedureDef,
    declaration_span: Optional[Span],
    out: List[Span],
) -> None:
    if isinstance(seg, InputSegment) and seg.name == name and seg.loc is not None:
        resolved = _resolve_lexical_name(proc, ancestors, name)
        if resolved is not None and resolved[2] is owner and resolved[1] == declaration_span:
            out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
    elif isinstance(seg, CallSegment):
        for arg in seg.args:
            for nested in iter_call_arg_nested(arg):
                _collect_lexical_input_refs(
                    nested,
                    name,
                    proc,
                    ancestors,
                    owner,
                    declaration_span,
                    out,
                )


def _collect_loop_binder_refs(
    prog: Program | ProgramWithValidation,
    target: SymbolRef,
    out: List[Span],
) -> None:
    def visit(statements, active: bool = False) -> None:
        for stmt in statements:
            if isinstance(stmt, LoopStmt):
                is_target = stmt.binder_loc == target.declaration_span
                if is_target:
                    visit(stmt.body, True)
                    visit(stmt.map_body or [], True)
                elif not (active and stmt.binder == target.name):
                    visit(stmt.body, active)
                    visit(stmt.map_body or [], active)
                continue

            if active:
                for segment in _segments_of(stmt):
                    _collect_input_refs(segment, target.name, out)
            if isinstance(stmt, IfStmt):
                visit(stmt.body, active)
                visit(stmt.else_body or [], active)
            elif isinstance(stmt, ConditionalLoopStmt):
                visit(stmt.body, active)
            elif isinstance(stmt, TaskGroupStmt):
                for arm in stmt.arms:
                    visit(arm.body or [], active and arm.wait.binder != target.name)

    visit(prog.toplevel_body)
    for proc in _all_procedures(prog):
        visit(proc.body)


def _collect_type_refs_in_stmt(stmt, name: str, out: List[Span]) -> None:
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.type_expr is not None:
        _collect_type_refs_in_expr(stmt.type_expr, name, out)
    for seg in _segments_of(stmt):
        if isinstance(seg, OutputSegment) and seg.type_expr is not None:
            _collect_type_refs_in_expr(seg.type_expr, name, out)
    for child in _control_body_statements(stmt):
        _collect_type_refs_in_stmt(child, name, out)


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
            out.append(name_token_span(expr.loc, expr.name, prefix_len=0))
        for arg in expr.args:
            _collect_type_refs_in_expr(arg, name, out)
    elif isinstance(expr, TypeUnion):
        _collect_type_refs_in_expr(expr.left, name, out)
        _collect_type_refs_in_expr(expr.right, name, out)


@with_source_coordinates(1)
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
        "agent_tool",
        "type",
        "type_field",
        "parameter",
        "param_decl",
        "local",
        "loop_binder",
        "output_decl",
        "module_value",
        "imported_value",
    ):
        return None
    return _span_to_range(sym.cursor_span)


@with_source_coordinates(2)
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
    return lsp.WorkspaceEdit(changes={uri: edits})
