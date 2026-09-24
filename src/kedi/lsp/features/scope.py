from __future__ import annotations

import ast as _py_ast
import re
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

from lsprotocol import types as lsp

from kedi.lang.ast import (
    AssignmentStmt,
    AwaitStmt,
    ConditionalLoopStmt,
    IfStmt,
    LoopStmt,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    PythonBlock,
    RawInvokeStmt,
    ReturnStmt,
    ShowStmt,
    Span,
    TaskGroupStmt,
    TaskStmt,
    TemplateBlock,
    TemplateLine,
    TypeDef,
    TypeDefStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lsp.ast_helpers import binding_name_span
from kedi.lsp.ast_spans import name_token_span
from kedi.lsp.coordinates import current_source
from kedi.lsp.coordinates import position_inside as _position_inside
from kedi.lsp.features.symbols import (
    _is_identifier,
    _line_start_byte_for_program,
    _output_name_span,
    _program_line_text,
    _ValueDeclaration,
)
from kedi.lsp.module_surface import ImportedValueInfo


def _segments_of(stmt) -> list:
    """All segment children of a statement (``segments`` or ``expr``)."""
    if isinstance(stmt, TemplateBlock):
        segs: list = []
        for line in stmt.lines:
            segs.extend(line.segments)
        return segs
    if isinstance(stmt, TaskStmt):
        return _segments_of(stmt.template)
    if isinstance(stmt, TemplateLine):
        return list(stmt.segments)
    if isinstance(stmt, RawInvokeStmt):
        return list(stmt.segments)
    if isinstance(stmt, (ReturnStmt, ShowStmt)):
        return list(stmt.segments)
    if isinstance(stmt, (VariableInitStmt, AssignmentStmt)):
        return list(stmt.expr)
    if isinstance(stmt, (IfStmt, ConditionalLoopStmt)):
        return list(stmt.condition_segments or [])
    return []


def _control_body_statements(stmt) -> list:
    if isinstance(stmt, TaskGroupStmt):
        return [child for arm in stmt.arms for child in [arm.wait, *(arm.body or [])]]
    if isinstance(stmt, IfStmt):
        return [*stmt.body, *(stmt.else_body or [])]
    if isinstance(stmt, LoopStmt):
        return [*stmt.body, *(stmt.map_body or [])]
    if isinstance(stmt, ConditionalLoopStmt):
        return list(stmt.body)
    return []


def _iter_control_statements(statements):
    for stmt in statements:
        yield stmt
        yield from _iter_control_statements(_control_body_statements(stmt))


def _visible_control_statements(statements, pos):
    """Walk control flow without exposing sibling or expired processing scopes."""
    for statement in statements:
        yield statement
        if isinstance(statement, TaskGroupStmt):
            for arm in statement.arms:
                if arm.loc is not None and _position_inside(pos, arm.loc):
                    yield from _visible_control_statements([arm.wait, *(arm.body or [])], pos)
        else:
            yield from _visible_control_statements(_control_body_statements(statement), pos)


def _exported_name_at_position(
    prog: Program | ProgramWithValidation,
    pos: lsp.Position,
) -> Optional[tuple[str, Span]]:
    for export in prog.exports:
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
        line_start = _line_start_byte_for_program(prog, pos.line + 1)
        if line_start is None:
            continue
        start_byte = line_start + len(line[:name_index].encode("utf-8"))
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


def _position_in_proc_name(proc: ProcedureDef, pos: lsp.Position) -> bool:
    """True iff the cursor is exactly on the procedure's name token.

    The header `@<name>(...)` starts at column 1 in 1-based terms. The
    name occupies columns 2..2+len(name).
    """
    assert proc.loc is not None
    if pos.line + 1 != proc.loc.start_line:
        return False
    return _position_inside(pos, _proc_name_span(proc))


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
    source = current_source()
    if proc.loc is None or not proc.params or source is None:
        return []
    header_bytes = source.source.encode("utf-8")[proc.loc.start_byte :]
    header = header_bytes.split(b"\n", 1)[0].decode("utf-8")
    cursor = header.find("(") + 1
    declarations: List[Tuple[str, Span]] = []
    for name in proc.params:
        match = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", header[cursor:])
        if match is None:
            continue
        start_char = cursor + match.start()
        end_char = cursor + match.end()
        start_offset = len(header[:start_char].encode("utf-8"))
        end_offset = len(header[:end_char].encode("utf-8"))
        declarations.append(
            (
                name,
                Span(
                    start_byte=proc.loc.start_byte + start_offset,
                    end_byte=proc.loc.start_byte + end_offset,
                    start_line=proc.loc.start_line,
                    start_col=start_offset + 1,
                    end_line=proc.loc.start_line,
                    end_col=end_offset + 1,
                    source_path=proc.loc.source_path,
                    source_id=proc.loc.source_id,
                ),
            )
        )
        cursor = end_char
    return declarations


def _param_decl_span(proc: ProcedureDef, name: str) -> Optional[Span]:
    return next((span for param_name, span in _param_decls(proc) if param_name == name), None)


def _resolve_lexical_name(
    proc: Optional[ProcedureDef],
    ancestor_procs: tuple[ProcedureDef, ...],
    name: str,
) -> tuple[str, Span, ProcedureDef] | None:
    """Find the nearest Kedi-local declaration visible from a procedure body."""
    if proc is None:
        return None
    for candidate in reversed((*ancestor_procs, proc)):
        if name in candidate.params:
            declaration_span = _param_decl_span(candidate, name)
            if declaration_span is not None:
                return "parameter", declaration_span, candidate
        declaration_span = _find_local_declaration(candidate, name)
        if declaration_span is not None:
            return "local", declaration_span, candidate
    return None


def _direct_statement_declarations(
    stmt,
    name: str,
    proc: Optional[ProcedureDef],
) -> list[_ValueDeclaration]:
    declarations: list[_ValueDeclaration] = []
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.name == name:
        span = binding_name_span(stmt)
        if span is not None:
            declarations.append(
                _ValueDeclaration(
                    kind="local" if proc is not None else "module_value",
                    name=name,
                    declaration_span=span,
                    scope_proc=proc,
                )
            )
    if isinstance(stmt, (TaskStmt, AwaitStmt)) and stmt.binder == name:
        span = stmt.binder_loc
        if span is not None:
            declarations.append(
                _ValueDeclaration(
                    kind="local" if proc is not None else "module_value",
                    name=name,
                    declaration_span=span,
                    scope_proc=proc,
                )
            )
    for segment in _segments_of(stmt):
        if (
            not isinstance(stmt, TaskStmt)
            and isinstance(segment, OutputSegment)
            and segment.name == name
        ):
            span = _output_name_span(segment)
            if span is not None:
                declarations.append(
                    _ValueDeclaration(
                        kind="output_decl",
                        name=name,
                        declaration_span=span,
                        scope_proc=proc,
                    )
                )
    return declarations


def _statements_contain_byte(statements: Sequence[object], byte: int) -> bool:
    for stmt in statements:
        loc = getattr(stmt, "loc", None)
        if isinstance(loc, Span) and loc.start_byte <= byte < loc.end_byte:
            return True
    return False


def _declaration_in_scope_path(
    statements: Sequence[object],
    name: str,
    reference_byte: int,
    proc: Optional[ProcedureDef],
) -> _ValueDeclaration | None:
    """Resolve ``name`` in the lexical block containing ``reference_byte``.

    Control-flow bodies are child scopes. A loop body and its attached map
    share one iteration scope, with the binder acting as that scope's first
    declaration.
    """

    latest: _ValueDeclaration | None = None
    for stmt in statements:
        loc = getattr(stmt, "loc", None)
        if loc is None:
            continue
        if loc.start_byte >= reference_byte:
            break
        if loc.start_byte <= reference_byte < loc.end_byte:
            if isinstance(stmt, TaskGroupStmt):
                for arm in stmt.arms:
                    if (
                        arm.loc is not None
                        and arm.loc.start_byte <= reference_byte < arm.loc.end_byte
                    ):
                        return (
                            _declaration_in_scope_path(
                                [arm.wait, *(arm.body or [])], name, reference_byte, proc
                            )
                            or latest
                        )
                return latest
            if isinstance(stmt, IfStmt):
                for branch in (stmt.body, stmt.else_body or []):
                    if _statements_contain_byte(branch, reference_byte):
                        return (
                            _declaration_in_scope_path(branch, name, reference_byte, proc) or latest
                        )
                return latest
            if isinstance(stmt, LoopStmt):
                iteration_body = [*stmt.body, *(stmt.map_body or [])]
                if not _statements_contain_byte(iteration_body, reference_byte):
                    return latest
                nested = _declaration_in_scope_path(
                    iteration_body,
                    name,
                    reference_byte,
                    proc,
                )
                if nested is not None:
                    return nested
                if stmt.binder == name and stmt.binder_loc is not None:
                    return _ValueDeclaration(
                        kind="loop_binder",
                        name=name,
                        declaration_span=stmt.binder_loc,
                        scope_proc=proc,
                    )
                return latest
            if isinstance(stmt, ConditionalLoopStmt):
                if _statements_contain_byte(stmt.body, reference_byte):
                    return (
                        _declaration_in_scope_path(
                            stmt.body,
                            name,
                            reference_byte,
                            proc,
                        )
                        or latest
                    )
                return latest
            return latest
        declarations = _direct_statement_declarations(stmt, name, proc)
        if declarations:
            latest = max(declarations, key=lambda item: item.declaration_span.start_byte)
    return latest


def _resolve_value_declaration(
    prog: Program | ProgramWithValidation,
    proc: Optional[ProcedureDef],
    ancestor_procs: tuple[ProcedureDef, ...],
    name: str,
    reference_span: Span,
) -> _ValueDeclaration | None:
    reference_byte = reference_span.start_byte
    if proc is not None:
        declaration = _declaration_in_scope_path(proc.body, name, reference_byte, proc)
        if declaration is not None:
            return declaration
        parameter_span = _param_decl_span(proc, name) if name in proc.params else None
        if parameter_span is not None:
            return _ValueDeclaration("parameter", name, parameter_span, proc)

        for ancestor in reversed(ancestor_procs):
            declaration = _declaration_in_scope_path(
                ancestor.body,
                name,
                reference_byte,
                ancestor,
            )
            if declaration is not None:
                return declaration
            parameter_span = _param_decl_span(ancestor, name) if name in ancestor.params else None
            if parameter_span is not None:
                return _ValueDeclaration("parameter", name, parameter_span, ancestor)

        top_level_reference = (
            max(
                (
                    loc.end_byte
                    for stmt in prog.toplevel_body
                    if isinstance((loc := getattr(stmt, "loc", None)), Span)
                ),
                default=reference_byte,
            )
            + 1
        )
        return _declaration_in_scope_path(
            prog.toplevel_body,
            name,
            top_level_reference,
            None,
        )

    return _declaration_in_scope_path(
        prog.toplevel_body,
        name,
        reference_byte,
        None,
    )


def _is_imported_python_class(value: ImportedValueInfo | None) -> bool:
    return (
        value is not None
        and value.python_stub is not None
        and value.python_stub.startswith("class ")
    )


def _find_local_declaration(proc: Optional[ProcedureDef], name: str) -> Optional[Span]:
    """Find an output/assign-target named ``name`` in a procedure body."""
    if proc is None:
        return None
    for stmt in _iter_control_statements(proc.body):
        if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.name == name:
            return binding_name_span(stmt)
        for seg in _segments_of(stmt):
            if isinstance(seg, OutputSegment) and seg.name == name:
                return _output_name_span(seg)
    return None


def _nested_procedures(proc: ProcedureDef, pos: lsp.Position | None = None) -> list[ProcedureDef]:
    statements = (
        _iter_control_statements(proc.body)
        if pos is None
        else _visible_control_statements(proc.body, pos)
    )
    return [
        stmt.proc for stmt in statements if isinstance(stmt, ProcDefStmt) and stmt.proc is not None
    ]


def _direct_type_defs(proc: ProcedureDef, pos: lsp.Position | None = None) -> list[TypeDef]:
    statements = (
        _iter_control_statements(proc.body)
        if pos is None
        else _visible_control_statements(proc.body, pos)
    )
    return [
        stmt.typedef
        for stmt in statements
        if isinstance(stmt, TypeDefStmt) and stmt.typedef is not None
    ]


def _all_type_defs(
    prog: Program | ProgramWithValidation, pos: lsp.Position | None = None
) -> list[TypeDef]:
    type_defs = list(prog.type_defs)

    def visit(proc: ProcedureDef) -> None:
        type_defs.extend(_direct_type_defs(proc, pos))
        for child in _nested_procedures(proc, pos):
            visit(child)

    for proc in prog.procedures:
        visit(proc)
    statements = (
        _iter_control_statements(prog.toplevel_body)
        if pos is None
        else _visible_control_statements(prog.toplevel_body, pos)
    )
    for stmt in statements:
        if isinstance(stmt, TypeDefStmt) and stmt.typedef is not None:
            type_defs.append(stmt.typedef)
        elif isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            visit(stmt.proc)
    return type_defs


def _type_def_for_declaration(
    prog: Program | ProgramWithValidation,
    declaration_span: Optional[Span],
) -> Optional[TypeDef]:
    if declaration_span is None:
        return None
    return next(
        (typedef for typedef in _all_type_defs(prog) if typedef.loc == declaration_span),
        None,
    )


def _procedure_for_declaration(
    prog: Program | ProgramWithValidation,
    declaration_span: Optional[Span],
) -> Optional[ProcedureDef]:
    if declaration_span is None:
        return None
    return next(
        (proc for proc in _all_procedures(prog) if proc.loc == declaration_span),
        None,
    )


def _all_procedures(
    prog: Union[Program, ProgramWithValidation], pos: lsp.Position | None = None
) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        for child in _nested_procedures(proc, pos):
            visit(child)

    for proc in prog.procedures:
        visit(proc)
    statements = (
        _iter_control_statements(prog.toplevel_body)
        if pos is None
        else _visible_control_statements(prog.toplevel_body, pos)
    )
    for stmt in statements:
        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            visit(stmt.proc)
    return out


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


def _procedure_scope_entries(
    prog: Union[Program, ProgramWithValidation],
) -> list[tuple[ProcedureDef, tuple[ProcedureDef, ...]]]:
    entries: list[tuple[ProcedureDef, tuple[ProcedureDef, ...]]] = []

    def visit(proc: ProcedureDef, ancestors: tuple[ProcedureDef, ...]) -> None:
        entries.append((proc, ancestors))
        for child in _nested_procedures(proc):
            visit(child, ancestors + (proc,))

    for proc in prog.procedures:
        visit(proc, ())
    return entries
