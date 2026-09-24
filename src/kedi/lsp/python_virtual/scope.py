from __future__ import annotations

import ast as py_ast
from typing import Any, Iterable, Optional

from kedi.lang.ast import (
    AssignmentBlockStmt,
    AssignmentStmt,
    AwaitStmt,
    ConditionalLoopStmt,
    IfStmt,
    LoopStmt,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    ProgramWithValidation,
    PythonBlock,
    RawInvokeStmt,
    ReturnStmt,
    ShowStmt,
    Span,
    Stmt,
    TaskGroupStmt,
    TaskStmt,
    TemplateBlock,
    TemplateLine,
    TypeDef,
    TypeExpr,
    TypePython,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.type_render import type_expr_to_python
from kedi.lsp.module_surface import ImportedSurface
from kedi.lsp.python_virtual.builder import (
    PythonRegion,
    Range,
    ScopeName,
    VirtualBuilder,
    _line_range,
    _span_range,
)


def _find_focused_procedure(
    procedures: list[ProcedureDef], focus_line: Optional[int]
) -> Optional[ProcedureDef]:
    if focus_line is None:
        return None
    source_line = focus_line + 1
    return _find_deepest_procedure_at_line(procedures, source_line)


def _find_deepest_procedure_at_line(
    procedures: Iterable[ProcedureDef], source_line: int
) -> Optional[ProcedureDef]:
    for proc in procedures:
        if proc.loc is None or not _span_contains_source_line(proc.loc, source_line):
            continue
        nested = _find_deepest_procedure_at_line(_nested_procedures(proc), source_line)
        return nested or proc
    return None


def _span_contains_source_line(span: Span, source_line: int) -> bool:
    """Whether a 1-based source line intersects an end-exclusive source span.

    Tree-sitter spans ending after a newline have ``end_col == 1`` on the
    following physical line. That line is outside the node. Treating the end
    line as inclusive made a top-level inline return immediately after a
    procedure disappear from its focused virtual Python document.
    """
    if source_line < span.start_line or source_line > span.end_line:
        return False
    if source_line != span.end_line:
        return True
    return span.end_col > 1


def _top_level_procedure_for(
    procedures: Iterable[ProcedureDef], target: Optional[ProcedureDef]
) -> Optional[ProcedureDef]:
    if target is None:
        return None
    for proc in procedures:
        if proc is target or _procedure_contains(proc, target):
            return proc
    return None


def _procedure_contains(proc: ProcedureDef, target: ProcedureDef) -> bool:
    return any(
        child is target or _procedure_contains(child, target) for child in _nested_procedures(proc)
    )


def _split_leading_toplevel_python(
    statements: list[Stmt],
) -> tuple[list[PythonBlock], list[Stmt]]:
    leading: list[PythonBlock] = []
    index = 0
    while index < len(statements) and isinstance(statements[index], PythonBlock):
        statement = statements[index]
        assert isinstance(statement, PythonBlock)
        leading.append(statement)
        index += 1
    return leading, statements[index:]


def _should_emit_full_module_python(
    region: Optional[PythonRegion], focus_line: Optional[int]
) -> bool:
    if focus_line is None:
        return True
    if region is None:
        return False
    return region.range["start"]["line"] <= focus_line < region.range["end"]["line"]


def _unparse_or_any(expr: Optional[py_ast.AST]) -> str:
    if expr is None:
        return "Any"
    try:
        return py_ast.unparse(expr)
    except Exception:
        return "Any"


def _nested_procedures(proc: ProcedureDef) -> list[ProcedureDef]:
    return [
        stmt.proc
        for stmt in _iter_control_statements(proc.body)
        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None
    ]


def _iter_control_statements(statements: Iterable[Stmt]) -> Iterable[Stmt]:
    for stmt in statements:
        yield stmt
        if isinstance(stmt, IfStmt):
            yield from _iter_control_statements(stmt.body)
            if stmt.else_body is not None:
                yield from _iter_control_statements(stmt.else_body)
        elif isinstance(stmt, LoopStmt):
            yield from _iter_control_statements(stmt.body)
            yield from _iter_control_statements(stmt.map_body or [])
        elif isinstance(stmt, ConditionalLoopStmt):
            yield from _iter_control_statements(stmt.body)
        elif isinstance(stmt, TaskGroupStmt):
            for arm in stmt.arms:
                yield from _iter_control_statements([arm.wait, *(arm.body or [])])


def _all_procedures(procedures: Iterable[ProcedureDef]) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        for child in _nested_procedures(proc):
            visit(child)

    for proc in procedures:
        visit(proc)
    return out


def _scope_params(scope: dict[str, ScopeName]) -> str:
    return ", ".join(f"{name}: {info.typ}" for name, info in scope.items())


def _collect_toplevel_scope(statements: Iterable[Stmt]) -> dict[str, ScopeName]:
    scope: dict[str, ScopeName] = {}
    for stmt in statements:
        if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)):
            scope[stmt.name] = ScopeName(
                typ=_type_to_python(stmt.type_expr, stmt.type_annotation),
                source_range=_binding_name_range(stmt),
            )
        elif isinstance(stmt, RawInvokeStmt):
            scope[stmt.name] = ScopeName(
                typ="str",
                source_range=_raw_invoke_name_range(stmt),
            )
        elif isinstance(stmt, TaskStmt):
            scope[stmt.binder] = ScopeName(
                typ="SubagentTask",
                source_range=_span_range(stmt.binder_loc) if stmt.binder_loc else None,
            )
        elif isinstance(stmt, AwaitStmt) and stmt.binder:
            scope[stmt.binder] = ScopeName(
                typ="SubagentResult",
                source_range=_span_range(stmt.binder_loc) if stmt.binder_loc else None,
            )
        for segment in _stmt_segments(stmt):
            if isinstance(segment, OutputSegment) and not isinstance(stmt, TaskStmt):
                scope[segment.name] = ScopeName(
                    typ=_type_to_python(segment.type_expr, segment.type_annotation),
                    source_range=_output_name_range(segment),
                )
    return scope


def _add_scope_param_symbols(
    builder: VirtualBuilder,
    scope: dict[str, ScopeName],
    line: int,
    text: str,
) -> None:
    search_from = text.find("(") + 1
    if search_from <= 0:
        return
    for name, info in scope.items():
        start = text.find(name, search_from)
        if start == -1:
            continue
        builder.add_symbol(
            kind="parameter",
            name=name,
            source_range=info.source_range,
            virtual_range=_line_range(line, start, name),
        )
        search_from = start + len(name)


def _imported_value_name_start(line_text: str, name: str) -> int:
    for prefix in ("def ", "async def ", "class "):
        if line_text.startswith(prefix):
            return len(prefix)
    start = line_text.find(name)
    return max(start, 0)


def _proc_name_range(proc: ProcedureDef) -> Optional[Range]:
    if proc.loc is None:
        return None
    line = proc.loc.start_line - 1
    start = proc.loc.start_col
    return _line_range(line, start, proc.name)


def _type_name_range(typedef: TypeDef) -> Optional[Range]:
    if typedef.loc is None:
        return None
    line = typedef.loc.start_line - 1
    start = typedef.loc.start_col
    return _line_range(line, start, typedef.name)


def _binding_name_range(
    stmt: VariableInitStmt | VariableInitBlockStmt | AssignmentStmt | AssignmentBlockStmt,
) -> Optional[Range]:
    if stmt.loc is None:
        return None
    line = stmt.loc.start_line - 1
    start = stmt.loc.start_col
    return _line_range(line, start, stmt.name)


def _raw_invoke_name_range(stmt: RawInvokeStmt) -> Optional[Range]:
    if stmt.loc is None:
        return None
    line = stmt.loc.start_line - 1
    start = stmt.loc.start_col
    return _line_range(line, start, stmt.name)


def _output_name_range(segment: OutputSegment) -> Optional[Range]:
    if segment.loc is None:
        return None
    line = segment.loc.start_line - 1
    start = segment.loc.start_col
    return _line_range(line, start, segment.name)


def _param_name_range(
    source_lines: list[str],
    proc: ProcedureDef,
    name: str,
) -> Optional[Range]:
    if proc.loc is None:
        return None
    line = proc.loc.start_line - 1
    if line < 0 or line >= len(source_lines):
        return None
    header = source_lines[line]
    open_paren = header.find("(")
    close_paren = header.find(")", open_paren + 1)
    if open_paren == -1:
        return None
    if close_paren == -1:
        close_paren = len(header)
    params = header[open_paren + 1 : close_paren]
    offset = 0
    for chunk in params.split(","):
        stripped = chunk.lstrip()
        leading = len(chunk) - len(stripped)
        candidate = stripped.split(":", 1)[0].strip()
        if candidate == name:
            start = open_paren + 1 + offset + leading
            return _line_range(line, start, name)
        offset += len(chunk) + 1
    return None


def _index_header_ranges(
    ranges: list[dict[str, Any]],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    indexed: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for item in ranges:
        owner_kind = item.get("ownerKind")
        container_line = item.get("containerLine")
        if owner_kind is None or container_line is None:
            continue
        indexed.setdefault(owner_kind, {}).setdefault(container_line, []).append(item)
    return indexed


def _container_line(node: ProcedureDef | TypeDef) -> int:
    if node.loc is None:
        return -1
    return node.loc.start_line - 1


def _add_proc_header_regions(
    builder: VirtualBuilder,
    proc: ProcedureDef,
    *,
    signature_line: int,
    signature: str,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    param_ranges = header_ranges.get("param", {}).get(_container_line(proc), [])
    type_ranges = [item["range"] for item in param_ranges if item["expressionKind"] == "type"]
    default_ranges = [item["range"] for item in param_ranges if item["expressionKind"] == "default"]
    search_from = signature.find("(") + 1
    for name in proc.params:
        piece = _proc_param_to_python(proc, name)
        start = signature.find(piece, search_from)
        if start == -1:
            continue
        type_expr = (proc.param_type_exprs or {}).get(name)
        if isinstance(type_expr, TypePython) and type_ranges:
            typ = _proc_param_type_to_python(proc, name)
            type_start = signature.find(typ, start, start + len(piece))
            if type_start != -1:
                builder.add_synthetic_inline_region(
                    source_range=type_ranges.pop(0),
                    virtual_range=_line_range(signature_line, type_start, typ),
                    text=typ,
                )
        default_expr = (proc.param_defaults or {}).get(name)
        if default_expr is not None and default_ranges:
            source_range = default_ranges.pop(0)
            expr_start = signature.find(default_expr, start, start + len(piece))
            if expr_start != -1:
                builder.add_synthetic_inline_region(
                    source_range=source_range,
                    virtual_range=_line_range(signature_line, expr_start, default_expr),
                    text=default_expr,
                )
        search_from = start + len(piece)

    if isinstance(proc.return_type_expr, TypePython):
        return_ranges = header_ranges.get("return_type", {}).get(_container_line(proc), [])
        if return_ranges:
            ret = _type_to_python(proc.return_type_expr, proc.return_type_annotation, default="str")
            arrow = signature.find("->")
            ret_start = signature.find(ret, arrow + 2)
            if ret_start != -1:
                builder.add_synthetic_inline_region(
                    source_range=return_ranges[0]["range"],
                    virtual_range=_line_range(signature_line, ret_start, ret),
                    text=ret,
                )


def _range_key(range_: Range) -> tuple[int, int, int, int]:
    return (
        range_["start"]["line"],
        range_["start"]["character"],
        range_["end"]["line"],
        range_["end"]["character"],
    )


def _add_default_expr_region(
    builder: VirtualBuilder,
    *,
    source_range: Range,
    line: int,
    line_text: str,
    expr: str,
) -> None:
    start = line_text.find(expr)
    if start == -1:
        return
    builder.add_synthetic_inline_region(
        source_range=source_range,
        virtual_range=_line_range(line, start, expr),
        text=expr,
    )


def _iter_imported_type_exprs(surface: ImportedSurface) -> Iterable[Optional[TypeExpr]]:
    for typedef in surface.types.values():
        for field in typedef.fields:
            yield field.type_expr
    for proc in surface.procedures.values():
        yield proc.return_type_expr
        yield from (proc.param_type_exprs or {}).values()
    for value in surface.values.values():
        yield value.type_expr


def _iter_type_exprs(program: ProgramWithValidation) -> Iterable[Optional[TypeExpr]]:
    for typedef in program.type_defs:
        for field in typedef.fields:
            yield field.type_expr
    for proc in program.procedures:
        yield proc.return_type_expr
        for expr in (proc.param_type_exprs or {}).values():
            yield expr
        for stmt in proc.body:
            yield from _iter_stmt_type_exprs(stmt)
    for stmt in program.toplevel_body:
        yield from _iter_stmt_type_exprs(stmt)


def _iter_stmt_type_exprs(stmt: Stmt) -> Iterable[Optional[TypeExpr]]:
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)):
        yield stmt.type_expr
    if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
        yield stmt.proc.return_type_expr
        for expr in (stmt.proc.param_type_exprs or {}).values():
            yield expr
        for child in stmt.proc.body:
            yield from _iter_stmt_type_exprs(child)
    for segment in _stmt_segments(stmt):
        if isinstance(segment, OutputSegment):
            yield segment.type_expr
    if isinstance(stmt, IfStmt):
        for child in stmt.body:
            yield from _iter_stmt_type_exprs(child)
        for child in stmt.else_body or []:
            yield from _iter_stmt_type_exprs(child)
    elif isinstance(stmt, LoopStmt):
        for child in stmt.body:
            yield from _iter_stmt_type_exprs(child)
        for child in stmt.map_body or []:
            yield from _iter_stmt_type_exprs(child)
    elif isinstance(stmt, ConditionalLoopStmt):
        for child in stmt.body:
            yield from _iter_stmt_type_exprs(child)
    elif isinstance(stmt, TaskGroupStmt):
        for arm in stmt.arms:
            for child in arm.body or []:
                yield from _iter_stmt_type_exprs(child)


def _stmt_segments(stmt: Stmt) -> list:
    if isinstance(stmt, TemplateBlock):
        segments: list = []
        for line in stmt.lines:
            segments.extend(line.segments)
        return segments
    if isinstance(stmt, TaskStmt):
        return _stmt_segments(stmt.template)
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


def _type_to_python(
    expr: Optional[TypeExpr], annotation: Optional[str], *, default: str = "str"
) -> str:
    return type_expr_to_python(expr, annotation, default=default)


def _proc_param_to_python(proc: ProcedureDef, name: str) -> str:
    typ = _proc_param_type_to_python(proc, name)
    piece = f"{name}: {typ}"
    default = (proc.param_defaults or {}).get(name)
    if default is not None:
        piece += f" = {default}"
    return piece


def _proc_param_type_to_python(proc: ProcedureDef, name: str) -> str:
    return _type_to_python(
        (proc.param_type_exprs or {}).get(name),
        (proc.param_type_annotations or {}).get(name),
        default="str",
    )
