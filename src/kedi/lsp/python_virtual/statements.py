from __future__ import annotations

from typing import Any, Iterable, Optional

from kedi.lang.ast import (
    AssignmentBlockStmt,
    AssignmentStmt,
    BacktickLineStmt,
    CallSegment,
    ConditionalLoopStmt,
    EvalDataset,
    EvalMetric,
    EvalSuite,
    IfStmt,
    LiteralArg,
    LoopStmt,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    ProgramWithValidation,
    PythonBlock,
    PythonExprNativeSegment,
    PythonExprSegment,
    RawInvokeStmt,
    ReturnBlockStmt,
    ReturnStmt,
    Span,
    Stmt,
    TemplateBlock,
    TemplateLine,
    TestCase,
    TypeDefStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lsp.python_virtual.builder import (
    PythonRegion,
    RegionCursor,
    ScopeName,
    VirtualBuilder,
    _line_range,
    _span_range,
)
from kedi.lsp.python_virtual.scope import (
    _add_proc_header_regions,
    _add_scope_param_symbols,
    _binding_name_range,
    _output_name_range,
    _param_name_range,
    _proc_name_range,
    _proc_param_to_python,
    _proc_param_type_to_python,
    _raw_invoke_name_range,
    _scope_params,
    _stmt_segments,
    _type_to_python,
)
from kedi.lsp.python_virtual.stubs import _docstring_lines, _emit_type_stub


def _emit_procedure_body(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    proc: ProcedureDef,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    params = ", ".join(_proc_param_to_python(proc, name) for name in proc.params)
    ret = _type_to_python(proc.return_type_expr, proc.return_type_annotation, default="str")
    signature = f"def __kedi_scope_{proc.name}({params}) -> {ret}:"
    signature_line = builder.add(signature)
    _add_proc_header_regions(
        builder,
        proc,
        signature_line=signature_line,
        signature=signature,
        header_ranges=header_ranges,
    )
    scope = {
        name: ScopeName(
            typ=_type_to_python(
                (proc.param_type_exprs or {}).get(name),
                (proc.param_type_annotations or {}).get(name),
                default="str",
            ),
            source_range=_param_name_range(builder.source_lines, proc, name),
        )
        for name in proc.params
    }
    _add_scope_param_symbols(builder, scope, signature_line, signature)
    if proc.docstring:
        for doc_line in _docstring_lines(proc.docstring, "    "):
            builder.add(doc_line)
    _emit_statements(builder, cursor, proc.body, scope, "    ", header_ranges)
    builder.add(f"    return cast({ret}, _kedi_placeholder())")
    builder.add("")


def _emit_nested_procedure(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    proc: ProcedureDef,
    parent_scope: dict[str, ScopeName],
    indent: str,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    params = ", ".join(_proc_param_to_python(proc, name) for name in proc.params)
    ret = _type_to_python(proc.return_type_expr, proc.return_type_annotation, default="str")
    signature = f"{indent}def {proc.name}({params}) -> {ret}:"
    signature_line = builder.add(signature)
    _add_proc_header_regions(
        builder,
        proc,
        signature_line=signature_line,
        signature=signature,
        header_ranges=header_ranges,
    )
    own_scope = {
        name: ScopeName(
            typ=_type_to_python(
                (proc.param_type_exprs or {}).get(name),
                (proc.param_type_annotations or {}).get(name),
                default="str",
            ),
            source_range=_param_name_range(builder.source_lines, proc, name),
        )
        for name in proc.params
    }
    scope = {**parent_scope, **own_scope}
    builder.add_symbol(
        kind="procedure",
        name=proc.name,
        source_range=_proc_name_range(proc),
        virtual_range=_line_range(signature_line, len(indent) + len("def "), proc.name),
    )
    _add_scope_param_symbols(builder, own_scope, signature_line, signature)
    if proc.docstring:
        for doc_line in _docstring_lines(proc.docstring, indent + "    "):
            builder.add(doc_line)
    _emit_statements(
        builder,
        cursor,
        proc.body,
        scope,
        indent + "    ",
        header_ranges,
    )
    builder.add(f"{indent}    return cast({ret}, _kedi_placeholder())")


def _emit_statements(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    statements: Iterable[Stmt],
    scope: dict[str, ScopeName],
    indent: str,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    for stmt in statements:
        if isinstance(stmt, IfStmt):
            _emit_if_stmt(builder, cursor, stmt, scope, indent, header_ranges)
        elif isinstance(stmt, LoopStmt):
            _emit_loop_stmt(builder, cursor, stmt, scope, indent, header_ranges)
        elif isinstance(stmt, ConditionalLoopStmt):
            _emit_conditional_loop_stmt(builder, cursor, stmt, scope, indent, header_ranges)
        elif isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            _emit_nested_procedure(builder, cursor, stmt.proc, scope, indent, header_ranges)
        elif isinstance(stmt, TypeDefStmt) and stmt.typedef is not None:
            _emit_type_stub(builder, stmt.typedef, header_ranges, prefix=indent)
        elif isinstance(stmt, VariableInitBlockStmt):
            _emit_variable_init_block(builder, cursor, stmt, scope, indent)
        elif isinstance(stmt, AssignmentBlockStmt):
            _emit_assignment_block(builder, cursor, stmt, scope, indent)
        elif isinstance(stmt, VariableInitStmt):
            _emit_variable_init_stmt(builder, cursor, stmt, scope, indent)
        elif isinstance(stmt, AssignmentStmt):
            _emit_assignment_stmt(builder, cursor, stmt, scope, indent)
        elif isinstance(stmt, RawInvokeStmt):
            _emit_raw_invoke_stmt(builder, stmt, scope, indent)
        elif isinstance(stmt, PythonBlock):
            region = cursor.next("fenced", stmt.loc)
            if region is not None:
                builder.add_mapped_region(region, indent)
        elif isinstance(stmt, BacktickLineStmt):
            region = cursor.next("inline", stmt.loc)
            if region is not None:
                builder.add_mapped_region(region, indent)
        elif isinstance(stmt, TemplateBlock):
            for line in stmt.lines:
                _emit_segments(builder, cursor, line, scope, indent)
        elif isinstance(stmt, TemplateLine):
            _emit_segments(builder, cursor, stmt, scope, indent)
        elif isinstance(stmt, ReturnStmt):
            _emit_return(builder, cursor, stmt, scope, indent)
        elif isinstance(stmt, ReturnBlockStmt):
            _emit_return(builder, cursor, stmt, scope, indent)
        _emit_unconsumed_inline_regions(builder, cursor, stmt.loc, indent)


def _emit_if_stmt(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: IfStmt,
    scope: dict[str, ScopeName],
    indent: str,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    builder.add(f"{indent}if (")
    condition = cursor.next("inline", stmt.condition_loc)
    if condition is not None:
        builder.add_mapped_region(condition, indent + "    ")
    else:
        builder.add(f"{indent}    cast(bool, _kedi_placeholder())")
    builder.add(f"{indent}):")

    consequence_scope = dict(scope)
    before_body = builder.line
    _emit_statements(
        builder,
        cursor,
        stmt.body,
        consequence_scope,
        indent + "    ",
        header_ranges,
    )
    if builder.line == before_body:
        builder.add(f"{indent}    pass")

    alternative_scope = dict(scope)
    if stmt.else_body is not None:
        builder.add(f"{indent}else:")
        before_body = builder.line
        _emit_statements(
            builder,
            cursor,
            stmt.else_body,
            alternative_scope,
            indent + "    ",
            header_ranges,
        )
        if builder.line == before_body:
            builder.add(f"{indent}    pass")


def _emit_loop_stmt(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: LoopStmt,
    scope: dict[str, ScopeName],
    indent: str,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    previous_binder = scope.get(stmt.binder)
    line = builder.add(f"{indent}for {stmt.binder} in (")
    builder.add_symbol(
        kind="local",
        name=stmt.binder,
        source_range=_span_range(stmt.binder_loc) if stmt.binder_loc is not None else None,
        virtual_range=_line_range(line, len(indent) + len("for "), stmt.binder),
    )
    iterable = cursor.next("inline", stmt.iterable_loc)
    if iterable is not None:
        builder.add_mapped_region(iterable, indent + "    ")
    else:
        builder.add(f"{indent}    cast(Iterable[Any], _kedi_placeholder())")
    builder.add(f"{indent}):")

    body_scope = dict(scope)
    body_scope[stmt.binder] = ScopeName(
        typ="Any",
        source_range=_span_range(stmt.binder_loc) if stmt.binder_loc is not None else None,
    )
    before_body = builder.line
    _emit_statements(
        builder,
        cursor,
        stmt.body,
        body_scope,
        indent + "    ",
        header_ranges,
    )
    if stmt.map_body is not None:
        _emit_statements(
            builder,
            cursor,
            stmt.map_body,
            body_scope,
            indent + "    ",
            header_ranges,
        )
    if builder.line == before_body:
        builder.add(f"{indent}    pass")

    body_scope.pop(stmt.binder, None)
    if previous_binder is None:
        builder.add(f"{indent}del {stmt.binder}  # pyright: ignore[reportPossiblyUnboundVariable]")
    else:
        builder.add(
            f"{indent}{stmt.binder}: {previous_binder.typ} = "
            f"cast({previous_binder.typ}, _kedi_placeholder())"
        )
        scope[stmt.binder] = previous_binder


def _emit_conditional_loop_stmt(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: ConditionalLoopStmt,
    scope: dict[str, ScopeName],
    indent: str,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
) -> None:
    builder.add(f"{indent}while (")
    condition = cursor.next("inline", stmt.condition_loc)
    if condition is not None:
        builder.add_mapped_region(condition, indent + "    ")
    else:
        builder.add(f"{indent}    cast(bool, _kedi_placeholder())")
    builder.add(f"{indent}):")

    body_scope = dict(scope)
    before_body = builder.line
    _emit_statements(
        builder,
        cursor,
        stmt.body,
        body_scope,
        indent + "    ",
        header_ranges,
    )
    if builder.line == before_body:
        builder.add(f"{indent}    pass")


def _emit_unconsumed_inline_regions(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    span: Optional[Span],
    indent: str,
) -> None:
    while (region := cursor.next("inline", span)) is not None:
        _emit_inline_expression_stub(builder, region, indent)


def _emit_variable_init_block(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: VariableInitBlockStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    typ = _type_to_python(stmt.type_expr, stmt.type_annotation)
    helper = f"__kedi_block_{stmt.name}_{stmt.loc.start_line if stmt.loc else len(scope)}"
    params = _scope_params(scope)
    signature = f"{indent}def {helper}({params}) -> {typ}:"
    signature_line = builder.add(signature)
    _add_scope_param_symbols(builder, scope, signature_line, signature)
    region = cursor.next("fenced", stmt.loc)
    if region is not None:
        builder.add_mapped_region(region, indent + "    ")
    builder.add(f"{indent}    return cast({typ}, _kedi_placeholder())")
    args = ", ".join(scope.keys())
    line = builder.add(f"{indent}{stmt.name}: {typ} = {helper}({args})")
    source_range = _binding_name_range(stmt)
    builder.add_symbol(
        kind="local",
        name=stmt.name,
        source_range=source_range,
        virtual_range=_line_range(line, len(indent), stmt.name),
    )
    scope[stmt.name] = ScopeName(typ=typ, source_range=source_range)


def _emit_variable_init_stmt(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: VariableInitStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    typ = _type_to_python(stmt.type_expr, stmt.type_annotation)
    source_range = _binding_name_range(stmt)
    inline = _single_python_segment(stmt.expr)
    if inline is not None:
        line = builder.add(f"{indent}{stmt.name}: {typ} = (")
        builder.add_symbol(
            kind="local",
            name=stmt.name,
            source_range=source_range,
            virtual_range=_line_range(line, len(indent), stmt.name),
        )
        region = cursor.next("inline", inline.loc)
        if region is not None:
            builder.add_mapped_region(region, indent + "    ")
        else:
            builder.add(f"{indent}    return cast({typ}, _kedi_placeholder())")
        builder.add(f"{indent})")
    else:
        line = builder.add(f"{indent}{stmt.name}: {typ} = cast({typ}, _kedi_placeholder())")
        builder.add_symbol(
            kind="local",
            name=stmt.name,
            source_range=source_range,
            virtual_range=_line_range(line, len(indent), stmt.name),
        )
        _emit_segments(builder, cursor, stmt, scope, indent)
    scope[stmt.name] = ScopeName(typ=typ, source_range=source_range)


def _emit_assignment_block(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: AssignmentBlockStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    info = scope.get(stmt.name, ScopeName(typ="Any", source_range=None))
    helper = f"__kedi_assign_block_{stmt.name}_{stmt.loc.start_line if stmt.loc else len(scope)}"
    params = _scope_params(scope)
    builder.add(f"{indent}def {helper}({params}) -> {info.typ}:")
    region = cursor.next("fenced", stmt.loc)
    if region is not None:
        builder.add_mapped_region(region, indent + "    ")
    builder.add(f"{indent}    return cast({info.typ}, _kedi_placeholder())")
    args = ", ".join(scope.keys())
    line = builder.add(f"{indent}{stmt.name}")
    result_name = f"__kedi_assigned_{stmt.name}_{stmt.loc.start_line if stmt.loc else builder.line}"
    builder.add(f"{indent}{result_name}: {info.typ} = {helper}({args})")
    source_range = _binding_name_range(stmt)
    builder.add_symbol(
        kind="reference",
        name=stmt.name,
        source_range=source_range,
        virtual_range=_line_range(line, len(indent), stmt.name),
    )


def _emit_assignment_stmt(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: AssignmentStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    info = scope.get(stmt.name, ScopeName(typ="Any", source_range=None))
    source_range = _binding_name_range(stmt)
    line = builder.add(f"{indent}{stmt.name}")
    temporary = f"__kedi_assigned_{stmt.name}_{stmt.loc.start_line if stmt.loc else builder.line}"
    inline = _single_python_segment(stmt.expr)
    if inline is not None:
        builder.add(f"{indent}{temporary}: {info.typ} = (")
        region = cursor.next("inline", inline.loc)
        if region is not None:
            builder.add_mapped_region(region, indent + "    ")
        else:
            builder.add(f"{indent}    cast({info.typ}, _kedi_placeholder())")
        builder.add(f"{indent})")
    else:
        builder.add(f"{indent}{temporary}: {info.typ} = cast({info.typ}, _kedi_placeholder())")
        _emit_segments(builder, cursor, stmt, scope, indent)
    builder.add_symbol(
        kind="reference",
        name=stmt.name,
        source_range=source_range,
        virtual_range=_line_range(line, len(indent), stmt.name),
    )


def _emit_raw_invoke_stmt(
    builder: VirtualBuilder,
    stmt: RawInvokeStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    line = builder.add(f"{indent}{stmt.name}: str = cast(str, _kedi_placeholder())")
    source_range = _raw_invoke_name_range(stmt)
    builder.add_symbol(
        kind="local",
        name=stmt.name,
        source_range=source_range,
        virtual_range=_line_range(line, len(indent), stmt.name),
    )
    scope[stmt.name] = ScopeName(typ="str", source_range=source_range)


def _emit_return(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: ReturnStmt | ReturnBlockStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    if isinstance(stmt, ReturnBlockStmt):
        region = cursor.next("fenced", stmt.loc)
        if region is None:
            # Native single-backtick returns intentionally share the
            # ReturnBlockStmt AST node so runtime values keep their Python type.
            # Preserve that expression in the virtual document as well.
            region = cursor.next("inline", stmt.loc)
        if region is not None:
            if region.kind == "fenced":
                builder.add_mapped_region(region, indent)
            else:
                _emit_inline_expression_stub(builder, region, indent)
        return
    _emit_segments(builder, cursor, stmt, scope, indent)


def _emit_segments(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: Any,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    for segment in _stmt_segments(stmt):
        if isinstance(segment, OutputSegment):
            typ = _type_to_python(segment.type_expr, segment.type_annotation)
            line = builder.add(f"{indent}{segment.name}: {typ} = cast({typ}, _kedi_placeholder())")
            source_range = _output_name_range(segment)
            builder.add_symbol(
                kind="local",
                name=segment.name,
                source_range=source_range,
                virtual_range=_line_range(line, len(indent), segment.name),
            )
            scope[segment.name] = ScopeName(typ=typ, source_range=source_range)
        elif isinstance(segment, (PythonExprSegment, PythonExprNativeSegment)):
            region = cursor.next("inline", segment.loc)
            if region is not None:
                _emit_inline_expression_stub(builder, region, indent)
        elif isinstance(segment, CallSegment):
            for nested_loc in _call_inline_python_locations(segment):
                region = cursor.next("inline", nested_loc)
                if region is not None:
                    _emit_inline_expression_stub(builder, region, indent)


def _call_inline_python_locations(segment: CallSegment) -> Iterable[Optional[Span]]:
    for arg in segment.args:
        if not isinstance(arg, LiteralArg):
            continue
        if arg.native_backtick is not None:
            yield arg.loc
        if arg.python_segment is not None:
            yield arg.python_segment.loc
        if arg.call_segment is not None:
            yield from _call_inline_python_locations(arg.call_segment)
        for child in arg.render_segments:
            if isinstance(child, (PythonExprSegment, PythonExprNativeSegment)):
                yield child.loc
            elif isinstance(child, CallSegment):
                yield from _call_inline_python_locations(child)


def _emit_inline_expression_stub(
    builder: VirtualBuilder,
    region: PythonRegion,
    indent: str,
) -> None:
    """Map a Kedi inline expression without producing an unused expression.

    A raw Python expression in a template or return segment is meaningful to
    Kedi but has no Python statement-level effect. Wrapping it in a synthetic
    assignment preserves the exact mapped expression for Python tooling while
    avoiding downstream ``reportUnusedExpression`` diagnostics.
    """
    name = f"__kedi_inline_expression_{builder.line}"
    builder.add(f"{indent}{name}: Any = (")
    builder.add_mapped_region(region, indent + "    ")
    builder.add(f"{indent})")


def _emit_validation(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    program: ProgramWithValidation,
    procedure_map: dict[str, ProcedureDef],
) -> None:
    for suite in program.tests:
        for case in suite.cases:
            _emit_validation_code(
                builder, cursor, case, f"__kedi_test_{suite.procedure_name}_{case.name}"
            )
    for suite in program.evals:
        for dataset in list(suite.datasets.values()) + list(suite.test_datasets.values()):
            _emit_validation_code(
                builder,
                cursor,
                dataset,
                f"__kedi_data_{suite.procedure_name}_{dataset.name}",
            )
        for metric in suite.metrics:
            _emit_metric(
                builder,
                cursor,
                suite,
                metric,
                procedure_map.get(suite.procedure_name),
            )


def _emit_validation_code(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    item: TestCase | EvalDataset,
    name: str,
) -> None:
    builder.add(f"def {name}() -> Any:")
    region = cursor.next("fenced" if item.is_block else "inline", item.loc)
    if region is not None:
        builder.add_mapped_region(region, "    ")
    else:
        builder.add("    pass")
    builder.add("")


def _emit_metric(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    suite: EvalSuite,
    metric: EvalMetric,
    procedure: Optional[ProcedureDef],
) -> None:
    builder.add(f"def __kedi_metric_{suite.procedure_name}_{metric.name}() -> Any:")
    if metric.dataset_name:
        builder.add(f"    {metric.dataset_name}: Any = _kedi_placeholder()")
    builder.add("    expected: Any = _kedi_placeholder()")
    if procedure is not None:
        for name in procedure.params:
            if name == metric.dataset_name:
                continue
            typ = _proc_param_type_to_python(procedure, name)
            builder.add(f"    {name}: {typ} = cast({typ}, _kedi_placeholder())")
    region = cursor.next("fenced" if metric.is_block else "inline", metric.loc)
    if region is not None:
        builder.add_mapped_region(region, "    ")
    else:
        builder.add("    pass")
    builder.add("")


def _single_python_segment(
    expr: list[Any],
) -> Optional[PythonExprNativeSegment | PythonExprSegment]:
    if len(expr) != 1:
        return None
    segment = expr[0]
    if isinstance(segment, (PythonExprNativeSegment, PythonExprSegment)):
        return segment
    return None
