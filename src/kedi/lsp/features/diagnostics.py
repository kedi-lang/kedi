from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Union

from lsprotocol import types as lsp

from kedi.lang.ast import (
    AssignmentBlockStmt,
    AssignmentStmt,
    CodeModeDirectiveStmt,
    OutputSegment,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    Span,
    TypeApply,
    TypeExpr,
    TypeLiteral,
    TypeRef,
    TypeUnion,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.compiler.environment import CLI_ARGS_IDENTIFIER
from kedi.lang.compiler.type_system import UNSUPPORTED_KEDI_TYPE_NAMES
from kedi.lang.parser import KediDiagnostic, KediParseError, parse_program
from kedi.lsp.ast_helpers import binding_name_span
from kedi.lsp.coordinates import span_to_range as _span_to_range
from kedi.lsp.coordinates import with_source_coordinates
from kedi.lsp.features.capabilities import _profile_capability_diagnostics, _scan_capability_scope
from kedi.lsp.features.scope import (
    _all_procedures,
    _control_body_statements,
    _iter_control_statements,
    _procedure_scope_entries,
    _resolve_value_declaration,
    _segments_of,
)
from kedi.lsp.features.symbols import _import_name_span, _merged_profile_defs
from kedi.lsp.module_surface import (
    import_resolution_error,
    imported_surface_for_program,
    load_imported_surface,
    program_path,
    resolved_import_path,
)
from kedi.module_registry import describe_module_search_paths, module_search_paths

from .capability_state import _AdapterDiagnosticState, _profile_adapter_state


@with_source_coordinates(1)
def compute_diagnostics(
    source: str,
    source_path: Optional[str],
    *,
    codemode_preload_warning_threshold: int = 5,
    interactive: bool = False,
) -> List[lsp.Diagnostic]:
    """Parse ``source`` and surface every ``KediDiagnostic`` as LSP."""
    try:
        prog = parse_program(source, source_path=source_path)
        if interactive:
            from kedi.lang.parser.control_flow_validation import validate_control_flow

            validate_control_flow(prog, interactive=True)
        out = _missing_import_diagnostics(prog)
        out.extend(_subagent_profile_diagnostics(prog))
        out.extend(_adapter_capability_diagnostics(prog))
        out.extend(_annotated_type_diagnostics(prog))
        out.extend(_assignment_diagnostics(prog))
        out.extend(
            _codemode_preload_diagnostics(
                prog,
                threshold=codemode_preload_warning_threshold,
            )
        )
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


def _codemode_preload_diagnostics(
    prog: Program | ProgramWithValidation,
    *,
    threshold: int,
) -> list[lsp.Diagnostic]:
    directives: list[CodeModeDirectiveStmt | object] = [
        item
        for item in _iter_control_statements(prog.ordered_toplevel or prog.toplevel_body)
        if isinstance(item, CodeModeDirectiveStmt)
    ]
    for procedure in _all_procedures(prog):
        directives.extend(
            statement
            for statement in _iter_control_statements(procedure.body)
            if isinstance(statement, CodeModeDirectiveStmt)
        )
    directives.extend(
        member for profile in prog.profile_defs.values() for member in profile.codemode_members
    )

    diagnostics: list[lsp.Diagnostic] = []
    seen_locations: set[tuple[str | None, int, int]] = set()
    for directive in directives:
        fields = getattr(directive, "fields", ())
        preload_field = next(
            (field for field in fields if field.name == "preload_tools"),
            None,
        )
        if preload_field is None or preload_field.backtick_code is not None:
            continue
        names = (
            preload_field.plain_values
            if preload_field.plain_values is not None
            else [preload_field.plain_value]
            if preload_field.plain_value
            else []
        )
        unique_count = len(dict.fromkeys(names))
        if unique_count <= threshold or preload_field.loc is None:
            continue
        location = (
            preload_field.loc.source_path,
            preload_field.loc.start_byte,
            preload_field.loc.end_byte,
        )
        if location in seen_locations:
            continue
        seen_locations.add(location)
        diagnostics.append(
            _warning(
                preload_field.loc,
                "kedi.codemode.excessive_preload",
                "Many tools are preloaded. Unused or rarely used schemas can enlarge context, "
                "increase cost, and obscure relevant information. Consider preloading only "
                "tools required at the start of the run.",
            )
        )
    return diagnostics


def _annotated_type_diagnostics(
    prog: Union[Program, ProgramWithValidation],
) -> List[lsp.Diagnostic]:
    out: List[lsp.Diagnostic] = []
    for expr in _iter_type_exprs(prog):
        _collect_annotated_type_diagnostics(expr, out)
    return out


def _assignment_diagnostics(
    prog: Program | ProgramWithValidation,
) -> List[lsp.Diagnostic]:
    imported_values = imported_surface_for_program(prog).values
    out: List[lsp.Diagnostic] = []

    def visit(
        statements: Sequence[object],
        proc: ProcedureDef | None,
        ancestors: tuple[ProcedureDef, ...],
    ) -> None:
        for stmt in _iter_control_statements(statements):
            if not isinstance(stmt, (AssignmentStmt, AssignmentBlockStmt)):
                continue
            target_span = binding_name_span(stmt)
            if target_span is None:
                continue
            if stmt.name == CLI_ARGS_IDENTIFIER:
                out.append(
                    _error(
                        target_span,
                        "K-ASSIGN-RESERVED",
                        f"Reserved CLI argument binding '{CLI_ARGS_IDENTIFIER}' cannot be assigned.",
                    )
                )
                continue
            if stmt.name in imported_values:
                continue
            if (
                _resolve_value_declaration(
                    prog,
                    proc,
                    ancestors,
                    stmt.name,
                    target_span,
                )
                is None
            ):
                out.append(
                    _error(
                        target_span,
                        "K-ASSIGN-UNKNOWN",
                        f"Cannot assign unknown Kedi binding '{stmt.name}'.",
                    )
                )

    visit(prog.toplevel_body, None, ())
    for proc, ancestors in _procedure_scope_entries(prog):
        visit(proc.body, proc, ancestors)
    return out


def _iter_type_exprs(prog: Union[Program, ProgramWithValidation]):
    for typedef in prog.type_defs:
        for field in typedef.fields:
            yield field.type_expr
    for profile in prog.profile_defs.values():
        for member in profile.output_members:
            yield member.type_expr
    for proc in _all_procedures(prog):
        yield proc.return_type_expr
        for expr in (proc.param_type_exprs or {}).values():
            yield expr
        for stmt in proc.body:
            yield from _iter_stmt_type_exprs(stmt)
    for stmt in prog.toplevel_body:
        yield from _iter_stmt_type_exprs(stmt)
    if prog.toplevel_return is not None:
        yield from _iter_stmt_type_exprs(prog.toplevel_return)


def _iter_stmt_type_exprs(stmt):
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)):
        yield stmt.type_expr
    for segment in _segments_of(stmt):
        if isinstance(segment, OutputSegment):
            yield segment.type_expr
    for child in _control_body_statements(stmt):
        yield from _iter_stmt_type_exprs(child)


def _collect_annotated_type_diagnostics(
    expr: Optional[TypeExpr],
    out: List[lsp.Diagnostic],
) -> None:
    if expr is None:
        return
    if isinstance(expr, (TypeRef, TypeApply)):
        if expr.name in UNSUPPORTED_KEDI_TYPE_NAMES and expr.loc is not None:
            out.append(
                _error(
                    expr.loc,
                    "K-TYPE-UNSUPPORTED",
                    f"Unsupported Kedi type annotation '{expr.name}'.",
                )
            )
    if isinstance(expr, TypeApply):
        if expr.name == "Annotated" and expr.loc is not None:
            if len(expr.args) == 1:
                out.append(
                    _warning(
                        expr.loc,
                        "K-ANNOTATED-DESCRIPTION",
                        "Annotated[type] is accepted as the runtime type, but Kedi "
                        "Annotated metadata should include a string description: "
                        'Annotated[type, "description"].',
                    )
                )
            elif len(expr.args) > 2:
                out.append(
                    _warning(
                        expr.loc,
                        "K-ANNOTATED-EXTRA",
                        'Kedi uses only Annotated[type, "description"]; extra Annotated '
                        "metadata is ignored.",
                    )
                )
            elif len(expr.args) == 2 and not _is_string_type_literal(expr.args[1]):
                out.append(
                    _warning(
                        expr.loc,
                        "K-ANNOTATED-DESCRIPTION",
                        'Kedi Annotated metadata expects Annotated[type, "description"]; '
                        "the second argument should be a string literal.",
                    )
                )
        for arg in expr.args:
            _collect_annotated_type_diagnostics(arg, out)
    elif isinstance(expr, TypeUnion):
        _collect_annotated_type_diagnostics(expr.left, out)
        _collect_annotated_type_diagnostics(expr.right, out)


def _is_string_type_literal(expr: TypeExpr) -> bool:
    return isinstance(expr, TypeLiteral) and isinstance(expr.value, str)


def _warning(span: Span, code: str, message: str) -> lsp.Diagnostic:
    return lsp.Diagnostic(
        range=_span_to_range(span),
        severity=lsp.DiagnosticSeverity.Warning,
        code=code,
        source="kedi",
        message=message,
    )


def _error(span: Span, code: str, message: str) -> lsp.Diagnostic:
    return lsp.Diagnostic(
        range=_span_to_range(span),
        severity=lsp.DiagnosticSeverity.Error,
        code=code,
        source="kedi",
        message=message,
    )


def _missing_import_diagnostics(
    prog: Program | ProgramWithValidation,
) -> List[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for module_import in prog.imports:
        resolution_error = import_resolution_error(prog, module_import.module)
        if resolution_error is not None:
            span = _import_name_span(prog, module_import) or module_import.loc
            if span is not None:
                diagnostics.append(
                    lsp.Diagnostic(
                        range=_span_to_range(span),
                        severity=lsp.DiagnosticSeverity.Error,
                        code="K-IMPORT-CORRUPT",
                        source="kedi",
                        message=(
                            f"installed package for import {module_import.module!r} is invalid: "
                            f"{resolution_error}"
                        ),
                    )
                )
            continue
        module_path = resolved_import_path(prog, module_import.module)
        if module_path is not None:
            imported = load_imported_surface(module_path, set(), module_import.module)
            exported_names = imported.exported_names()
            for name in dict.fromkeys(module_import.names):
                if name in exported_names:
                    continue
                span = module_import.name_locs[name]
                diagnostics.append(
                    lsp.Diagnostic(
                        range=_span_to_range(span),
                        severity=lsp.DiagnosticSeverity.Error,
                        code="K-IMPORT-NAME",
                        source="kedi",
                        message=f"module '{module_import.module}' does not export '{name}'",
                    )
                )
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


def _subagent_profile_diagnostics(
    prog: Program | ProgramWithValidation,
) -> list[lsp.Diagnostic]:
    profile_defs = _merged_profile_defs(prog, imported_surface_for_program(prog))
    diagnostics: list[lsp.Diagnostic] = []
    graph = {
        name: tuple(entry.name for entry in profile.subagent_members)
        for name, profile in prog.profile_defs.items()
    }

    for profile in prog.profile_defs.values():
        for entry in profile.subagent_members:
            if entry.name in profile_defs or entry.loc is None:
                continue
            diagnostics.append(
                lsp.Diagnostic(
                    range=_span_to_range(entry.loc),
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-SUBAGENT-UNKNOWN",
                    source="kedi",
                    message=(
                        f"Unknown subagent profile '{entry.name}' referenced by "
                        f"profile '{profile.name}'"
                    ),
                )
            )

    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(profile_name: str) -> None:
        if profile_name in visited:
            return
        if profile_name in active_set:
            cycle_start = active.index(profile_name)
            cycle = (*active[cycle_start:], profile_name)
            parent = prog.profile_defs[cycle[0]]
            entry = next(item for item in parent.subagent_members if item.name == cycle[1])
            assert entry.loc is not None
            diagnostics.append(
                lsp.Diagnostic(
                    range=_span_to_range(entry.loc),
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-SUBAGENT-CYCLE",
                    source="kedi",
                    message=("Cyclic subagent profile reference: " + " -> ".join(cycle)),
                )
            )
            return
        active.append(profile_name)
        active_set.add(profile_name)
        for child_name in graph.get(profile_name, ()):
            if child_name in graph:
                visit(child_name)
        active.pop()
        active_set.remove(profile_name)
        visited.add(profile_name)

    for profile_name in graph:
        visit(profile_name)
    return diagnostics


def _adapter_capability_diagnostics(
    prog: Program | ProgramWithValidation,
) -> list[lsp.Diagnostic]:
    imported_surface = imported_surface_for_program(prog)
    profile_defs = _merged_profile_defs(prog, imported_surface)
    proc_names = {proc.name for proc in _all_procedures(prog)}
    proc_names.update(imported_surface.procedures)
    profile_states = {
        name: _profile_adapter_state(profile) for name, profile in profile_defs.items()
    }
    type_defs = {typedef.name: typedef for typedef in prog.type_defs}
    type_defs.update(imported_surface.types)

    diagnostics: list[lsp.Diagnostic] = []
    warned_approval_spans: set[tuple[str | None, int, int]] = set()
    for profile in prog.profile_defs.values():
        diagnostics.extend(
            _profile_capability_diagnostics(
                prog,
                profile,
                _profile_adapter_state(profile),
                warned_approval_spans,
            )
        )

    diagnostics.extend(
        _scan_capability_scope(
            prog,
            prog.ordered_toplevel or prog.toplevel_body,
            _AdapterDiagnosticState(),
            profile_defs,
            profile_states,
            proc_names,
            type_defs,
            warned_approval_spans,
        )
    )
    return diagnostics
