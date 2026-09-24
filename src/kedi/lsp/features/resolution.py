from __future__ import annotations

from typing import Optional, Union

from lsprotocol import types as lsp

from kedi.lang.ast import (
    AssignmentBlockStmt,
    AssignmentStmt,
    AwaitStmt,
    CallSegment,
    InputSegment,
    LoopStmt,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    TaskGroupStmt,
    TaskStmt,
    TypeApply,
    TypeDef,
    TypeDefStmt,
    TypeExpr,
    TypeRef,
    TypeUnion,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.call_arg_access import iter_call_arg_nested
from kedi.lsp.ast_helpers import binding_name_span
from kedi.lsp.ast_spans import name_token_span
from kedi.lsp.coordinates import position_inside as _position_inside
from kedi.lsp.features.scope import (
    _control_body_statements,
    _direct_type_defs,
    _exported_name_at_position,
    _is_imported_python_class,
    _nested_procedures,
    _param_decls,
    _position_in_proc_name,
    _proc_name_span,
    _resolve_lexical_name,
    _resolve_value_declaration,
    _segments_of,
    _visible_control_statements,
)
from kedi.lsp.features.symbols import (
    SymbolRef,
    _directive_name_span,
    _import_name_span,
    _input_name_span,
    _merged_profile_defs,
    _output_name_span,
    _profile_use_entry_spans,
    _use_single_target_span,
    _use_tool_entry_spans,
    _validation_procedure_span,
    _validation_suites,
    _ValueDeclaration,
)
from kedi.lsp.module_surface import (
    ImportedSurface,
    ImportedValueInfo,
    imported_surface_for_program,
    load_imported_surface,
    resolved_import_path,
    top_level_value_infos,
)


def _resolve_symbol_at(
    prog: Union[Program, ProgramWithValidation], pos: lsp.Position
) -> Optional[SymbolRef]:
    """Identify what the cursor is on. Returns ``None`` if nothing useful."""
    imported_surface = imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in prog.procedures}
    proc_by_name.update(imported_surface.procedures)
    type_by_name = {t.name: t for t in prog.type_defs}
    type_by_name.update(imported_surface.types)
    top_level_values = top_level_value_infos(prog)

    for module_import in prog.imports:
        name_span = _import_name_span(prog, module_import)
        if name_span is None or not _position_inside(pos, name_span):
            for selected_name in module_import.names:
                selected_span = module_import.name_locs[selected_name]
                if not _position_inside(pos, selected_span):
                    continue
                module_path = resolved_import_path(prog, module_import.module)
                selected_surface = (
                    load_imported_surface(module_path, set(), module_import.module)
                    if module_path is not None
                    else ImportedSurface()
                )
                if selected_name in selected_surface.procedures:
                    target = selected_surface.procedures[selected_name]
                    kind = "procedure"
                    declaration_span = target.loc
                elif selected_name in selected_surface.types:
                    target = selected_surface.types[selected_name]
                    kind = "type"
                    declaration_span = target.loc
                elif selected_name in selected_surface.values:
                    target = selected_surface.values[selected_name]
                    kind = "imported_value"
                    declaration_span = target.declaration_span
                elif selected_name in selected_surface.profiles:
                    target = selected_surface.profiles[selected_name]
                    kind = "agent_profile"
                    declaration_span = target.loc
                else:
                    kind = "imported_value"
                    declaration_span = None
                return SymbolRef(
                    kind=kind,
                    name=selected_name,
                    declaration_span=declaration_span,
                    scope_proc=None,
                    cursor_span=selected_span,
                )
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

    for profile in prog.profile_defs.values():
        for entry in profile.output_members:
            resolved = _resolve_in_type_expr(
                entry.type_expr,
                pos,
                type_by_name,
                imported_surface.values,
            )
            if resolved is not None:
                return resolved
        for entry in profile.subagent_members:
            assert entry.loc is not None
            if not _position_inside(pos, entry.loc):
                continue
            target = profile_defs.get(entry.name)
            return SymbolRef(
                kind="agent_profile",
                name=entry.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=entry.loc,
            )
        for target_name, name_span in _profile_use_entry_spans(prog, profile):
            if not _position_inside(pos, name_span):
                continue
            target_proc = proc_by_name.get(target_name)
            if target_proc is not None:
                return SymbolRef(
                    kind="agent_tool",
                    name=target_name,
                    declaration_span=target_proc.loc,
                    scope_proc=None,
                    cursor_span=name_span,
                )
            target_profile = profile_defs.get(target_name)
            if target_profile is not None:
                return SymbolRef(
                    kind="agent_profile",
                    name=target_name,
                    declaration_span=target_profile.loc,
                    scope_proc=None,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="agent_tool",
                name=target_name,
                declaration_span=None,
                scope_proc=None,
                cursor_span=name_span,
            )

    for item in prog.ordered_toplevel:
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

    for suite in _validation_suites(prog):
        name_span = _validation_procedure_span(prog, suite)
        if name_span is None or not _position_inside(pos, name_span):
            continue
        target = proc_by_name.get(suite.procedure_name)
        return SymbolRef(
            kind="procedure",
            name=suite.procedure_name,
            declaration_span=(target.loc if target is not None else None),
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
        r = _resolve_in_type_def(td, pos, type_by_name, imported_surface.values)
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
    ancestor_procs: tuple[ProcedureDef, ...] = (),
) -> Optional[SymbolRef]:
    if proc.loc is None or not _position_inside(pos, proc.loc):
        return None
    scoped_type_by_name = {
        **type_by_name,
        **{typedef.name: typedef for typedef in _direct_type_defs(proc, pos)},
    }
    scoped_proc_by_name = {
        **proc_by_name,
        **{child.name: child for child in _nested_procedures(proc, pos)},
    }
    for child in _nested_procedures(proc, pos):
        r = _resolve_in_procedure(
            prog,
            child,
            pos,
            scoped_proc_by_name,
            scoped_type_by_name,
            imported_values,
            ancestor_procs + (proc,),
        )
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
    r = _resolve_in_proc_signature_types(proc, pos, scoped_type_by_name, imported_values)
    if r is not None:
        return r
    for stmt in proc.body:
        r = _resolve_in_stmt(
            prog,
            stmt,
            pos,
            proc,
            scoped_proc_by_name,
            scoped_type_by_name,
            imported_values,
            ancestor_procs,
        )
        if r is not None:
            return r
    return None


def _resolve_in_proc_signature_types(
    proc: ProcedureDef,
    pos: lsp.Position,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
) -> Optional[SymbolRef]:
    """Resolve type names used in a procedure signature."""
    for expr in (proc.param_type_exprs or {}).values():
        r = _resolve_in_type_expr(expr, pos, type_by_name, imported_values)
        if r is not None:
            return r
    return _resolve_in_type_expr(proc.return_type_expr, pos, type_by_name, imported_values)


def _resolve_in_stmt(
    prog: Program | ProgramWithValidation,
    stmt,
    pos: lsp.Position,
    proc: Optional[ProcedureDef],
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
    ancestor_procs: tuple[ProcedureDef, ...] = (),
) -> Optional[SymbolRef]:
    if not hasattr(stmt, "loc") or stmt.loc is None or not _position_inside(pos, stmt.loc):
        return None
    if (
        isinstance(stmt, LoopStmt)
        and stmt.binder_loc is not None
        and _position_inside(pos, stmt.binder_loc)
    ):
        return SymbolRef(
            kind="loop_binder",
            name=stmt.binder,
            declaration_span=stmt.binder_loc,
            scope_proc=proc,
            cursor_span=stmt.binder_loc,
        )
    if isinstance(stmt, (TaskStmt, AwaitStmt)) and stmt.binder_loc is not None:
        if _position_inside(pos, stmt.binder_loc):
            return SymbolRef(
                kind="local" if proc is not None else "module_value",
                name=stmt.binder or "",
                declaration_span=stmt.binder_loc,
                scope_proc=proc,
                cursor_span=stmt.binder_loc,
            )
    if isinstance(stmt, TaskStmt) and stmt.subagent_loc is not None:
        if _position_inside(pos, stmt.subagent_loc):
            profile = _merged_profile_defs(prog, imported_surface_for_program(prog)).get(
                stmt.subagent
            )
            return SymbolRef(
                kind="agent_profile",
                name=stmt.subagent,
                declaration_span=profile.loc if profile is not None else None,
                scope_proc=proc,
                cursor_span=stmt.subagent_loc,
            )
    if isinstance(stmt, AwaitStmt) and stmt.handle_loc is not None:
        if _position_inside(pos, stmt.handle_loc):
            declaration = _resolve_value_declaration(
                prog, proc, ancestor_procs, stmt.handle, stmt.handle_loc
            )
            return SymbolRef(
                kind=declaration.kind if declaration is not None else "local",
                name=stmt.handle,
                declaration_span=declaration.declaration_span if declaration is not None else None,
                scope_proc=declaration.scope_proc if declaration is not None else proc,
                cursor_span=stmt.handle_loc,
            )
    if isinstance(stmt, TypeDefStmt) and stmt.typedef is not None:
        return _resolve_in_type_def(stmt.typedef, pos, type_by_name, imported_values)
    # VariableInitStmt / VariableInitBlockStmt target name
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.loc is not None:
        name_span = binding_name_span(stmt)
        if name_span is not None and _position_inside(pos, name_span):
            return SymbolRef(
                kind=(
                    "module_value"
                    if proc is None and any(stmt is item for item in prog.toplevel_body)
                    else "local"
                ),
                name=stmt.name,
                declaration_span=name_span,
                scope_proc=proc,
                cursor_span=name_span,
            )
    if isinstance(stmt, (AssignmentStmt, AssignmentBlockStmt)) and stmt.loc is not None:
        name_span = binding_name_span(stmt)
        if name_span is not None and _position_inside(pos, name_span):
            declaration = _resolve_value_declaration(
                prog,
                proc,
                ancestor_procs,
                stmt.name,
                name_span,
            )
            if declaration is not None:
                return SymbolRef(
                    kind=declaration.kind,
                    name=stmt.name,
                    declaration_span=declaration.declaration_span,
                    scope_proc=declaration.scope_proc,
                    cursor_span=name_span,
                )
            if stmt.name in imported_values:
                return SymbolRef(
                    kind="imported_value",
                    name=stmt.name,
                    declaration_span=imported_values[stmt.name].declaration_span,
                    scope_proc=proc,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="runtime_value" if stmt.name == "args" else "local",
                name=stmt.name,
                declaration_span=None,
                scope_proc=proc,
                cursor_span=name_span,
            )
    if isinstance(stmt, VariableInitStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name, imported_values)
        if r is not None:
            return r
    if isinstance(stmt, VariableInitBlockStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name, imported_values)
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
    if isinstance(stmt, TaskGroupStmt):
        for arm in stmt.arms:
            if arm.loc is None or not _position_inside(pos, arm.loc):
                continue
            children = [arm.wait, *(arm.body or [])]
            scoped_procs = dict(proc_by_name)
            scoped_types = dict(type_by_name)
            for child in _visible_control_statements(children, pos):
                if isinstance(child, ProcDefStmt) and child.proc is not None:
                    scoped_procs[child.proc.name] = child.proc
                elif isinstance(child, TypeDefStmt) and child.typedef is not None:
                    scoped_types[child.typedef.name] = child.typedef
            for child in children:
                if isinstance(child, ProcDefStmt) and child.proc is not None:
                    result = _resolve_in_procedure(
                        prog,
                        child.proc,
                        pos,
                        scoped_procs,
                        scoped_types,
                        imported_values,
                        (*ancestor_procs, proc) if proc is not None else ancestor_procs,
                    )
                else:
                    result = _resolve_in_stmt(
                        prog,
                        child,
                        pos,
                        proc,
                        scoped_procs,
                        scoped_types,
                        imported_values,
                        ancestor_procs,
                    )
                if result is not None:
                    return result
        return None
    for seg in _segments_of(stmt):
        r = _resolve_in_segment(
            seg,
            pos,
            proc,
            proc_by_name,
            type_by_name,
            imported_values,
            ancestor_procs,
            prog=prog,
        )
        if r is not None:
            return r
    for child in _control_body_statements(stmt):
        r = _resolve_in_stmt(
            prog,
            child,
            pos,
            proc,
            proc_by_name,
            type_by_name,
            imported_values,
            ancestor_procs,
        )
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
    ancestor_procs: tuple[ProcedureDef, ...] = (),
    *,
    prog: Program | ProgramWithValidation | None = None,
) -> Optional[SymbolRef]:
    if not hasattr(seg, "loc") or seg.loc is None or not _position_inside(pos, seg.loc):
        return None
    if isinstance(seg, InputSegment):
        # Could be a parameter, an output, or an assign-target reference.
        name_span = _input_name_span(seg) or seg.loc
        declaration = (
            _resolve_value_declaration(
                prog,
                proc,
                ancestor_procs,
                seg.name,
                name_span,
            )
            if prog is not None
            else None
        )
        if declaration is None and prog is None:
            lexical = _resolve_lexical_name(proc, ancestor_procs, seg.name)
            if lexical is not None:
                kind, declaration_span, owner = lexical
                declaration = _ValueDeclaration(kind, seg.name, declaration_span, owner)
        if declaration is not None:
            return SymbolRef(
                kind=declaration.kind,
                name=seg.name,
                declaration_span=declaration.declaration_span,
                scope_proc=declaration.scope_proc,
                cursor_span=name_span,
            )
        if seg.name in imported_values:
            return SymbolRef(
                kind="imported_value",
                name=seg.name,
                declaration_span=imported_values[seg.name].declaration_span,
                scope_proc=proc,
                cursor_span=name_span,
            )
        if seg.name == "args":
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
            declaration_span=None,
            scope_proc=proc,
            cursor_span=name_span,
        )
    if isinstance(seg, CallSegment):
        name_span = name_token_span(seg.loc, seg.name, prefix_len=1)
        if _position_inside(pos, name_span):
            target = proc_by_name.get(seg.name)
            if target is None and seg.name in imported_values:
                imported_value = imported_values[seg.name]
                return SymbolRef(
                    kind="imported_value",
                    name=seg.name,
                    declaration_span=imported_value.declaration_span,
                    scope_proc=proc,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="procedure",
                name=seg.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=name_token_span(seg.loc, seg.name, prefix_len=1),
            )
        # Recurse into args.

        for arg in seg.args:
            for inner in iter_call_arg_nested(arg):
                r = _resolve_in_segment(
                    inner,
                    pos,
                    proc,
                    proc_by_name,
                    type_by_name,
                    imported_values,
                    ancestor_procs,
                    prog=prog,
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
            r = _resolve_in_type_expr(seg.type_expr, pos, type_by_name, imported_values)
            if r is not None:
                return r
    return None


def _resolve_in_type_def(
    td: TypeDef,
    pos: lsp.Position,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo] | None = None,
) -> Optional[SymbolRef]:
    if td.loc is None or not _position_inside(pos, td.loc):
        return None
    # `~Name(...)` — the name follows the sigil.
    name_span = name_token_span(td.loc, td.name, prefix_len=1)
    if _position_inside(pos, name_span):
        return SymbolRef(
            kind="type",
            name=td.name,
            declaration_span=td.loc,
            scope_proc=None,
            cursor_span=name_token_span(td.loc, td.name, prefix_len=1),
        )
    for f in td.fields:
        r = _resolve_in_type_expr(f.type_expr, pos, type_by_name, imported_values)
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
    expr: Optional[TypeExpr],
    pos: lsp.Position,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo] | None = None,
) -> Optional[SymbolRef]:
    if expr is None or expr.loc is None or not _position_inside(pos, expr.loc):
        return None
    if isinstance(expr, TypeRef):
        target = type_by_name.get(expr.name)
        imported = (imported_values or {}).get(expr.name)
        if target is None and imported is not None and _is_imported_python_class(imported):
            return SymbolRef(
                kind="imported_value",
                name=expr.name,
                declaration_span=imported.declaration_span,
                scope_proc=None,
                cursor_span=expr.loc,
            )
        return SymbolRef(
            kind="type",
            name=expr.name,
            declaration_span=target.loc if target is not None else None,
            scope_proc=None,
            cursor_span=expr.loc,
        )
    if isinstance(expr, TypeApply):
        # `Name[arg,...]` — cursor on Name?
        name_span = name_token_span(expr.loc, expr.name, prefix_len=0)
        on_name = _position_inside(pos, name_span)
        if on_name:
            target = type_by_name.get(expr.name)
            imported = (imported_values or {}).get(expr.name)
            if target is None and imported is not None and _is_imported_python_class(imported):
                return SymbolRef(
                    kind="imported_value",
                    name=expr.name,
                    declaration_span=imported.declaration_span,
                    scope_proc=None,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="type",
                name=expr.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=name_span,
            )
        for arg in expr.args:
            r = _resolve_in_type_expr(arg, pos, type_by_name, imported_values)
            if r is not None:
                return r
    if isinstance(expr, TypeUnion):
        for child in (expr.left, expr.right):
            r = _resolve_in_type_expr(child, pos, type_by_name, imported_values)
            if r is not None:
                return r
    return None
