from __future__ import annotations

import ast as py_ast
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from tree_sitter import Node

from kedi.lang.ast import (
    AssignmentBlockStmt,
    AssignmentStmt,
    BacktickLineStmt,
    CallSegment,
    ConditionalLoopStmt,
    EvalSuite,
    IfStmt,
    LiteralArg,
    LoopStmt,
    ModuleExport,
    ModuleImport,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    Program,
    ProgramWithValidation,
    PythonBlock,
    PythonExprNativeSegment,
    PythonExprSegment,
    RawInvokeStmt,
    ReturnBlockStmt,
    ReturnStmt,
    Segment,
    Span,
    Stmt,
    TemplateLine,
    TestSuite,
    TypeDefStmt,
    TypeExpr,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.parser.cst_helpers import (
    _decode_block_comment_body,
    _field,
    _identifier_name,
    _named_children,
    _src_text,
)
from kedi.lang.parser.diagnostics import KediDiagnostic, KediParseError

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker


def walk_program(self: _Walker) -> Union[Program, ProgramWithValidation]:
    root = self.tree.root_node
    if root.type != "source_file":
        raise KediParseError(
            [
                KediDiagnostic(
                    severity="error",
                    message=f"Expected source_file at root, got {root.type!r}",
                    span=self._span(root),
                )
            ]
        )

    # Pre-pass: structural validation of `###` block comments.
    # Mirrors the legacy `strip_comments` invariants — `###` must
    # occupy a line on its own; unterminated blocks raise; using
    # `###` inside a block comment body raises. The tree-sitter
    # `block_comment` regex is lenient about these conditions, so
    # we enforce them in Python.
    self._validate_block_comments()
    if any(d.severity == "error" for d in self.diagnostics):
        raise KediParseError(self.diagnostics)

    # Incomplete editor input can produce directive-shaped nodes whose
    # required fields are missing. Surface the tree-sitter errors before
    # walking those nodes so live LSP requests never reach AST invariants.
    self._collect_pre_walk_tree_errors(root)
    if any(d.severity == "error" for d in self.diagnostics):
        raise KediParseError(self.diagnostics)

    # First pass — discover whether we need ProgramWithValidation.
    children = _named_children(root)
    has_validation = any(c.type == "validation_block" for c in children)

    prog: Union[Program, ProgramWithValidation]
    if has_validation:
        prog = ProgramWithValidation()
    else:
        prog = Program()
    prog.loc = self._span(root)

    prelude_consumed = False
    pending_else_owner: Optional[IfStmt] = None
    for child in children:
        t = child.type
        if t in ("block_comment", "line_comment"):
            continue
        if t == "else_clause":
            owners: List[Any] = [pending_else_owner] if pending_else_owner is not None else []
            self._attach_else_clause(owners, child, None)
            pending_else_owner = None
            continue
        if t == "map_clause":
            self._attach_map_clause(prog.ordered_toplevel, child, None)
            pending_else_owner = None
            continue
        pending_else_owner = None
        if t == "python_block":
            # The first top-level python_block, IF it appears before
            # any other top-level content, becomes the prelude
            # string. Subsequent python_blocks remain in toplevel_body.
            if (
                not prelude_consumed
                and prog.prelude is None
                and not prog.toplevel_body
                and prog.toplevel_return is None
                and not prog.procedures
                and not prog.type_defs
                and not (isinstance(prog, ProgramWithValidation) and (prog.tests or prog.evals))
            ):
                prog.prelude = self._walk_python_block_code(child)
                prog.prelude_loc = self._span(child)
                prelude_consumed = True
            else:
                stmt = self._walk_python_block(child)
                prog.toplevel_body.append(stmt)
                prog.ordered_toplevel.append(stmt)
        elif t == "procedure_def":
            proc = self._walk_procedure_def(child)
            prog.procedures.append(proc)
            prog.ordered_toplevel.append(proc)
        elif t == "model_directive":
            stmt = self._walk_model_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "agent_directive":
            stmt = self._walk_agent_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "adapter_directive":
            stmt = self._walk_adapter_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "effort_directive":
            stmt = self._walk_effort_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "approval_directive":
            stmt = self._walk_approval_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "hooks_directive":
            stmt = self._walk_hooks_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "history_directive":
            stmt = self._walk_history_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "skills_directive":
            stmt = self._walk_skills_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "codemode_directive":
            stmt = self._walk_codemode_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "system_directive":
            stmt = self._walk_system_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "mcp_directive":
            stmt = self._walk_mcp_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "settings_directive":
            stmt = self._walk_settings_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "artifacts_directive":
            stmt = self._walk_artifacts_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "profile_directive":
            stmt = self._walk_profile_directive(child)
            if stmt.name in prog.profile_defs:
                self._error(
                    child,
                    f"Duplicate profile definition: {stmt.name!r}",
                    code="K-PROFILE-DUP",
                )
            else:
                prog.profile_defs[stmt.name] = stmt
            prog.ordered_toplevel.append(stmt)
        elif t == "use_directive":
            stmt = self._walk_use_directive(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "type_def":
            typedef = self._walk_type_def(child)
            prog.type_defs.append(typedef)
            prog.ordered_toplevel.append(typedef)
        elif t == "module_import":
            module_import = self._walk_module_import(child)
            prog.imports.append(module_import)
            # Imports are compiled eagerly so their public surface is
            # available to declarations, then republished in this stream
            # to preserve source-order binding writes at runtime.
            prog.ordered_toplevel.append(module_import)
        elif t == "module_export":
            prog.exports.append(self._walk_module_export(child))
        elif t == "package_directive":
            if prog.package is not None:
                self._error(
                    child,
                    "a package manifest can define only one package",
                    code="K-PACKAGE-DUP",
                )
            else:
                if self.source_path is not None and Path(self.source_path).name != "package.kedi":
                    self._error(
                        child,
                        "package metadata is only allowed in package.kedi",
                        code="K-PACKAGE-PATH",
                    )
                prog.package = self._walk_package_directive(child)
        elif t == "validation_block":
            assert isinstance(prog, ProgramWithValidation)
            suite = self._walk_validation_block(child)
            if isinstance(suite, TestSuite):
                prog.tests.append(suite)
            elif isinstance(suite, EvalSuite):
                prog.evals.append(suite)
        elif t == "variable_init_stmt":
            stmt = self._walk_variable_init_stmt(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "assignment_stmt":
            stmt = self._walk_assignment_stmt(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "if_stmt":
            stmt = self._walk_if_stmt(child, None)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
            pending_else_owner = stmt
        elif t == "loop_stmt":
            stmt = self._walk_loop_stmt(child, None)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "conditional_loop_stmt":
            stmt = self._walk_conditional_loop_stmt(child, None)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "variable_init_block_stmt":
            stmt = self._walk_variable_init_block_stmt(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "assignment_block_stmt":
            stmt = self._walk_assignment_block_stmt(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "raw_invoke_stmt":
            stmt = self._walk_raw_invoke_stmt(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "return_stmt":
            prog.toplevel_return = self._walk_return_stmt(child)
        elif t == "return_block_stmt":
            prog.toplevel_return = self._walk_return_block_stmt(child)
        elif t == "backtick_line_stmt":
            stmt = self._walk_backtick_line_stmt(child)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "template_block_stmt":
            for stmt in self._walk_template_block_statements(child):
                prog.toplevel_body.append(stmt)
                prog.ordered_toplevel.append(stmt)
        elif t == "template_line":
            stmt = self._walk_template_line(child, deprecated=True)
            prog.toplevel_body.append(stmt)
            prog.ordered_toplevel.append(stmt)
        elif t == "ERROR" or child.is_error:
            self._error(child, self._parse_error_message(child, "unexpected source at top level"))
        else:
            self._error(child, f"unexpected top-level node {t!r}")

    # ProgramWithValidation back-compat: expose the lists via the
    # legacy `_validation_*` attribute names too.
    if isinstance(prog, ProgramWithValidation):
        setattr(prog, "_validation_tests", prog.tests)
        setattr(prog, "_validation_evals", prog.evals)

    if any(d.severity == "error" for d in self.diagnostics):
        raise KediParseError(self.diagnostics)

    prog.parser_warnings = [d for d in self.diagnostics if d.severity == "warning"]
    return prog


def _walk_module_import(self: _Walker, node: Node) -> ModuleImport:
    module_node = _field(node, "module")
    assert module_node is not None
    names: list[str] = []
    name_locs: dict[str, Span] = {}
    body = _field(node, "body")
    if body is not None:
        for child in _named_children(body):
            name_node = _field(child, "name")
            assert name_node is not None
            name = _identifier_name(self.source, name_node)
            if name in name_locs:
                self._error(
                    name_node,
                    f"duplicate selective import name {name!r}",
                    code="K-IMPORT-DUPLICATE",
                )
                continue
            names.append(name)
            name_locs[name] = self._span(name_node)
    return ModuleImport(
        module=_src_text(self.source, module_node),
        names=names,
        name_locs=name_locs,
        loc=self._span(node),
    )


def _walk_module_export(self: _Walker, node: Node) -> ModuleExport:
    export_all = "*" in _src_text(self.source, node).split(":", 1)[1].splitlines()[0]
    names: list[str] = []
    body = _field(node, "body")
    if body is not None:
        for child in _named_children(body):
            if child.type != "module_export_name":
                continue
            name_node = _field(child, "name")
            if name_node is not None:
                names.append(_identifier_name(self.source, name_node))
    return ModuleExport(names=names, export_all=export_all, loc=self._span(node))


def _walk_procedure_def(self: _Walker, node: Node) -> ProcedureDef:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)

    params: List[str] = []
    param_type_annos: dict[str, Optional[str]] = {}
    param_type_exprs: dict[str, Optional[TypeExpr]] = {}
    param_defaults: dict[str, str] = {}
    seen_params: set[str] = set()
    seen_default = False
    any_typed = False
    param_list = _field(node, "params")
    if param_list is not None:
        for param in _named_children(param_list):
            if param.type != "param":
                continue
            pname_node = _field(param, "name")
            assert pname_node is not None
            pname = _identifier_name(self.source, pname_node)
            if pname in seen_params:
                self._error(
                    param,
                    f"duplicate parameter {pname!r} in procedure {name!r}",
                    code="K-PARAM-DUP",
                )
            seen_params.add(pname)
            params.append(pname)
            ptype_node = _field(param, "type")
            if ptype_node is not None:
                expr = self._walk_type_expr(ptype_node)
                param_type_annos[pname] = self._type_expr_to_str(ptype_node)
                param_type_exprs[pname] = expr
                any_typed = True
            else:
                param_type_annos[pname] = None
                param_type_exprs[pname] = None
            default_node = _field(param, "default")
            if default_node is not None:
                seen_default = True
                param_defaults[pname] = self._walk_inline_python_expr(default_node)
            elif seen_default:
                self._error(
                    param,
                    "required parameter cannot follow a defaulted parameter",
                    code="K-PARAM-ORDER",
                )

    return_type_node = _field(node, "return_type")
    return_type_anno: Optional[str] = None
    return_type_expr: Optional[TypeExpr] = None
    if return_type_node is not None:
        return_type_anno = self._type_expr_to_str(return_type_node)
        return_type_expr = self._walk_type_expr(return_type_node)

    proc = ProcedureDef(
        name=name,
        params=params,
        body=[],
        return_type_annotation=return_type_anno,
        return_type_expr=return_type_expr,
        param_type_annotations=param_type_annos if any_typed or params else None,
        param_type_exprs=param_type_exprs if any_typed or params else None,
        param_defaults=param_defaults or None,
        loc=self._span(node),
    )

    body_node = _field(node, "body")
    if body_node is not None:
        first_named = _named_children(body_node)
        if first_named and first_named[0].type == "block_comment":
            comment = first_named[0]
            proc.docstring = _decode_block_comment_body(_src_text(self.source, comment))
            proc.docstring_loc = self._span(comment)
        self._walk_block_into(body_node, proc)
    return proc


def _walk_block_into(self: _Walker, block_node: Node, proc: ProcedureDef) -> None:
    proc.body.extend(self._walk_block_statements(block_node, proc))


def _walk_block_statements(
    self: _Walker, block_node: Node, proc: Optional[ProcedureDef]
) -> List[Stmt]:
    statements: List[Stmt] = []
    pending_else_owner: Optional[IfStmt] = None
    for child in _named_children(block_node):
        if child.type in ("block_comment", "line_comment"):
            continue
        if child.type == "else_clause":
            owners: List[Any] = [pending_else_owner] if pending_else_owner is not None else []
            self._attach_else_clause(owners, child, proc)
            pending_else_owner = None
            continue
        if child.type == "map_clause":
            self._attach_map_clause(statements, child, proc)
            pending_else_owner = None
            continue
        pending_else_owner = None
        if child.type == "template_block_stmt":
            statements.extend(self._walk_template_block_statements(child))
            continue
        stmt = self._walk_stmt(child, proc)
        if stmt is None:
            continue
        statements.append(stmt)
        if isinstance(stmt, IfStmt):
            pending_else_owner = stmt
    return statements


def _walk_stmt(self: _Walker, node: Node, proc: Optional[ProcedureDef]) -> Optional[Stmt]:
    t = node.type
    if t == "procedure_def":
        nested = self._walk_procedure_def(node)
        return ProcDefStmt(proc=nested, loc=self._span(node))
    if t == "type_def_stmt":
        inner = _named_children(node)
        if inner and inner[0].type == "type_def":
            td = self._walk_type_def(inner[0])
            return TypeDefStmt(typedef=td, loc=self._span(node))
        self._error(node, "type_def_stmt missing inner type_def")
        return None
    if t == "auto_directive":
        if proc is None:
            self._error(node, "'> auto:' is only valid inside a procedure body")
            return None
        proc.auto_spec = self._walk_auto_directive(node)
        return None
    if t == "optimize_directive":
        if proc is None:
            self._error(node, "'> optimize:' is only valid inside a procedure body")
            return None
        span, block = self._walk_optimize_directive(node)
        proc.optimize_spans.append(span)
        return block
    if t == "model_directive":
        return self._walk_model_directive(node)
    if t == "agent_directive":
        return self._walk_agent_directive(node)
    if t == "adapter_directive":
        return self._walk_adapter_directive(node)
    if t == "effort_directive":
        return self._walk_effort_directive(node)
    if t == "approval_directive":
        return self._walk_approval_directive(node)
    if t == "hooks_directive":
        return self._walk_hooks_directive(node)
    if t == "history_directive":
        return self._walk_history_directive(node)
    if t == "skills_directive":
        return self._walk_skills_directive(node)
    if t == "codemode_directive":
        return self._walk_codemode_directive(node)
    if t == "system_directive":
        return self._walk_system_directive(node)
    if t == "mcp_directive":
        return self._walk_mcp_directive(node)
    if t == "settings_directive":
        return self._walk_settings_directive(node)
    if t == "artifacts_directive":
        return self._walk_artifacts_directive(node)
    if t == "use_directive":
        return self._walk_use_directive(node)
    if t == "profile_directive":
        self._error(node, "'> profile:' is only valid at top level", code="K-PROFILE-SCOPE")
        return None
    if t == "variable_init_stmt":
        return self._walk_variable_init_stmt(node)
    if t == "assignment_stmt":
        return self._walk_assignment_stmt(node)
    if t == "if_stmt":
        return self._walk_if_stmt(node, proc)
    if t == "loop_stmt":
        return self._walk_loop_stmt(node, proc)
    if t == "conditional_loop_stmt":
        return self._walk_conditional_loop_stmt(node, proc)
    if t == "variable_init_block_stmt":
        return self._walk_variable_init_block_stmt(node)
    if t == "assignment_block_stmt":
        return self._walk_assignment_block_stmt(node)
    if t == "raw_invoke_stmt":
        return self._walk_raw_invoke_stmt(node)
    if t == "return_stmt":
        return self._walk_return_stmt(node)
    if t == "return_block_stmt":
        return self._walk_return_block_stmt(node)
    if t == "python_block":
        return self._walk_python_block(node)
    if t == "backtick_line_stmt":
        return self._walk_backtick_line_stmt(node)
    if t == "template_block_stmt":
        block = self._walk_template_block_stmt(node)
        self._validate_template_block_scope(block.lines)
        return block
    if t == "template_line":
        return self._walk_template_line(node, deprecated=True)
    if t in ("block_comment", "line_comment"):
        return None
    if t == "ERROR" or node.is_error:
        # Probe the raw text of the ERROR to give callers a better
        # message for the common malformed-directive cases — in
        # particular preserving the legacy ``"must have an indented
        # block"`` substring that several tests assert on.
        raw = _src_text(self.source, node).strip()
        if raw.startswith("> optimize"):
            m = raw.split(":", 2)
            name = m[1].strip() if len(m) > 1 else ""
            self._error(
                node,
                f"Optimize span {name!r} must have an indented block after it",
                code="K-OPT-EMPTY",
            )
        else:
            self._error(node, self._parse_error_message(node, f"unexpected statement node {t!r}"))
        return None
    # Anything else is an unexpected node — surface a diagnostic
    # but keep parsing the rest of the body.
    self._error(node, f"unexpected statement node {t!r}")
    return None


def _walk_if_stmt(self: _Walker, node: Node, proc: Optional[ProcedureDef]) -> IfStmt:
    condition = _field(node, "condition")
    consequence = _field(node, "consequence")
    assert condition is not None
    assert consequence is not None

    condition_code, condition_segments = self._walk_control_condition(condition)
    return IfStmt(
        condition_code=condition_code,
        condition_segments=condition_segments,
        body=self._walk_block_statements(consequence, proc),
        else_body=None,
        condition_loc=self._span(condition),
        loc=self._span(node),
    )


def _attach_else_clause(
    self: _Walker,
    statements: List[Any],
    node: Node,
    proc: Optional[ProcedureDef],
) -> None:
    if not statements or not isinstance(statements[-1], IfStmt):
        self._error(node, "'> else:' must immediately follow an if block", code="K-ELSE-ORPHAN")
        return

    owner = statements[-1]
    if owner.else_body is not None:
        self._error(node, "an if block can have only one else block", code="K-ELSE-DUP")
        return

    body = _field(node, "body")
    assert body is not None
    owner.else_body = self._walk_block_statements(body, proc)

    end = self._span(node)
    start = owner.loc
    if start is not None:
        owner.loc = Span(
            start_byte=start.start_byte,
            end_byte=end.end_byte,
            start_line=start.start_line,
            start_col=start.start_col,
            end_line=end.end_line,
            end_col=end.end_col,
            source_path=start.source_path,
            source_id=start.source_id,
        )


def _attach_map_clause(
    self: _Walker,
    statements: List[Any],
    node: Node,
    proc: Optional[ProcedureDef],
) -> None:
    if not statements or not isinstance(statements[-1], LoopStmt):
        self._error(
            node,
            "'> map:' must immediately follow a binder loop",
            code="K-MAP-ORPHAN",
        )
        return

    owner = statements[-1]
    if owner.map_body is not None:
        self._error(node, "a loop can have only one map stage", code="K-MAP-DUP")
        return

    body = _field(node, "body")
    assert body is not None
    owner.map_body = self._walk_block_statements(body, proc)
    owner.map_loc = self._span(node)

    end = owner.map_loc
    start = owner.loc
    assert start is not None
    owner.loc = Span(
        start_byte=start.start_byte,
        end_byte=end.end_byte,
        start_line=start.start_line,
        start_col=start.start_col,
        end_line=end.end_line,
        end_col=end.end_col,
        source_path=start.source_path,
        source_id=start.source_id,
    )


def _walk_loop_stmt(self: _Walker, node: Node, proc: Optional[ProcedureDef]) -> LoopStmt:
    binder = _field(node, "binder")
    iterable = _field(node, "iterable")
    body = _field(node, "body")
    assert binder is not None
    assert iterable is not None
    assert body is not None
    return LoopStmt(
        binder=_identifier_name(self.source, binder),
        iterable_code=self._walk_inline_python_expr(iterable),
        body=self._walk_block_statements(body, proc),
        binder_loc=self._span(binder),
        iterable_loc=self._span(iterable),
        loc=self._span(node),
    )


def _walk_conditional_loop_stmt(
    self: _Walker,
    node: Node,
    proc: Optional[ProcedureDef],
) -> ConditionalLoopStmt:
    condition = _field(node, "condition")
    body = _field(node, "body")
    assert condition is not None
    assert body is not None
    condition_code, condition_segments = self._walk_control_condition(condition)
    return ConditionalLoopStmt(
        condition_code=condition_code,
        condition_segments=condition_segments,
        body=self._walk_block_statements(body, proc),
        condition_loc=self._span(condition),
        loc=self._span(node),
    )


def _walk_control_condition(self: _Walker, node: Node) -> tuple[str, Optional[List[Segment]]]:
    if node.type == "inline_python_expr":
        return self._walk_inline_python_expr(node), None

    segments = self._walk_template_expr(node)
    for output in self._condition_outputs(segments):
        self.diagnostics.append(
            KediDiagnostic(
                severity="error",
                message=(
                    "Template conditions cannot declare output fields; use plain text, "
                    "<substitutions>, or inline Python values"
                ),
                span=output.loc or self._span(node),
                code="K-CONDITION-OUTPUT",
            )
        )
    return "", segments


def _condition_outputs(segments: List[Segment]) -> List[OutputSegment]:
    outputs: List[OutputSegment] = []

    def visit(segment: Segment) -> None:
        if isinstance(segment, OutputSegment):
            outputs.append(segment)
            return
        if not isinstance(segment, CallSegment):
            return
        for argument in segment.args:
            if not isinstance(argument, LiteralArg):
                continue
            for nested in argument.render_segments:
                visit(nested)
            if argument.call_segment is not None:
                visit(argument.call_segment)

    for segment in segments:
        visit(segment)
    return outputs


def _walk_variable_init_stmt(self: _Walker, node: Node) -> VariableInitStmt:
    target = _named_children(node)[0]
    assert target.type == "binding_target"
    name_node = _field(target, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    type_node = _field(target, "type")
    type_anno = self._type_expr_to_str(type_node) if type_node is not None else None
    type_expr = self._walk_type_expr(type_node) if type_node is not None else None

    rhs_node = _field(node, "rhs")
    # A sole backtick RHS preserves its native Python value. The
    # tree-sitter grammar also admits inline expressions as template
    # segments, so accept either CST shape here without changing the
    # long-standing native initialization semantics.
    native_expr = self._sole_inline_python_expr(rhs_node)
    if native_expr is not None:
        code = self._walk_inline_python_expr(native_expr)
        seg = PythonExprNativeSegment(code=code, loc=self._span(native_expr))
        return VariableInitStmt(
            name=name,
            expr=[seg],
            type_annotation=type_anno,
            type_expr=type_expr,
            loc=self._span(node),
        )
    # template_expr
    segs = self._walk_template_expr(rhs_node) if rhs_node is not None else []
    return VariableInitStmt(
        name=name,
        expr=segs,
        type_annotation=type_anno,
        type_expr=type_expr,
        loc=self._span(node),
    )


def _walk_variable_init_block_stmt(self: _Walker, node: Node) -> VariableInitBlockStmt:
    target = _named_children(node)[0]
    name_node = _field(target, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    type_node = _field(target, "type")
    type_anno = self._type_expr_to_str(type_node) if type_node is not None else None
    type_expr = self._walk_type_expr(type_node) if type_node is not None else None
    code_node = _field(node, "code")
    code = self._dedent_fenced(code_node) if code_node is not None else ""
    return VariableInitBlockStmt(
        name=name,
        code=code,
        type_annotation=type_anno,
        type_expr=type_expr,
        loc=self._span(node),
    )


def _walk_assignment_stmt(self: _Walker, node: Node) -> AssignmentStmt:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    rhs_node = _field(node, "rhs")
    native_expr = self._sole_inline_python_expr(rhs_node)
    if native_expr is not None:
        code = self._walk_inline_python_expr(native_expr)
        segments: List[Segment] = [PythonExprNativeSegment(code=code, loc=self._span(native_expr))]
    else:
        segments = self._walk_template_expr(rhs_node) if rhs_node is not None else []
    return AssignmentStmt(name=name, expr=segments, loc=self._span(node))


def _walk_assignment_block_stmt(self: _Walker, node: Node) -> AssignmentBlockStmt:
    name_node = _field(node, "name")
    assert name_node is not None
    code_node = _field(node, "code")
    return AssignmentBlockStmt(
        name=_identifier_name(self.source, name_node),
        code=self._dedent_fenced(code_node) if code_node is not None else "",
        loc=self._span(node),
    )


def _walk_raw_invoke_stmt(self: _Walker, node: Node) -> RawInvokeStmt:
    target = _field(node, "target")
    assert target is not None and target.type == "binding_target"
    name_node = _field(target, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    type_node = _field(target, "type")
    type_anno = self._type_expr_to_str(type_node) if type_node is not None else None
    type_expr = self._walk_type_expr(type_node) if type_node is not None else None

    if type_node is not None:
        if type_anno == "str":
            self._warning(
                type_node,
                "Raw invoke captures always produce string values; ': str' is redundant.",
                code="K-RAW-INVOKE-REDUNDANT-STR",
            )
        else:
            self._error(
                type_node,
                "Raw invoke captures always produce string values.",
                code="K-RAW-INVOKE-TYPE",
            )

    prompt_node = _field(node, "prompt")
    segments = self._walk_template_expr(prompt_node) if prompt_node is not None else []
    for segment in segments:
        if isinstance(segment, OutputSegment):
            self.diagnostics.append(
                KediDiagnostic(
                    severity="error",
                    message="Raw invoke prompts cannot contain output fields.",
                    span=segment.loc or self._span(prompt_node or node),
                    code="K-RAW-INVOKE-OUTPUT",
                )
            )

    return RawInvokeStmt(
        name=name,
        segments=segments,
        type_annotation=type_anno,
        type_expr=type_expr,
        loc=self._span(node),
    )


def _walk_return_stmt(self: _Walker, node: Node) -> Union[ReturnStmt, ReturnBlockStmt]:
    value_node = _field(node, "value")
    # Legacy invariant: `= \`code\`` (sole single-backtick) is a
    # ``ReturnBlockStmt`` so the runtime preserves the native
    # Python value (e.g. ``int``) rather than rendering it to a
    # string. The branch must distinguish ``= \`code\``` from a
    # ``ReturnStmt`` whose value happens to include a backtick
    # segment alongside other content (which IS a ReturnStmt with
    # a single PythonExprNativeSegment, rendered to string).
    native_expr = self._sole_inline_python_expr(value_node)
    if native_expr is not None:
        code = self._walk_inline_python_expr(native_expr)
        return ReturnBlockStmt(code=code, loc=self._span(node))
    segs = self._walk_template_expr(value_node) if value_node is not None else []
    return ReturnStmt(segments=segs, loc=self._span(node))


def _walk_return_block_stmt(self: _Walker, node: Node) -> ReturnBlockStmt:
    code_node = _field(node, "code")
    code = self._dedent_fenced(code_node) if code_node is not None else ""
    return ReturnBlockStmt(code=code, loc=self._span(node))


def _walk_python_block(self: _Walker, node: Node) -> PythonBlock:
    return PythonBlock(code=self._walk_python_block_code(node), loc=self._span(node))


def _walk_python_block_code(self: _Walker, node: Node) -> str:
    code_node = _field(node, "code")
    return self._dedent_fenced(code_node) if code_node is not None else ""


def _walk_backtick_line_stmt(self: _Walker, node: Node) -> BacktickLineStmt:
    # The single child is an inline_python_expr.
    for child in _named_children(node):
        if child.type == "inline_python_expr":
            code = self._walk_inline_python_expr(child)
            return BacktickLineStmt(code=code, loc=self._span(node))
    self._error(node, "backtick_line_stmt missing inline_python_expr child")
    return BacktickLineStmt(code="", loc=self._span(node))


def _standalone_backtick_continuation_code(line: TemplateLine) -> str | None:
    """Return code when a continuation is a Python statement, not an expression."""

    if len(line.segments) != 1 or not isinstance(line.segments[0], PythonExprSegment):
        return None
    code = line.segments[0].code
    try:
        parsed = py_ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    if len(parsed.body) != 1 or isinstance(parsed.body[0], py_ast.Expr):
        return None
    return code
