"""Walk a tree-sitter Kedi CST and produce typed Python AST dataclasses.

This module is the bridge between the tree-sitter parser (which produces
a concrete syntax tree of opaque ``tree_sitter.Node`` objects) and the
Kedi AST defined in :mod:`kedi.lang.ast`.

Design notes:

  - Every produced AST node carries a ``Span`` populated from its CST
    counterpart's ``start_byte`` / ``end_byte`` / ``start_point`` /
    ``end_point``.
  - Type annotations are produced in two forms: a structured
    ``TypeExpr`` (the new representation; populated on the
    ``type_expr`` field) and the user's raw source slice as a string
    (kept on the legacy ``type_annotation`` field for compatibility
    with the compiler / codegen / optimizer modules that have not
    yet migrated). The two forms agree by construction.
  - ``parse_program`` returns ``ProgramWithValidation`` whenever any
    ``@test:`` or ``@eval:`` block is present in the source. For
    backward compatibility with read sites that still use the legacy
    ``setattr``-based access pattern (``getattr(prog,
    "_validation_evals", [])``), the same lists are also exposed as
    ``_validation_tests`` and ``_validation_evals`` attributes.

The walker owns source and diagnostic state, dispatching CST conversion
to the specialized converter modules through its ``_walk_<rule>`` methods.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from tree_sitter import Node, Tree

from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    ArtifactsField,
    AssignmentBlockStmt,
    AssignmentStmt,
    AutoSpec,
    BacktickLineStmt,
    BudgetDirectiveStmt,
    CallArg,
    CallSegment,
    CodeModeDirectiveStmt,
    ConditionalLoopStmt,
    EffortDirectiveStmt,
    EvalDataset,
    EvalMetric,
    EvalSuite,
    GroupStmt,
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    IfStmt,
    LoopStmt,
    McpDirectiveStmt,
    McpField,
    ModelDirectiveStmt,
    ModuleExport,
    ModuleImport,
    OptimizeSpan,
    OutputSegment,
    PackageMetadata,
    ProcedureDef,
    ProfileDefStmt,
    Program,
    ProgramWithValidation,
    PythonBlock,
    RawInvokeStmt,
    RequiresDirectiveStmt,
    ReturnBlockStmt,
    ReturnStmt,
    Segment,
    SettingsDirectiveStmt,
    SettingsField,
    SkillsDirectiveStmt,
    Span,
    Stmt,
    SystemDirectiveStmt,
    TemplateBlock,
    TemplateLine,
    TestCase,
    TestSuite,
    ToolDeclaration,
    TypeDef,
    TypeExpr,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.parser import cst_helpers as _cst_helpers
from kedi.lang.parser import directive_conversion as _directive_conversion
from kedi.lang.parser import profile_conversion as _profile_conversion
from kedi.lang.parser import statement_conversion as _statement_conversion
from kedi.lang.parser import template_conversion as _template_conversion
from kedi.lang.parser import type_conversion as _type_conversion
from kedi.lang.parser import validation_conversion as _validation_conversion
from kedi.lang.parser.cst_helpers import _span
from kedi.lang.parser.diagnostics import KediDiagnostic
from kedi.source_loading import CompositeSourceMap


class _Walker:
    def __init__(
        self,
        source: bytes,
        tree: Tree,
        source_path: Optional[str],
        source_map: CompositeSourceMap | None,
    ):
        self.source = source
        self.tree = tree
        self.source_path = source_path
        self.source_map = source_map
        self.diagnostics: List[KediDiagnostic] = []
        self.groups: dict[str, GroupStmt] = {}

    # ----- entry point -----
    def walk_program(self) -> Union[Program, ProgramWithValidation]:
        return _statement_conversion.walk_program(self)

    # ----- diagnostics helpers -----

    def _span(self, node: Node) -> Span:
        return _span(node, self.source_path, self.source_map)

    def _error(self, node: Node, message: str, code: Optional[str] = None) -> None:
        self.diagnostics.append(
            KediDiagnostic(
                severity="error",
                message=message,
                span=self._span(node),
                code=code,
            )
        )

    def _warning(self, node: Node, message: str, code: Optional[str] = None) -> None:
        self.diagnostics.append(
            KediDiagnostic(
                severity="warning",
                message=message,
                span=self._span(node),
                code=code,
            )
        )

    def _unescape_text_segment(self, node: Node, raw: str) -> tuple[str, set[int]]:
        return _template_conversion._unescape_text_segment(self, node, raw)

    def _validate_effort_plain_value(self, node: Node, value: str) -> None:
        return _directive_conversion._validate_effort_plain_value(self, node, value)

    def _validate_approval_plain_value(self, node: Node, value: str) -> None:
        return _directive_conversion._validate_approval_plain_value(self, node, value)

    def _validate_mcp_transport_plain_value(self, node: Node, value: str) -> None:
        return _directive_conversion._validate_mcp_transport_plain_value(self, node, value)

    def _validate_setting_key(self, node: Node, name: str) -> None:
        return _directive_conversion._validate_setting_key(self, node, name)

    def _validate_block_comments(self) -> None:
        r"""Enforce legacy `###` block-comment invariants.

        Rules (matching ``kedi.lang.parser.preprocessing.strip_comments``):

          - A line that is exactly ``###`` (after stripping horizontal
            whitespace) toggles block-comment mode.
          - Inside a block comment, any occurrence of ``###`` (other
            than the closing fence) is invalid.
          - Outside a block comment, ``###`` MUST occupy an entire
            line — substrings like ``= text ###`` or ``### content``
            are rejected.
          - Triple-backtick fenced Python blocks (``\`\`\``) and their
            contents are exempt: comments inside Python source are
            normal Python comments, not Kedi block fences.
          - On EOF, an open block raises "Unterminated block comment".
        """
        return _cst_helpers._validate_block_comments(self)

    def _line_error(self, line_index: int, message: str, *, code: Optional[str] = None) -> None:
        """Record a diagnostic at a specific (0-based) line, full-line span."""
        return _cst_helpers._line_error(self, line_index, message, code=code)

    def _record_tree_error(self, node: Node) -> None:
        """Convert one ERROR / MISSING node into a parser diagnostic."""
        return _cst_helpers._record_tree_error(self, node)

    def _collect_pre_walk_tree_errors(
        self,
        node: Node,
        parent_type: str | None = None,
    ) -> None:
        """Catch structurally unsafe CST errors before AST field access."""
        return _cst_helpers._collect_pre_walk_tree_errors(self, node, parent_type)

    def _parse_error_message(self, node: Node, fallback: str) -> str:
        return _cst_helpers._parse_error_message(self, node, fallback)

    # =================================================================
    # Module directives
    # =================================================================

    def _walk_package_directive(self, node: Node) -> PackageMetadata:
        return _directive_conversion._walk_package_directive(self, node)

    def _walk_module_import(self, node: Node) -> ModuleImport:
        return _statement_conversion._walk_module_import(self, node)

    def _walk_module_export(self, node: Node) -> ModuleExport:
        return _statement_conversion._walk_module_export(self, node)

    # =================================================================
    # Procedures
    # =================================================================

    def _walk_procedure_def(self, node: Node) -> ProcedureDef:
        return _statement_conversion._walk_procedure_def(self, node)

    def _walk_block_into(self, block_node: Node, proc: ProcedureDef) -> None:
        return _statement_conversion._walk_block_into(self, block_node, proc)

    def _walk_block_statements(
        self,
        block_node: Node,
        proc: Optional[ProcedureDef],
        *,
        procedure_header: bool = False,
    ) -> List[Stmt]:
        return _statement_conversion._walk_block_statements(
            self,
            block_node,
            proc,
            procedure_header=procedure_header,
        )

    # =================================================================
    # Statements
    # =================================================================

    def _walk_stmt(self, node: Node, proc: Optional[ProcedureDef]) -> Optional[Stmt]:
        return _statement_conversion._walk_stmt(self, node, proc)

    def _walk_if_stmt(self, node: Node, proc: Optional[ProcedureDef]) -> IfStmt:
        return _statement_conversion._walk_if_stmt(self, node, proc)

    def _attach_else_clause(
        self,
        statements: List[Any],
        node: Node,
        proc: Optional[ProcedureDef],
    ) -> None:
        return _statement_conversion._attach_else_clause(self, statements, node, proc)

    def _attach_map_clause(
        self,
        statements: List[Any],
        node: Node,
        proc: Optional[ProcedureDef],
    ) -> None:
        return _statement_conversion._attach_map_clause(self, statements, node, proc)

    def _walk_loop_stmt(self, node: Node, proc: Optional[ProcedureDef]) -> LoopStmt:
        return _statement_conversion._walk_loop_stmt(self, node, proc)

    def _walk_conditional_loop_stmt(
        self,
        node: Node,
        proc: Optional[ProcedureDef],
    ) -> ConditionalLoopStmt:
        return _statement_conversion._walk_conditional_loop_stmt(self, node, proc)

    def _walk_control_condition(self, node: Node) -> tuple[str, Optional[List[Segment]]]:
        return _statement_conversion._walk_control_condition(self, node)

    @staticmethod
    def _condition_outputs(segments: List[Segment]) -> List[OutputSegment]:
        return _statement_conversion._condition_outputs(segments)

    def _walk_variable_init_stmt(self, node: Node) -> VariableInitStmt:
        return _statement_conversion._walk_variable_init_stmt(self, node)

    def _walk_variable_init_block_stmt(self, node: Node) -> VariableInitBlockStmt:
        return _statement_conversion._walk_variable_init_block_stmt(self, node)

    def _walk_assignment_stmt(self, node: Node) -> AssignmentStmt:
        return _statement_conversion._walk_assignment_stmt(self, node)

    def _walk_assignment_block_stmt(self, node: Node) -> AssignmentBlockStmt:
        return _statement_conversion._walk_assignment_block_stmt(self, node)

    def _walk_raw_invoke_stmt(self, node: Node) -> RawInvokeStmt:
        return _statement_conversion._walk_raw_invoke_stmt(self, node)

    def _walk_return_stmt(self, node: Node) -> Union[ReturnStmt, ReturnBlockStmt]:
        return _statement_conversion._walk_return_stmt(self, node)

    def _walk_return_block_stmt(self, node: Node) -> ReturnBlockStmt:
        return _statement_conversion._walk_return_block_stmt(self, node)

    def _walk_python_block(self, node: Node) -> PythonBlock:
        return _statement_conversion._walk_python_block(self, node)

    def _walk_python_block_code(self, node: Node) -> str:
        return _statement_conversion._walk_python_block_code(self, node)

    def _walk_backtick_line_stmt(self, node: Node) -> BacktickLineStmt:
        # The single child is an inline_python_expr.
        return _statement_conversion._walk_backtick_line_stmt(self, node)

    def _template_line_from_expr(self, expr_node: Node) -> TemplateLine:
        return _template_conversion._template_line_from_expr(self, expr_node)

    def _system_line_from_expr(self, expr_node: Node) -> TemplateLine:
        return _template_conversion._system_line_from_expr(self, expr_node)

    def _walk_template_block_stmt(self, node: Node) -> TemplateBlock:
        return _template_conversion._walk_template_block_stmt(self, node)

    def _walk_template_block_statements(self, node: Node) -> List[Stmt]:
        """Split legacy backtick statements out of a ``>>`` continuation.

        A bare inline Python expression is a valid prompt segment after ``>>``.
        Historically, however, a continuation containing a Python *statement*
        (for example ``value = value.strip()``) remained a standalone
        backtick statement. Tree-sitter deliberately parses both forms as a
        template continuation so plain expressions can stay in the block; this
        AST pass restores the statement boundary without restricting Python
        expression syntax in the grammar.
        """
        return _template_conversion._walk_template_block_statements(self, node)

    def _validated_template_block(
        self, lines: List[TemplateLine], loc: Span | None
    ) -> TemplateBlock:
        return _template_conversion._validated_template_block(self, lines, loc)

    @staticmethod
    def _standalone_backtick_continuation_code(line: TemplateLine) -> str | None:
        """Return code when a continuation is a Python statement, not an expression."""
        return _statement_conversion._standalone_backtick_continuation_code(line)

    def _validate_template_block_scope(self, lines: List[TemplateLine]) -> None:
        return _template_conversion._validate_template_block_scope(self, lines)

    def _walk_template_line(self, node: Node, *, deprecated: bool = False) -> TemplateLine:
        return _template_conversion._walk_template_line(self, node, deprecated=deprecated)

    # =================================================================
    # Type expressions
    # =================================================================

    def _walk_type_expr(self, node: Node) -> Optional[TypeExpr]:
        # `type_expr` is a thin wrapper; its sole named child is the
        # actual term (type_ref / type_apply / type_union / type_python).
        return _type_conversion._walk_type_expr(self, node)

    def _walk_type_arg(self, node: Node) -> Optional[TypeExpr]:
        """Walk a ``type_arg`` wrapper inside generic type applications."""
        return _type_conversion._walk_type_arg(self, node)

    def _walk_type_term(self, node: Node) -> Optional[TypeExpr]:
        """Walk a node that is a `_type_expr_term` alternative (no `type_expr` wrapper)."""
        return _type_conversion._walk_type_term(self, node)

    def _type_expr_to_str(self, node: Node) -> str:
        """Render a type_expr / inner term back to its source slice.

        Used to populate the legacy ``type_annotation`` string field for
        consumers (compiler / codegen / optimizer) that have not yet
        migrated to the structured ``type_expr`` form. Returning the
        raw source slice means the rendering is byte-perfect — including
        the backticks for ``type_python`` cases, matching the legacy
        ``KediType.resolve`` expectation.
        """
        return _type_conversion._type_expr_to_str(self, node)

    # =================================================================
    # Type definitions
    # =================================================================

    def _walk_type_def(self, node: Node) -> TypeDef:
        return _type_conversion._walk_type_def(self, node)

    # =================================================================
    # Template segments
    # =================================================================

    def _walk_template_expr(self, node: Node) -> List[Segment]:
        return _template_conversion._walk_template_expr(self, node)

    @staticmethod
    def _sole_inline_python_expr(node: Node | None) -> Node | None:
        """Return the sole inline expression in an initialization, assignment, or return RHS."""
        return _template_conversion._sole_inline_python_expr(node)

    def _strip_leading_template_indent(self, segs: List[Segment]) -> None:
        # The scanner's text-token rule absorbs any leading horizontal
        # whitespace of a body line (because tree-sitter cannot
        # interleave extras with a retried external-scan). Template body
        # lines therefore carry that leading whitespace in TextSegments;
        # lstripping the leading WS off the FIRST segment of each
        # template_expr restores the expected shape when it's a plain
        # TextSegment. Internal-segment whitespace
        # (the gap between an `<input>` and the text that follows it)
        # is preserved.
        return _template_conversion._strip_leading_template_indent(self, segs)

    def _walk_segment(self, node: Node) -> Segment:
        return _template_conversion._walk_segment(self, node)

    def _walk_call_segment(self, node: Node) -> CallSegment:
        return _template_conversion._walk_call_segment(self, node)

    def _walk_call_arg(self, node: Node) -> CallArg:
        r"""Build a call argument with explicit payload fields.

        A literal argument can carry one of these payloads:

          - ``native_backtick``: arg passes a native Python value
            (from a bare backtick `` `expr` ``).
          - ``python_segment``: arg is a single ``<\`...\`>`` segment.
          - ``call_segment``: arg is a single ``<func(...)>`` segment.
          - ``render_segments``: arg is a mini-template rendered to a string.
        """
        return _template_conversion._walk_call_arg(self, node)

    def _walk_segment_in_call(self, node: Node) -> Segment:
        """A segment appearing inside a ``template_arg`` (call argument).

        Each call-argument text segment is BOTH the first AND the last
        text of its enclosing arg expression in the legacy semantics —
        so we trim both ends here (matching ``parse_call_arguments``,
        which calls ``raw.strip()`` on the arg before segment-parsing
        it). Internal whitespace is preserved.
        """
        return _template_conversion._walk_segment_in_call(self, node)

    # =================================================================
    # Inline python expression body
    # =================================================================

    def _walk_inline_python_expr(self, node: Node) -> str:
        return _template_conversion._walk_inline_python_expr(self, node)

    # =================================================================
    # Fenced code dedenting
    # =================================================================

    def _dedent_fenced(self, code_node: Node) -> str:
        """Extract and dedent a python_code node's body."""
        return _template_conversion._dedent_fenced(self, code_node)

    # =================================================================
    # Directives
    # =================================================================

    def _walk_auto_directive(self, node: Node) -> AutoSpec:
        return _directive_conversion._walk_auto_directive(self, node)

    def _walk_tool_directive(self, node: Node) -> ToolDeclaration:
        return _directive_conversion._walk_tool_directive(self, node)

    def _walk_requires_directive(self, node: Node) -> RequiresDirectiveStmt:
        return _directive_conversion._walk_requires_directive(self, node)

    def _walk_budget_directive(self, node: Node) -> BudgetDirectiveStmt:
        return _directive_conversion._walk_budget_directive(self, node)

    def _walk_model_directive(self, node: Node) -> ModelDirectiveStmt:
        return _directive_conversion._walk_model_directive(self, node)

    def _walk_agent_directive(self, node: Node) -> AgentDirectiveStmt:
        return _directive_conversion._walk_agent_directive(self, node)

    def _walk_agent_directive_parts(
        self,
        node: Node,
    ) -> tuple[str | None, str | None, str | None, str | None, list[Any]]:
        return _directive_conversion._walk_agent_directive_parts(self, node)

    def _walk_adapter_directive(self, node: Node) -> AdapterDirectiveStmt:
        return _directive_conversion._walk_adapter_directive(self, node)

    def _walk_simple_directive_value(self, node: Node) -> tuple[str | None, str | None]:
        return _directive_conversion._walk_simple_directive_value(self, node)

    def _walk_effort_directive(self, node: Node) -> EffortDirectiveStmt:
        return _directive_conversion._walk_effort_directive(self, node)

    def _walk_approval_directive(self, node: Node) -> ApprovalDirectiveStmt:
        return _directive_conversion._walk_approval_directive(self, node)

    def _walk_hooks_directive(self, node: Node) -> HooksDirectiveStmt:
        return _directive_conversion._walk_hooks_directive(self, node)

    def _walk_history_directive(self, node: Node) -> HistoryDirectiveStmt:
        return _directive_conversion._walk_history_directive(self, node)

    def _walk_skills_directive(self, node: Node) -> SkillsDirectiveStmt:
        return _directive_conversion._walk_skills_directive(self, node)

    def _walk_codemode_directive(self, node: Node) -> CodeModeDirectiveStmt:
        return _directive_conversion._walk_codemode_directive(self, node)

    def _walk_system_directive(self, node: Node) -> SystemDirectiveStmt:
        return _directive_conversion._walk_system_directive(self, node)

    def _walk_mcp_directive(self, node: Node) -> McpDirectiveStmt:
        return _directive_conversion._walk_mcp_directive(self, node)

    def _walk_mcp_field(self, node: Node) -> McpField:
        return _directive_conversion._walk_mcp_field(self, node)

    def _walk_settings_directive(self, node: Node) -> SettingsDirectiveStmt:
        return _directive_conversion._walk_settings_directive(self, node)

    def _walk_settings_field(self, node: Node) -> SettingsField:
        return _directive_conversion._walk_settings_field(self, node)

    def _walk_settings_value_field(self, node: Node) -> SettingsField:
        return _directive_conversion._walk_settings_value_field(self, node)

    def _walk_artifacts_directive(self, node: Node) -> ArtifactsDirectiveStmt:
        return _directive_conversion._walk_artifacts_directive(self, node)

    def _walk_artifacts_field(self, node: Node) -> ArtifactsField:
        return _directive_conversion._walk_artifacts_field(self, node)

    def _walk_use_directive(self, node: Node) -> UseDirectiveStmt:
        return _directive_conversion._walk_use_directive(self, node)

    def _walk_use_tool_names(self, body_node: Node) -> list[str]:
        return _directive_conversion._walk_use_tool_names(self, body_node)

    def _walk_profile_directive(self, node: Node) -> ProfileDefStmt:
        return _profile_conversion._walk_profile_directive(self, node)

    def _walk_optimize_directive(self, node: Node) -> tuple[OptimizeSpan, TemplateBlock]:
        return _validation_conversion._walk_optimize_directive(self, node)

    # =================================================================
    # Validation blocks
    # =================================================================

    def _walk_validation_block(self, node: Node) -> Union[TestSuite, EvalSuite, None]:
        """Walk a `@test:` or `@eval:` block.

        The grammar accepts ``@<kw>:<procedure>:`` with a uniform body
        rule that allows ``> case:`` (test) or ``> data:`` / ``>
        test_data:`` / ``> metric:`` (eval) directives. We dispatch
        here based on the kw text, and emit a structured "Unknown
        directive" error for any body entry that does not match the
        chosen suite type.
        """
        return _validation_conversion._walk_validation_block(self, node)

    def _walk_test_case(self, node: Node) -> TestCase:
        return _validation_conversion._walk_test_case(self, node)

    def _walk_eval_dataset(self, node: Node, *, is_test: bool) -> EvalDataset:
        return _validation_conversion._walk_eval_dataset(self, node, is_test=is_test)

    def _walk_eval_metric(self, node: Node) -> EvalMetric:
        return _validation_conversion._walk_eval_metric(self, node)

    def _walk_eval_entry_body(self, body: Optional[Node]) -> tuple[str, bool]:
        return _validation_conversion._walk_eval_entry_body(self, body)


def cst_to_ast(
    source: bytes,
    tree: Tree,
    source_path: Optional[str] = None,
    source_map: CompositeSourceMap | None = None,
) -> Union[Program, ProgramWithValidation]:
    """Walk ``tree`` and produce a Kedi AST.

    Raises :class:`KediParseError` if any error-severity diagnostic was
    collected (parse errors from tree-sitter or structural-validation
    issues).
    """
    return _Walker(source, tree, source_path, source_map).walk_program()
