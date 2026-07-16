"""Walk a tree-sitter Kedi CST and produce typed Python AST dataclasses.

This module is the bridge between the tree-sitter parser (which produces
a concrete syntax tree of opaque ``tree_sitter.Node`` objects) and the
Kedi AST defined in :mod:`kedi.lang.ast`.

Design notes:

  - Every produced AST node carries a ``Span`` populated from its CST
    counterpart's ``start_byte`` / ``end_byte`` / ``start_point`` /
    ``end_point``.
  - Type annotations are produced in two forms: a structured
    ``TypeExpr`` (the new representation; populated on the ``type_expr``
    field) and the user's raw source slice as a string (kept on the
    legacy ``type_annotation`` field for compatibility with the compiler
    / codegen / optimizer modules that have not yet migrated). The two
    forms agree by construction.
  - ``parse_program`` returns ``ProgramWithValidation`` whenever any
    ``@test:`` or ``@eval:`` block is present in the source. For
    backward compatibility with read sites that still use the legacy
    ``setattr``-based access pattern (``getattr(prog,
    "_validation_evals", [])``), the same lists are also exposed as
    ``_validation_tests`` and ``_validation_evals`` attributes.

The walker is implemented as a single class with one ``_walk_<rule>``
method per relevant CST node type, plus a small dispatcher
``_walk_node``.
"""

from __future__ import annotations

import ast as py_ast
import re
from typing import List, Optional, Union

from tree_sitter import Node, Tree

from ...agent_profile.settings import MODEL_SETTING_KEYS, is_valid_model_setting_key
from ...effort import EFFORT_VALUES, is_valid_effort
from ...mcp import MCP_TRANSPORT_VALUES, is_valid_mcp_transport
from ...source_loading import CompositeSourceMap
from ..ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    AssignBlockStmt,
    AssignStmt,
    AutoSpec,
    BacktickLineStmt,
    CallArg,
    CallSegment,
    EffortDirectiveStmt,
    EvalDataset,
    EvalMetric,
    EvalSuite,
    InputSegment,
    LiteralArg,
    McpDirectiveStmt,
    McpField,
    ModelDirectiveStmt,
    ModuleExport,
    ModuleImport,
    OptimizeSpan,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    ProfileAdapterEntry,
    ProfileAgentEntry,
    ProfileDefStmt,
    ProfileEffortEntry,
    ProfileMcpEntry,
    ProfileModelEntry,
    ProfileSettingsEntry,
    ProfileSystemEntry,
    Program,
    ProgramWithValidation,
    PythonBlock,
    PythonExprNativeSegment,
    PythonExprSegment,
    RawInvokeStmt,
    ReturnBlockStmt,
    ReturnStmt,
    Segment,
    SettingsDirectiveStmt,
    SettingsField,
    Span,
    Stmt,
    SystemDirectiveStmt,
    TemplateBlock,
    TemplateLine,
    TestCase,
    TestSuite,
    TextSegment,
    TypeApply,
    TypeDef,
    TypeDefStmt,
    TypeExpr,
    TypeField,
    TypeLiteral,
    TypePython,
    TypeRef,
    TypeUnion,
    UseDirectiveStmt,
)
from .diagnostics import KediDiagnostic, KediParseError
from .escapes import dedent_block, unescape_text_with_tracking

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _span(
    node: Node,
    source_path: Optional[str] = None,
    source_map: CompositeSourceMap | None = None,
) -> Span:
    """Build a :class:`Span` from a tree-sitter node."""
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    start_resolved = source_map.resolve_line(start_line) if source_map is not None else None
    end_resolved = source_map.resolve_line(end_line) if source_map is not None else None
    start_byte = node.start_byte
    end_byte = node.end_byte
    mapped_end_line = end_resolved.source_line if end_resolved else end_line
    if source_map is not None and start_resolved is not None:
        local_end_line = start_resolved.source_line + end_line - start_line
        local_start_byte = source_map.get_byte_offset(
            start_resolved.source_path,
            start_resolved.source_line,
            node.start_point[1],
        )
        local_end_byte = source_map.get_byte_offset(
            start_resolved.source_path,
            local_end_line,
            node.end_point[1],
        )
        if local_start_byte is not None and local_end_byte is not None:
            start_byte = local_start_byte
            end_byte = local_end_byte
            mapped_end_line = local_end_line
    return Span(
        start_byte=start_byte,
        end_byte=end_byte,
        # tree-sitter Point.row/column are 0-based; convert to 1-based.
        start_line=start_resolved.source_line if start_resolved else start_line,
        start_col=node.start_point[1] + 1,
        end_line=mapped_end_line,
        end_col=node.end_point[1] + 1,
        source_path=start_resolved.source_path if start_resolved else source_path,
        source_id=start_resolved.source_id if start_resolved else source_path,
    )


def _src_text(source: bytes, node: Node) -> str:
    """Extract the raw source slice corresponding to ``node`` as UTF-8."""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _identifier_name(source: bytes, node: Node) -> str:
    """Return the text of an identifier node."""
    return _src_text(source, node)


def _hyphenated_identifier_from_error(source: bytes, node: Node) -> str | None:
    """Recover an invalid identifier only from tokens adjacent to the error node."""
    line_end = source.find(b"\n", node.start_byte)
    if line_end < 0:
        line_end = len(source)

    for child in node.children:
        if child.type != "identifier" or child.end_point[0] != node.start_point[0]:
            continue
        suffix = source[child.end_byte : line_end].decode("utf-8")
        match = re.match(r"(?:-[A-Za-z0-9_]+)+", suffix)
        if match is not None:
            return _identifier_name(source, child) + match.group(0)

    fragment = source[node.start_byte : line_end].decode("utf-8")
    match = re.match(r"-[A-Za-z_][A-Za-z0-9_]*", fragment)
    if match is None:
        return None
    previous = node.prev_named_sibling
    if (
        previous is not None
        and previous.type == "identifier"
        and previous.end_byte == node.start_byte
        and previous.end_point[0] == node.start_point[0]
    ):
        return _identifier_name(source, previous) + match.group(0)
    return match.group(0)


def _named_children(node: Node) -> List[Node]:
    """All named children of ``node`` in source order."""
    return [c for c in node.children if c.is_named]


def _field(node: Node, name: str) -> Optional[Node]:
    """Get the (single) named child stored under field ``name``, or None."""
    return node.child_by_field_name(name)


def _decode_block_comment_body(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return ""
    inner = lines[1:-1]
    dedented = dedent_block(inner)
    return "\n".join(dedented).strip("\n")


def _fields(node: Node, name: str) -> List[Node]:
    """Get all named children stored under field ``name``."""
    return [c for c in node.children if c.is_named and node.field_name_for_child(c.id) == name]


# ---------------------------------------------------------------------
# Inline-Python body decoding
# ---------------------------------------------------------------------


def _decode_inline_python_body(source: bytes, body_node: Node) -> str:
    r"""Decode the body of an ``inline_python_expr`` (between backticks).

    Backslash-backtick (``\``) is the only escape recognised inside
    inline Python; legacy ``segment_parser`` does the same with
    ``raw.replace("\\`", "`")``.
    """
    raw = _src_text(source, body_node)
    return raw.replace("\\`", "`")


def _decode_type_python_body(source: bytes, body_node: Node) -> str:
    """Decode the body of a ``type_python`` (backtick-wrapped type expr).

    Same escape rule as :func:`_decode_inline_python_body`.
    """
    raw = _src_text(source, body_node)
    return raw.replace("\\`", "`")


# ---------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------


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

    # ----- entry point -----
    def walk_program(self) -> Union[Program, ProgramWithValidation]:
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
        for child in children:
            t = child.type
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
                prog.type_defs.append(self._walk_type_def(child))
            elif t == "module_import":
                prog.imports.append(self._walk_module_import(child))
            elif t == "module_export":
                prog.exports.append(self._walk_module_export(child))
            elif t == "validation_block":
                assert isinstance(prog, ProgramWithValidation)
                suite = self._walk_validation_block(child)
                if isinstance(suite, TestSuite):
                    prog.tests.append(suite)
                elif isinstance(suite, EvalSuite):
                    prog.evals.append(suite)
            elif t == "assign_stmt":
                stmt = self._walk_assign_stmt(child)
                prog.toplevel_body.append(stmt)
                prog.ordered_toplevel.append(stmt)
            elif t == "assign_block_stmt":
                stmt = self._walk_assign_block_stmt(child)
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
                stmt = self._walk_template_block_stmt(child)
                prog.toplevel_body.append(stmt)
                prog.ordered_toplevel.append(stmt)
            elif t == "template_line":
                stmt = self._walk_template_line(child, deprecated=True)
                prog.toplevel_body.append(stmt)
                prog.ordered_toplevel.append(stmt)
            elif t in ("block_comment", "line_comment"):
                # Extras — not part of the AST. Already handled by
                # tree-sitter; we skip them here.
                continue
            elif t == "ERROR" or child.is_error:
                self._error(
                    child, self._parse_error_message(child, "unexpected source at top level")
                )
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
        try:
            return unescape_text_with_tracking(raw)
        except ValueError as exc:
            self._error(node, str(exc), code="K-TEXT-ESCAPE")
            return raw, set()

    def _validate_effort_plain_value(self, node: Node, value: str) -> None:
        if is_valid_effort(value):
            return
        expected = ", ".join(sorted(EFFORT_VALUES))
        self._error(
            node,
            f"effort directive must be one of: {expected}",
            code="K-EFFORT-VALUE",
        )

    def _validate_mcp_transport_plain_value(self, node: Node, value: str) -> None:
        if is_valid_mcp_transport(value):
            return
        expected = ", ".join(sorted(MCP_TRANSPORT_VALUES))
        self._error(
            node,
            f"MCP transport must be one of: {expected}",
            code="K-MCP-TRANSPORT",
        )

    def _validate_setting_key(self, node: Node, name: str) -> None:
        if is_valid_model_setting_key(name):
            return
        expected = ", ".join(sorted(MODEL_SETTING_KEYS))
        self._error(
            node,
            f"model setting must be one of: {expected}",
            code="K-SETTING-KEY",
        )

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
        try:
            text = self.source.decode("utf-8")
        except UnicodeDecodeError:
            return
        lines = text.split("\n")
        # Drop the trailing empty element from a final '\n'.
        if lines and lines[-1] == "":
            lines = lines[:-1]
        in_python = False
        in_block = False
        block_open_line: Optional[int] = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Toggle python-block mode on any line that opens or
            # closes a triple-backtick fence. Legacy `strip_comments`
            # uses the loose `endswith("```")` heuristic to handle
            # cases like `= ``` ` (an opening fence with leading `=`
            # on the same line) and `[x] = ``` `.
            if stripped == "```" or ("```" in stripped and stripped.endswith("```")):
                in_python = not in_python
                continue
            if in_python:
                continue
            if stripped == "###":
                if not in_block:
                    in_block = True
                    block_open_line = i
                else:
                    in_block = False
                    block_open_line = None
                continue
            if in_block:
                if "###" in stripped:
                    self._line_error(
                        i,
                        "Invalid use of '###' inside block comment",
                        code="K-BLOCK-COMMENT",
                    )
                continue
            if "###" in stripped:
                self._line_error(
                    i,
                    "'###' must occupy an entire line for block comments",
                    code="K-BLOCK-COMMENT",
                )
        if in_block:
            self._line_error(
                block_open_line if block_open_line is not None else 0,
                "Unterminated block comment",
                code="K-BLOCK-COMMENT",
            )

    def _line_error(self, line_index: int, message: str, *, code: Optional[str] = None) -> None:
        """Record a diagnostic at a specific (0-based) line, full-line span."""
        try:
            text = self.source.decode("utf-8")
        except UnicodeDecodeError:
            return
        lines = text.split("\n")
        line = lines[line_index] if 0 <= line_index < len(lines) else ""
        # Compute byte range for this line.
        start = 0
        for j in range(line_index):
            start += len(lines[j].encode("utf-8")) + 1  # +1 for '\n'
        end_byte = start + len(line.encode("utf-8"))
        resolved = self.source_map.resolve_line(line_index + 1) if self.source_map else None
        span = Span(
            start_byte=start,
            end_byte=end_byte,
            start_line=resolved.source_line if resolved else line_index + 1,
            start_col=1,
            end_line=resolved.source_line if resolved else line_index + 1,
            end_col=max(1, len(line) + 1),
            source_path=resolved.source_path if resolved else self.source_path,
            source_id=resolved.source_id if resolved else self.source_path,
        )
        self.diagnostics.append(
            KediDiagnostic(severity="error", message=message, span=span, code=code, snippet=line)
        )

    def _record_tree_error(self, node: Node) -> None:
        """Convert one ERROR / MISSING node into a parser diagnostic."""
        raw = _src_text(self.source, node).strip()
        if raw.startswith("> optimize"):
            parts = raw.split(":", 2)
            name = parts[1].strip() if len(parts) > 1 else ""
            self._error(
                node,
                f"Optimize span {name!r} must have an indented block after it",
                code="K-OPT-EMPTY",
            )
            return
        if node.is_missing:
            msg = f"missing {node.type!r}"
        else:
            msg = self._parse_error_message(node, "unexpected syntax")
        span = self._span(node)
        if node.start_point[0] != node.end_point[0]:
            line_end = self.source.index(b"\n", node.start_byte)
            span.end_byte = line_end
            span.end_line = span.start_line
            span.end_col = node.start_point[1] + (line_end - node.start_byte) + 1
        self.diagnostics.append(
            KediDiagnostic(severity="error", message=msg, span=span, code="K-PARSE")
        )

    def _collect_pre_walk_tree_errors(
        self,
        node: Node,
        parent_type: str | None = None,
    ) -> None:
        """Catch structurally unsafe CST errors before AST field access."""
        if node.has_error and (node.type == "ERROR" or node.is_missing):
            if node.is_missing and node.type == "_newline":
                return
            contextual_parents = {"source_file", "validation_body", "block"}
            if node.is_missing or parent_type not in contextual_parents:
                self._record_tree_error(node)
            return  # don't recurse into ERROR subtrees
        for child in node.children:
            if child.has_error:
                self._collect_pre_walk_tree_errors(child, node.type)

    def _parse_error_message(self, node: Node, fallback: str) -> str:
        raw = _src_text(self.source, node)
        if "``" in raw:
            return "inline Python expression cannot be empty"
        invalid_identifier = _hyphenated_identifier_from_error(self.source, node)
        if invalid_identifier is not None:
            return (
                f"invalid identifier {invalid_identifier!r}: '-' cannot be used in "
                "Kedi identifiers; use '_' instead"
            )
        return fallback

    # =================================================================
    # Module directives
    # =================================================================

    def _walk_module_import(self, node: Node) -> ModuleImport:
        module_node = _field(node, "module")
        assert module_node is not None
        return ModuleImport(
            module=_identifier_name(self.source, module_node),
            loc=self._span(node),
        )

    def _walk_module_export(self, node: Node) -> ModuleExport:
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

    # =================================================================
    # Procedures
    # =================================================================

    def _walk_procedure_def(self, node: Node) -> ProcedureDef:
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

    def _walk_block_into(self, block_node: Node, proc: ProcedureDef) -> None:
        for child in _named_children(block_node):
            stmt = self._walk_stmt(child, proc)
            if stmt is None:
                continue
            proc.body.append(stmt)

    # =================================================================
    # Statements
    # =================================================================

    def _walk_stmt(self, node: Node, proc: Optional[ProcedureDef]) -> Optional[Stmt]:
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
            span = self._walk_optimize_directive(node, proc)
            proc.optimize_spans.append(span)
            return None
        if t == "model_directive":
            return self._walk_model_directive(node)
        if t == "agent_directive":
            return self._walk_agent_directive(node)
        if t == "adapter_directive":
            return self._walk_adapter_directive(node)
        if t == "effort_directive":
            return self._walk_effort_directive(node)
        if t == "system_directive":
            return self._walk_system_directive(node)
        if t == "mcp_directive":
            return self._walk_mcp_directive(node)
        if t == "settings_directive":
            return self._walk_settings_directive(node)
        if t == "use_directive":
            return self._walk_use_directive(node)
        if t == "profile_directive":
            self._error(node, "'> profile:' is only valid at top level", code="K-PROFILE-SCOPE")
            return None
        if t == "assign_stmt":
            return self._walk_assign_stmt(node)
        if t == "assign_block_stmt":
            return self._walk_assign_block_stmt(node)
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
            return self._walk_template_block_stmt(node)
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
                self._error(
                    node, self._parse_error_message(node, f"unexpected statement node {t!r}")
                )
            return None
        # Anything else is an unexpected node — surface a diagnostic
        # but keep parsing the rest of the body.
        self._error(node, f"unexpected statement node {t!r}")
        return None

    def _walk_assign_stmt(self, node: Node) -> AssignStmt:
        target = _named_children(node)[0]
        assert target.type == "assign_target"
        name_node = _field(target, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        type_node = _field(target, "type")
        type_anno = self._type_expr_to_str(type_node) if type_node is not None else None
        type_expr = self._walk_type_expr(type_node) if type_node is not None else None

        rhs_node = _field(node, "rhs")
        # RHS is either an inline_python_expr or a template_expr.
        if rhs_node is not None and rhs_node.type == "inline_python_expr":
            code = self._walk_inline_python_expr(rhs_node)
            seg = PythonExprNativeSegment(code=code, loc=self._span(rhs_node))
            return AssignStmt(
                name=name,
                expr=[seg],
                type_annotation=type_anno,
                type_expr=type_expr,
                loc=self._span(node),
            )
        # template_expr
        segs = self._walk_template_expr(rhs_node) if rhs_node is not None else []
        return AssignStmt(
            name=name,
            expr=segs,
            type_annotation=type_anno,
            type_expr=type_expr,
            loc=self._span(node),
        )

    def _walk_assign_block_stmt(self, node: Node) -> AssignBlockStmt:
        target = _named_children(node)[0]
        name_node = _field(target, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        type_node = _field(target, "type")
        type_anno = self._type_expr_to_str(type_node) if type_node is not None else None
        type_expr = self._walk_type_expr(type_node) if type_node is not None else None
        code_node = _field(node, "code")
        code = self._dedent_fenced(code_node) if code_node is not None else ""
        return AssignBlockStmt(
            name=name,
            code=code,
            type_annotation=type_anno,
            type_expr=type_expr,
            loc=self._span(node),
        )

    def _walk_raw_invoke_stmt(self, node: Node) -> RawInvokeStmt:
        target = _field(node, "target")
        assert target is not None and target.type == "assign_target"
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

    def _walk_return_stmt(self, node: Node) -> Union[ReturnStmt, ReturnBlockStmt]:
        value_node = _field(node, "value")
        # Legacy invariant: `= \`code\`` (sole single-backtick) is a
        # ``ReturnBlockStmt`` so the runtime preserves the native
        # Python value (e.g. ``int``) rather than rendering it to a
        # string. The branch must distinguish ``= \`code\``` from a
        # ``ReturnStmt`` whose value happens to include a backtick
        # segment alongside other content (which IS a ReturnStmt with
        # a single PythonExprNativeSegment, rendered to string).
        if value_node is not None and value_node.type == "inline_python_expr":
            code = self._walk_inline_python_expr(value_node)
            return ReturnBlockStmt(code=code, loc=self._span(node))
        segs = self._walk_template_expr(value_node) if value_node is not None else []
        return ReturnStmt(segments=segs, loc=self._span(node))

    def _walk_return_block_stmt(self, node: Node) -> ReturnBlockStmt:
        code_node = _field(node, "code")
        code = self._dedent_fenced(code_node) if code_node is not None else ""
        return ReturnBlockStmt(code=code, loc=self._span(node))

    def _walk_python_block(self, node: Node) -> PythonBlock:
        return PythonBlock(code=self._walk_python_block_code(node), loc=self._span(node))

    def _walk_python_block_code(self, node: Node) -> str:
        code_node = _field(node, "code")
        return self._dedent_fenced(code_node) if code_node is not None else ""

    def _walk_backtick_line_stmt(self, node: Node) -> BacktickLineStmt:
        # The single child is an inline_python_expr.
        for child in _named_children(node):
            if child.type == "inline_python_expr":
                code = self._walk_inline_python_expr(child)
                return BacktickLineStmt(code=code, loc=self._span(node))
        self._error(node, "backtick_line_stmt missing inline_python_expr child")
        return BacktickLineStmt(code="", loc=self._span(node))

    def _template_line_from_expr(self, expr_node: Node) -> TemplateLine:
        return TemplateLine(
            segments=self._walk_template_expr(expr_node),
            loc=self._span(expr_node),
        )

    def _system_line_from_expr(self, expr_node: Node) -> TemplateLine:
        segments: List[Segment] = []
        for child in _named_children(expr_node):
            if child.type == "system_angle_segment":
                raw = _src_text(self.source, child)
                inner = raw[1:-1] if raw.endswith(">") else raw[1:]
                if raw == "<``>":
                    segments.append(TextSegment(text="<``>", loc=self._span(child)))
                elif inner.isidentifier():
                    segments.append(InputSegment(name=inner, loc=self._span(child)))
                elif len(inner) >= 2 and inner[0] == "`" and inner[-1] == "`":
                    segments.append(
                        PythonExprSegment(
                            code=inner[1:-1].replace("\\`", "`"),
                            loc=self._span(child),
                        )
                    )
                else:
                    self._error(
                        child,
                        "System instructions only support literal text, <name>, and <`python`> substitutions.",
                        code="K-SYSTEM-SEGMENT",
                    )
                continue
            if child.type == "text_segment":
                segments.append(self._walk_segment(child))
                continue
            self._error(
                child,
                "System instructions only support literal text, <name>, and <`python`> substitutions.",
                code="K-SYSTEM-SEGMENT",
            )
        self._strip_leading_template_indent(segments)
        return TemplateLine(segments=segments, loc=self._span(expr_node))

    def _walk_template_block_stmt(self, node: Node) -> TemplateBlock:
        lines: List[TemplateLine] = []
        head = _field(node, "head")
        if head is not None:
            lines.append(self._template_line_from_expr(head))
        for cont in node.children_by_field_name("continuation"):
            lines.append(self._template_line_from_expr(cont))
        self._validate_template_block_scope(lines)
        return TemplateBlock(lines=lines, loc=self._span(node))

    def _validate_template_block_scope(self, lines: List[TemplateLine]) -> None:
        produced_names: set[str] = set()
        for line in lines:
            for seg in line.segments:
                if isinstance(seg, InputSegment) and seg.name in produced_names:
                    self.diagnostics.append(
                        KediDiagnostic(
                            severity="error",
                            message=(
                                f"Template block cannot read LLM output '{seg.name}' "
                                "before the block has finished. Start a new `>>` block "
                                "before using it as a substitution."
                            ),
                            span=seg.loc or line.loc or Span(0, 0, 1, 1, 1, 1),
                            code="K-TEMPLATE-BLOCK-FORWARD-OUTPUT",
                        )
                    )
                elif isinstance(seg, OutputSegment):
                    produced_names.add(seg.name)

    def _walk_template_line(self, node: Node, *, deprecated: bool = False) -> TemplateLine:
        if deprecated:
            self._error(
                node,
                "Template prompts at procedure or top level must start with `>>`.",
                code="K-BARE-TEMPLATE",
            )
        for child in _named_children(node):
            if child.type == "template_expr":
                return self._template_line_from_expr(child)
        return TemplateLine(segments=[], loc=self._span(node))

    # =================================================================
    # Type expressions
    # =================================================================

    def _walk_type_expr(self, node: Node) -> Optional[TypeExpr]:
        # `type_expr` is a thin wrapper; its sole named child is the
        # actual term (type_ref / type_apply / type_union / type_python).
        inner = _named_children(node)
        if not inner:
            return None
        target = inner[0]
        if target.type == "type_ref":
            name = _identifier_name(
                self.source, _field(target, "name") or _named_children(target)[0]
            )
            return TypeRef(name=name, loc=self._span(target))
        if target.type == "type_apply":
            name_node = _field(target, "name")
            assert name_node is not None
            name = _identifier_name(self.source, name_node)
            args = [
                self._walk_type_arg(arg)
                for arg in _named_children(target)
                if arg.type == "type_arg"
            ]
            args = [a for a in args if a is not None]
            return TypeApply(name=name, args=args, loc=self._span(target))
        if target.type == "type_union":
            left_node = _field(target, "left")
            right_node = _field(target, "right")
            left = self._walk_type_term(left_node) if left_node is not None else None
            right = self._walk_type_term(right_node) if right_node is not None else None
            return TypeUnion(left=left, right=right, loc=self._span(target))
        if target.type == "type_python":
            body_node = _field(target, "code")
            code = _decode_type_python_body(self.source, body_node) if body_node is not None else ""
            return TypePython(code=code, loc=self._span(target))
        # Unknown type_expr shape
        self._error(target, f"unexpected type_expr inner {target.type!r}")
        return None

    def _walk_type_arg(self, node: Node) -> Optional[TypeExpr]:
        """Walk a ``type_arg`` wrapper inside generic type applications."""
        target = _named_children(node)[0]
        if target.type == "type_python":
            body_node = _field(target, "code")
            code = _decode_type_python_body(self.source, body_node) if body_node is not None else ""
            return TypePython(code=code, loc=self._span(target))
        if target.type == "type_string":
            source = _src_text(self.source, target).strip()
            value = py_ast.literal_eval(source)
            return TypeLiteral(value=value, source=source, loc=self._span(target))
        return self._walk_type_term(target)

    def _walk_type_term(self, node: Node) -> Optional[TypeExpr]:
        """Walk a node that is a `_type_expr_term` alternative (no `type_expr` wrapper)."""
        t = node.type
        if t == "type_ref":
            name = _identifier_name(self.source, _field(node, "name") or _named_children(node)[0])
            return TypeRef(name=name, loc=self._span(node))
        if t == "type_apply":
            name_node = _field(node, "name")
            assert name_node is not None
            name = _identifier_name(self.source, name_node)
            args = [
                self._walk_type_arg(arg) for arg in _named_children(node) if arg.type == "type_arg"
            ]
            args = [a for a in args if a is not None]
            return TypeApply(name=name, args=args, loc=self._span(node))
        if t == "type_union":
            left_node = _field(node, "left")
            right_node = _field(node, "right")
            left = self._walk_type_term(left_node) if left_node is not None else None
            right = self._walk_type_term(right_node) if right_node is not None else None
            return TypeUnion(left=left, right=right, loc=self._span(node))
        self._error(node, f"unexpected type term {t!r}")
        return None

    def _type_expr_to_str(self, node: Node) -> str:
        """Render a type_expr / inner term back to its source slice.

        Used to populate the legacy ``type_annotation`` string field for
        consumers (compiler / codegen / optimizer) that have not yet
        migrated to the structured ``type_expr`` form. Returning the
        raw source slice means the rendering is byte-perfect — including
        the backticks for ``type_python`` cases, matching the legacy
        ``KediType.resolve`` expectation.
        """
        return _src_text(self.source, node).strip()

    # =================================================================
    # Type definitions
    # =================================================================

    def _walk_type_def(self, node: Node) -> TypeDef:
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        fields: List[TypeField] = []
        seen_names: set[str] = set()
        seen_default = False
        field_list = _field(node, "fields")
        if field_list is not None:
            for f in _named_children(field_list):
                if f.type != "type_field":
                    continue
                fname_node = _field(f, "name")
                assert fname_node is not None
                fname = _identifier_name(self.source, fname_node)
                if fname in seen_names:
                    self._error(
                        f,
                        f"duplicate field {fname!r} in type {name!r}",
                        code="K-TYPE-DUP",
                    )
                seen_names.add(fname)
                ftype_node = _field(f, "type")
                ftype_anno = self._type_expr_to_str(ftype_node) if ftype_node is not None else None
                ftype_expr = self._walk_type_expr(ftype_node) if ftype_node is not None else None
                default_node = _field(f, "default")
                default_expr = None
                if default_node is not None:
                    seen_default = True
                    if ftype_node is None:
                        self._error(
                            f,
                            "defaulted type field requires a type annotation",
                            code="K-TYPE-DEFAULT-ANNOTATION",
                        )
                    default_expr = self._walk_inline_python_expr(default_node)
                elif seen_default:
                    self._error(
                        f,
                        "required field cannot follow a defaulted field",
                        code="K-TYPE-FIELD-ORDER",
                    )
                fields.append(
                    TypeField(
                        name=fname,
                        type_annotation=ftype_anno,
                        type_expr=ftype_expr,
                        default_expr=default_expr,
                        loc=self._span(f),
                    )
                )
        return TypeDef(name=name, fields=fields, loc=self._span(node))

    # =================================================================
    # Template segments
    # =================================================================

    def _walk_template_expr(self, node: Node) -> List[Segment]:
        segs = [self._walk_segment(c) for c in _named_children(node) if c.is_named]
        self._strip_leading_template_indent(segs)
        return segs

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
        if segs and isinstance(segs[0], TextSegment):
            first = segs[0]
            # Find how many leading non-escaped whitespace chars to
            # strip. An escaped-WS index (from `\s`, `\t`, `\n`) must
            # be preserved, so we stop stripping at the first such
            # position.
            shift = 0
            for idx, ch in enumerate(first.text):
                if ch not in (" ", "\t"):
                    break
                if idx in first.escaped_whitespace:
                    break
                shift += 1
            if shift:
                first.text = first.text[shift:]
                if first.escaped_whitespace:
                    first.escaped_whitespace = {
                        i - shift for i in first.escaped_whitespace if i >= shift
                    }

    def _walk_segment(self, node: Node) -> Segment:
        t = node.type
        if t == "text_segment":
            raw = _src_text(self.source, node)
            # NOTE: do NOT strip leading/trailing whitespace here. The
            # runtime in `compiler/evaluation.py:prepare_segments`
            # already trims the first/last text-segment of an
            # expression — including the body-indent absorbed by the
            # scanner's leading-WS rule. Stripping in this walker would
            # double up: it would also remove meaningful whitespace
            # that sits BETWEEN segments (e.g. the gap between `<x>`
            # and `mid` in ``  left   <x>   mid``).
            text, escaped_positions = self._unescape_text_segment(node, raw)
            return TextSegment(
                text=text,
                escaped_whitespace=escaped_positions,
                loc=self._span(node),
            )
        if t == "input_segment":
            name = _identifier_name(self.source, _field(node, "name") or _named_children(node)[0])
            return InputSegment(name=name, loc=self._span(node))
        if t == "call_segment":
            return self._walk_call_segment(node)
        if t == "output_segment":
            name = _identifier_name(self.source, _field(node, "name") or _named_children(node)[0])
            type_node = _field(node, "type")
            type_anno = self._type_expr_to_str(type_node) if type_node is not None else None
            type_expr = self._walk_type_expr(type_node) if type_node is not None else None
            return OutputSegment(
                name=name,
                type_annotation=type_anno,
                type_expr=type_expr,
                loc=self._span(node),
            )
        if t == "python_expr_segment":
            # Child is an inline_python_expr.
            for child in _named_children(node):
                if child.type == "inline_python_expr":
                    code = self._walk_inline_python_expr(child)
                    return PythonExprSegment(code=code, loc=self._span(node))
            self._error(node, "python_expr_segment missing inline_python_expr child")
            return PythonExprSegment(code="", loc=self._span(node))
        self._error(node, f"unexpected segment {t!r}")
        # Fallback so callers don't need None checks.
        return TextSegment(text="", loc=self._span(node))

    def _walk_call_segment(self, node: Node) -> CallSegment:
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        args: List[CallArg] = []
        arg_list = _field(node, "args")
        if arg_list is not None:
            for arg_node in _named_children(arg_list):
                if arg_node.type == "call_arg":
                    args.append(self._walk_call_arg(arg_node))
        return CallSegment(name=name, args=args, loc=self._span(node))

    def _walk_call_arg(self, node: Node) -> CallArg:
        r"""Build a call argument with explicit payload fields.

        A literal argument can carry one of these payloads:

          - ``native_backtick``: arg passes a native Python value
            (from a bare backtick `` `expr` ``).
          - ``python_segment``: arg is a single ``<\`...\`>`` segment.
          - ``call_segment``: arg is a single ``<func(...)>`` segment.
          - ``render_segments``: arg is a mini-template rendered to a string.
        """
        children = _named_children(node)
        if not children:
            return LiteralArg(value="", loc=self._span(node))

        first = children[0]
        if first.type == "inline_python_expr":
            code = self._walk_inline_python_expr(first)
            lit = LiteralArg(
                value="__NATIVE_BACKTICK__", loc=self._span(node), native_backtick=code
            )
            return lit

        if first.type == "template_arg":
            segs = [self._walk_segment_in_call(c) for c in _named_children(first)]
            if len(segs) == 1:
                only = segs[0]
                if isinstance(only, PythonExprSegment):
                    return LiteralArg(
                        value="__RENDER__",
                        loc=self._span(node),
                        python_segment=only,
                    )
                if isinstance(only, CallSegment):
                    return LiteralArg(
                        value="__RENDER__",
                        loc=self._span(node),
                        call_segment=only,
                    )
            return LiteralArg(
                value="__RENDER__",
                loc=self._span(node),
                render_segments=segs,
            )

        # Unknown shape — make a best-effort literal.
        lit = LiteralArg(value=_src_text(self.source, node), loc=self._span(node))
        return lit

    def _walk_segment_in_call(self, node: Node) -> Segment:
        """A segment appearing inside a ``template_arg`` (call argument).

        Each call-argument text segment is BOTH the first AND the last
        text of its enclosing arg expression in the legacy semantics —
        so we trim both ends here (matching ``parse_call_arguments``,
        which calls ``raw.strip()`` on the arg before segment-parsing
        it). Internal whitespace is preserved.
        """
        if node.type == "text_in_call_segment":
            raw = _src_text(self.source, node).strip(" \t")
            text, escaped_positions = self._unescape_text_segment(node, raw)
            return TextSegment(
                text=text,
                escaped_whitespace=escaped_positions,
                loc=self._span(node),
            )
        return self._walk_segment(node)

    # =================================================================
    # Inline python expression body
    # =================================================================

    def _walk_inline_python_expr(self, node: Node) -> str:
        body = _field(node, "code")
        if body is None:
            return ""
        return _decode_inline_python_body(self.source, body)

    # =================================================================
    # Fenced code dedenting
    # =================================================================

    def _dedent_fenced(self, code_node: Node) -> str:
        """Extract and dedent a python_code node's body."""
        raw = _src_text(self.source, code_node)
        # Strip the trailing newline (if any) before splitting so we
        # don't get a phantom empty last line, then re-join.
        lines = raw.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        dedented = dedent_block(lines)
        return "\n".join(dedented)

    # =================================================================
    # Directives
    # =================================================================

    def _walk_auto_directive(self, node: Node) -> AutoSpec:
        body_node = _field(node, "body")
        if body_node is None:
            return AutoSpec(prompt="", loc=self._span(node))
        prompt_lines: List[str] = []
        for tline in _named_children(body_node):
            if tline.type != "template_line":
                continue
            # Auto prompts are free text — render the raw source minus
            # trailing newline, then dedent to remove the body's leading
            # indentation.
            expr = next(
                (c for c in _named_children(tline) if c.type == "template_expr"),
                None,
            )
            if expr is None:
                prompt_lines.append("")
            else:
                prompt_lines.append(_src_text(self.source, expr).rstrip())
        # Strip common leading indent (the auto body is indented under
        # the directive header; the user's prompt content typically lives
        # at one consistent indent level).
        dedented = dedent_block(prompt_lines)
        prompt = "\n".join(dedented).strip("\n")
        return AutoSpec(prompt=prompt, loc=self._span(node))

    def _walk_model_directive(self, node: Node) -> ModelDirectiveStmt:
        plain, code = self._walk_simple_directive_value(node)
        return ModelDirectiveStmt(plain_value=plain, backtick_code=code, loc=self._span(node))

    def _walk_agent_directive(self, node: Node) -> AgentDirectiveStmt:
        plain, code, command_plain, command_code = self._walk_agent_directive_parts(node)
        return AgentDirectiveStmt(
            plain_value=plain,
            backtick_code=code,
            command_plain_value=command_plain,
            command_backtick_code=command_code,
            loc=self._span(node),
        )

    def _walk_agent_directive_parts(
        self,
        node: Node,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        body_node = _field(node, "body")
        if body_node is None:
            plain, code = self._walk_simple_directive_value(node)
            return plain, code, None, None

        fields = [child for child in _named_children(body_node) if child.type == "agent_field"]
        if len(fields) != 1:
            self._error(
                body_node,
                "Multiline '> agent:' must contain exactly one agent entry",
                code="K-AGENT-BODY",
            )
            return None, None, None, None
        field = fields[0]
        name_node = _field(field, "name")
        value_node = _field(field, "value")
        assert name_node is not None
        assert value_node is not None
        plain = _identifier_name(self.source, name_node)
        if value_node.type == "inline_python_expr":
            code_node = _field(value_node, "code")
            assert code_node is not None
            return plain, None, None, _src_text(self.source, code_node)
        return plain, None, _src_text(self.source, value_node).strip(), None

    def _walk_adapter_directive(self, node: Node) -> AdapterDirectiveStmt:
        plain, code = self._walk_simple_directive_value(node)
        return AdapterDirectiveStmt(plain_value=plain, backtick_code=code, loc=self._span(node))

    def _walk_simple_directive_value(self, node: Node) -> tuple[str | None, str | None]:
        value_node = _field(node, "value")
        assert value_node is not None
        if value_node.type == "inline_python_expr":
            code_node = _field(value_node, "code")
            assert code_node is not None
            return None, _src_text(self.source, code_node)
        return _src_text(self.source, value_node).strip(), None

    def _walk_effort_directive(self, node: Node) -> EffortDirectiveStmt:
        value_node = _field(node, "value")
        assert value_node is not None
        if value_node.type == "inline_python_expr":
            code_node = _field(value_node, "code")
            assert code_node is not None
            return EffortDirectiveStmt(
                backtick_code=_src_text(self.source, code_node),
                loc=self._span(node),
            )
        plain = _src_text(self.source, value_node).strip()
        self._validate_effort_plain_value(value_node, plain)
        return EffortDirectiveStmt(plain_value=plain, loc=self._span(node))

    def _walk_system_directive(self, node: Node) -> SystemDirectiveStmt:
        head = _field(node, "head")
        body = _field(node, "body")
        lines: List[TemplateLine] = []
        if head is not None:
            lines.append(self._system_line_from_expr(head))
        if body is not None:
            for child in _named_children(body):
                if child.type != "system_line":
                    continue
                expr_node = _field(child, "line")
                if expr_node is not None:
                    lines.append(self._system_line_from_expr(expr_node))
        if lines:
            plain = "\n".join(
                "".join(seg.text for seg in line.segments if isinstance(seg, TextSegment)).strip()
                for line in lines
            ).strip()
            return SystemDirectiveStmt(
                plain_value=plain or None,
                template_lines=lines,
                loc=self._span(node),
            )

        value_node = _field(node, "value")
        if value_node is None:
            return SystemDirectiveStmt(loc=self._span(node))
        if value_node.type == "inline_python_expr":
            code_node = _field(value_node, "code")
            assert code_node is not None
            return SystemDirectiveStmt(
                backtick_code=_src_text(self.source, code_node),
                loc=self._span(node),
            )
        plain = _src_text(self.source, value_node).strip()
        return SystemDirectiveStmt(plain_value=plain, loc=self._span(node))

    def _walk_mcp_directive(self, node: Node) -> McpDirectiveStmt:
        body_node = _field(node, "body")
        assert body_node is not None
        fields: list[McpField] = []
        for child in _named_children(body_node):
            if child.type in ("line_comment", "block_comment"):
                continue
            fields.append(self._walk_mcp_field(child))
        return McpDirectiveStmt(fields=fields, loc=self._span(node))

    def _walk_mcp_field(self, node: Node) -> McpField:
        name_node = _field(node, "name")
        value_node = _field(node, "value")
        assert name_node is not None
        assert value_node is not None
        name = _identifier_name(self.source, name_node)
        if value_node.type == "inline_python_expr":
            code_node = _field(value_node, "code")
            assert code_node is not None
            return McpField(
                name=name,
                backtick_code=_src_text(self.source, code_node),
                loc=self._span(node),
            )
        plain = _src_text(self.source, value_node).strip()
        if name == "transport":
            self._validate_mcp_transport_plain_value(value_node, plain)
        return McpField(
            name=name,
            plain_value=plain,
            loc=self._span(node),
        )

    def _walk_settings_directive(self, node: Node) -> SettingsDirectiveStmt:
        body_node = _field(node, "body")
        assert body_node is not None
        fields: list[SettingsField] = []
        for child in _named_children(body_node):
            if child.type in ("line_comment", "block_comment"):
                continue
            fields.append(self._walk_settings_field(child))
        return SettingsDirectiveStmt(fields=fields, loc=self._span(node))

    def _walk_settings_field(self, node: Node) -> SettingsField:
        name_node = _field(node, "name")
        value_node = _field(node, "value")
        assert name_node is not None
        assert value_node is not None
        name = _identifier_name(self.source, name_node)
        self._validate_setting_key(name_node, name)
        if value_node.type == "inline_python_expr":
            code_node = _field(value_node, "code")
            assert code_node is not None
            return SettingsField(
                name=name,
                backtick_code=_src_text(self.source, code_node),
                loc=self._span(node),
            )
        return SettingsField(
            name=name,
            plain_value=_src_text(self.source, value_node).strip(),
            loc=self._span(node),
        )

    def _walk_use_directive(self, node: Node) -> UseDirectiveStmt:
        body_node = _field(node, "body")
        if body_node is not None:
            tool_names = self._walk_use_tool_names(body_node)
            return UseDirectiveStmt(tool_names=tool_names, loc=self._span(node))
        name_node = _field(node, "name")
        assert name_node is not None
        return UseDirectiveStmt(
            target=_identifier_name(self.source, name_node),
            loc=self._span(node),
        )

    def _walk_use_tool_names(self, body_node: Node) -> list[str]:
        tool_names: list[str] = []
        for child in _named_children(body_node):
            if child.type == "use_tool_name":
                name_node = _field(child, "name")
                assert name_node is not None
                tool_names.append(_identifier_name(self.source, name_node))
            elif child.type == "use_tool_backtick":
                name_node = _field(child, "name")
                assert name_node is not None
                tool_names.append(_identifier_name(self.source, name_node))
        if not tool_names:
            self._error(
                body_node,
                "Multiline '> use:' must list at least one tool",
                code="K-USE-EMPTY",
            )
        return tool_names

    def _walk_profile_directive(self, node: Node) -> ProfileDefStmt:
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        members: list[ProfileModelEntry] = []
        agent_members: list[ProfileAgentEntry] = []
        adapter_members: list[ProfileAdapterEntry] = []
        effort_members: list[ProfileEffortEntry] = []
        system_members: list[ProfileSystemEntry] = []
        mcp_members: list[ProfileMcpEntry] = []
        settings_members: list[ProfileSettingsEntry] = []
        tool_names: list[str] = []
        docstring: str | None = None
        docstring_loc: Span | None = None
        body_node = _field(node, "body")
        if body_node is not None:
            body_children = _named_children(body_node)
            if body_children and body_children[0].type == "block_comment":
                comment = body_children[0]
                docstring = _decode_block_comment_body(_src_text(self.source, comment))
                docstring_loc = self._span(comment)
            for child in body_children:
                if child.type == "agent_directive":
                    plain, code, command_plain, command_code = self._walk_agent_directive_parts(
                        child
                    )
                    agent_members.append(
                        ProfileAgentEntry(
                            plain_value=plain,
                            backtick_code=code,
                            command_plain_value=command_plain,
                            command_backtick_code=command_code,
                            loc=self._span(child),
                        )
                    )
                elif child.type == "adapter_directive":
                    plain, code = self._walk_simple_directive_value(child)
                    adapter_members.append(
                        ProfileAdapterEntry(
                            plain_value=plain,
                            backtick_code=code,
                            loc=self._span(child),
                        )
                    )
                elif child.type == "model_directive":
                    value_node = _field(child, "value")
                    if value_node is None:
                        continue
                    if value_node.type == "inline_python_expr":
                        code_node = _field(value_node, "code")
                        assert code_node is not None
                        members.append(
                            ProfileModelEntry(
                                backtick_code=_src_text(self.source, code_node),
                                loc=self._span(child),
                            )
                        )
                    else:
                        members.append(
                            ProfileModelEntry(
                                plain_value=_src_text(self.source, value_node).strip(),
                                loc=self._span(child),
                            )
                        )
                elif child.type == "effort_directive":
                    value_node = _field(child, "value")
                    assert value_node is not None
                    if value_node.type == "inline_python_expr":
                        code_node = _field(value_node, "code")
                        assert code_node is not None
                        effort_members.append(
                            ProfileEffortEntry(
                                backtick_code=_src_text(self.source, code_node),
                                loc=self._span(child),
                            )
                        )
                    else:
                        plain = _src_text(self.source, value_node).strip()
                        self._validate_effort_plain_value(value_node, plain)
                        effort_members.append(
                            ProfileEffortEntry(
                                plain_value=plain,
                                loc=self._span(child),
                            )
                        )
                elif child.type == "system_directive":
                    system_stmt = self._walk_system_directive(child)
                    if system_stmt.template_lines:
                        system_members.append(
                            ProfileSystemEntry(
                                plain_value=system_stmt.plain_value,
                                template_lines=system_stmt.template_lines,
                                loc=self._span(child),
                            )
                        )
                        continue
                    value_node = _field(child, "value")
                    if value_node is None:
                        continue
                    if value_node.type == "inline_python_expr":
                        code_node = _field(value_node, "code")
                        assert code_node is not None
                        system_members.append(
                            ProfileSystemEntry(
                                backtick_code=_src_text(self.source, code_node),
                                loc=self._span(child),
                            )
                        )
                    else:
                        system_members.append(
                            ProfileSystemEntry(
                                plain_value=_src_text(self.source, value_node).strip(),
                                loc=self._span(child),
                            )
                        )
                elif child.type == "use_directive":
                    use_stmt = self._walk_use_directive(child)
                    if use_stmt.tool_names:
                        tool_names.extend(use_stmt.tool_names)
                    elif use_stmt.name:
                        tool_names.append(use_stmt.name)
                elif child.type == "mcp_directive":
                    mcp_stmt = self._walk_mcp_directive(child)
                    mcp_members.append(
                        ProfileMcpEntry(fields=mcp_stmt.fields, loc=self._span(child))
                    )
                elif child.type == "settings_directive":
                    settings_stmt = self._walk_settings_directive(child)
                    settings_members.append(
                        ProfileSettingsEntry(
                            fields=settings_stmt.fields,
                            loc=self._span(child),
                        )
                    )
        if (
            not members
            and not agent_members
            and not adapter_members
            and not effort_members
            and not system_members
            and not mcp_members
            and not settings_members
            and not tool_names
        ):
            self._error(
                node,
                f"Profile {name!r} must contain at least one '> agent:', '> adapter:', '> model:', '> effort:', '> system:', '> mcp:', '> settings:', or '> use:' member",
                code="K-PROFILE-EMPTY",
            )
        return ProfileDefStmt(
            name=name,
            agent_members=agent_members,
            adapter_members=adapter_members,
            model_members=members,
            effort_members=effort_members,
            system_members=system_members,
            mcp_members=mcp_members,
            settings_members=settings_members,
            tool_names=tool_names,
            docstring=docstring,
            docstring_loc=docstring_loc,
            loc=self._span(node),
        )

    def _walk_optimize_directive(self, node: Node, proc: ProcedureDef) -> OptimizeSpan:
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        body_node = _field(node, "body")
        template_lines: List[TemplateLine] = []
        if body_node is not None:
            for tline in _named_children(body_node):
                if tline.type != "template_line":
                    continue
                tl = self._walk_template_line(tline)
                template_lines.append(tl)
        if not template_lines:
            # Legacy invariant — preserve the exact substring so
            # ``pytest.raises(ValueError, match="must have an indented block")``
            # keeps matching.
            self._error(
                node,
                f"Optimize span {name!r} must have an indented block after it",
                code="K-OPT-EMPTY",
            )
        # The optimize span's template_lines are ALSO appended to the
        # procedure body so they execute normally during a run; the
        # optimizer mutates these line objects in place.
        proc.body.extend(template_lines)
        return OptimizeSpan(
            name=name,
            template_lines=template_lines,
            loc=self._span(node),
        )

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
        kw_node = _field(node, "kw")
        if kw_node is None:
            self._error(node, "validation block missing keyword")
            return None
        kw_text = _src_text(self.source, kw_node)
        proc_node = _field(node, "procedure")
        if proc_node is None:
            self._error(node, "validation block missing procedure name")
            return None
        procedure_name = _identifier_name(self.source, proc_node)
        body = _field(node, "body")
        body_children = _named_children(body) if body is not None else []

        if kw_text == "test":
            suite_t = TestSuite(procedure_name=procedure_name, cases=[], loc=self._span(node))
            for c in body_children:
                if c.type == "test_case":
                    suite_t.cases.append(self._walk_test_case(c))
                elif c.type in ("eval_data", "eval_test_data", "eval_metric"):
                    self._error(
                        c,
                        f"Unknown directive in @test body: {c.type!r}",
                        code="K-VALID-DIR",
                    )
                elif c.type == "ERROR" or c.is_error:
                    self._error(
                        c,
                        "Unknown directive in @test body",
                        code="K-VALID-DIR",
                    )
            return suite_t

        if kw_text == "eval":
            suite_e = EvalSuite(procedure_name=procedure_name, loc=self._span(node))
            metric_count = 0
            for c in body_children:
                if c.type == "test_case":
                    self._error(
                        c,
                        "Unknown directive in @eval body: 'case' is only valid in @test",
                        code="K-VALID-DIR",
                    )
                elif c.type == "eval_data":
                    ds = self._walk_eval_dataset(c, is_test=False)
                    suite_e.datasets[ds.name] = ds
                elif c.type == "eval_test_data":
                    ds = self._walk_eval_dataset(c, is_test=True)
                    suite_e.test_datasets[ds.name] = ds
                elif c.type == "eval_metric":
                    metric_count += 1
                    suite_e.metrics.append(self._walk_eval_metric(c))
                elif c.type == "ERROR" or c.is_error:
                    self._error(
                        c,
                        "Unknown directive in @eval body",
                        code="K-VALID-DIR",
                    )
            if metric_count > 1:
                self._error(
                    node,
                    "Multiple metrics defined for the same procedure",
                    code="K-EVAL-MULTI-METRIC",
                )
            return suite_e

        self._error(
            kw_node,
            f"Unknown validation block keyword {kw_text!r}; expected 'test' or 'eval'",
            code="K-VALID-KW",
        )
        return None

    def _walk_test_case(self, node: Node) -> TestCase:
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        body = _field(node, "code")
        code = ""
        is_block = False
        if body is not None:
            for inner in _named_children(body):
                if inner.type == "backtick_line_stmt":
                    # Single-line backtick: extract the inline python
                    # body (without backticks).
                    bn = next(
                        (c for c in _named_children(inner) if c.type == "inline_python_expr"),
                        None,
                    )
                    if bn is not None:
                        code = self._walk_inline_python_expr(bn)
                    is_block = False
                elif inner.type == "python_block":
                    code = self._walk_python_block_code(inner)
                    is_block = True
        return TestCase(name=name, code=code, is_block=is_block, loc=self._span(node))

    def _walk_eval_dataset(self, node: Node, *, is_test: bool) -> EvalDataset:
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        code, is_block = self._walk_eval_entry_body(_field(node, "body"))
        return EvalDataset(name=name, code=code, is_block=is_block, loc=self._span(node))

    def _walk_eval_metric(self, node: Node) -> EvalMetric:
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        dataset_node = _field(node, "dataset")
        dataset_name = (
            _identifier_name(self.source, dataset_node) if dataset_node is not None else None
        )
        code, is_block = self._walk_eval_entry_body(_field(node, "body"))
        return EvalMetric(
            name=name,
            code=code,
            is_block=is_block,
            dataset_name=dataset_name,
            loc=self._span(node),
        )

    def _walk_eval_entry_body(self, body: Optional[Node]) -> tuple[str, bool]:
        if body is None:
            return ("", False)
        for inner in _named_children(body):
            if inner.type == "return_stmt":
                # The return's value is either inline_python_expr or
                # template_expr — eval entries always use the former
                # in practice.
                value = _field(inner, "value")
                if value is not None and value.type == "inline_python_expr":
                    return (self._walk_inline_python_expr(value), False)
                if value is not None:
                    return (_src_text(self.source, value), False)
                return ("", False)
            if inner.type == "return_block_stmt":
                code_node = _field(inner, "code")
                code = self._dedent_fenced(code_node) if code_node is not None else ""
                return (code, True)
        return ("", False)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


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
