from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from tree_sitter import Node

from kedi.lang.ast import (
    BacktickLineStmt,
    CallArg,
    CallSegment,
    InputSegment,
    LiteralArg,
    OutputSegment,
    PythonExprNativeSegment,
    PythonExprSegment,
    Segment,
    Span,
    Stmt,
    TemplateBlock,
    TemplateLine,
    TextSegment,
)
from kedi.lang.parser.cst_helpers import (
    _decode_inline_python_body,
    _field,
    _identifier_name,
    _named_children,
    _src_text,
)
from kedi.lang.parser.diagnostics import KediDiagnostic
from kedi.lang.parser.escapes import dedent_block, unescape_text_with_tracking

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker


def _unescape_text_segment(self: _Walker, node: Node, raw: str) -> tuple[str, set[int]]:
    try:
        return unescape_text_with_tracking(raw)
    except ValueError as exc:
        self._error(node, str(exc), code="K-TEXT-ESCAPE")
        return raw, set()


def _template_line_from_expr(self: _Walker, expr_node: Node) -> TemplateLine:
    return TemplateLine(
        segments=self._walk_template_expr(expr_node),
        loc=self._span(expr_node),
    )


def _system_line_from_expr(self: _Walker, expr_node: Node) -> TemplateLine:
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


def _walk_template_block_stmt(self: _Walker, node: Node) -> TemplateBlock:
    lines: List[TemplateLine] = []
    for child in node.children:
        if child.type == "template_prompt_expr":
            lines.append(self._template_line_from_expr(child))
        elif child.type == "template_blank_line":
            lines.append(TemplateLine(segments=[], loc=self._span(child)))
    return TemplateBlock(lines=lines, loc=self._span(node))


def _walk_template_block_statements(self: _Walker, node: Node) -> List[Stmt]:
    """Split legacy backtick statements out of a ``>>`` continuation.

    A bare inline Python expression is a valid prompt segment after ``>>``.
    Historically, however, a continuation containing a Python *statement*
    (for example ``value = value.strip()``) remained a standalone
    backtick statement. Tree-sitter deliberately parses both forms as a
    template continuation so plain expressions can stay in the block; this
    AST pass restores the statement boundary without restricting Python
    expression syntax in the grammar.
    """

    block = self._walk_template_block_stmt(node)
    statements: List[Stmt] = []
    pending_lines: List[TemplateLine] = []

    for index, line in enumerate(block.lines):
        code = self._standalone_backtick_continuation_code(line) if index else None
        if code is None:
            pending_lines.append(line)
            continue
        if pending_lines:
            statements.append(self._validated_template_block(pending_lines, block.loc))
            pending_lines = []
        statements.append(BacktickLineStmt(code=code, loc=line.loc))

    if pending_lines:
        statements.append(self._validated_template_block(pending_lines, block.loc))
    return statements


def _validated_template_block(
    self: _Walker, lines: List[TemplateLine], loc: Span | None
) -> TemplateBlock:
    self._validate_template_block_scope(lines)
    return TemplateBlock(lines=lines, loc=loc)


def _validate_template_block_scope(self: _Walker, lines: List[TemplateLine]) -> None:
    produced_names: set[str] = set()
    seen_output_names: set[str] = set()
    for line in lines:
        for output in self._condition_outputs(line.segments):
            if output.name in seen_output_names:
                self.diagnostics.append(
                    KediDiagnostic(
                        severity="error",
                        message=(
                            f"Template block declares output field {output.name!r} more than once."
                        ),
                        span=output.loc or line.loc or Span(0, 0, 1, 1, 1, 1),
                        code="K-TEMPLATE-DUPLICATE-OUTPUT",
                    )
                )
            else:
                seen_output_names.add(output.name)
        for seg in line.segments:
            for name in sorted(_read_names(seg) & produced_names):
                self.diagnostics.append(
                    KediDiagnostic(
                        severity="error",
                        message=(
                            f"Template block cannot read LLM output '{name}' "
                            "before the block has finished. A row under `>>` is part of "
                            "the same prompt, even after a blank line; read it in a "
                            "separate statement (for example `[value] = ...`) or a new "
                            "`>>` block."
                        ),
                        span=seg.loc or line.loc or Span(0, 0, 1, 1, 1, 1),
                        code="K-TEMPLATE-BLOCK-FORWARD-OUTPUT",
                    )
                )
            if isinstance(seg, OutputSegment):
                produced_names.add(seg.name)


def _read_names(seg: Segment) -> set[str]:
    """Names a prompt segment reads when it is rendered."""
    if isinstance(seg, InputSegment):
        return {seg.name}
    if isinstance(seg, (PythonExprSegment, PythonExprNativeSegment)):
        try:
            tree = ast.parse(seg.code.strip(), mode="eval")
        except SyntaxError:
            return set()  # Reported where the expression is compiled.
        return {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
    return set()


def _walk_template_line(self: _Walker, node: Node, *, deprecated: bool = False) -> TemplateLine:
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


def _walk_template_expr(self: _Walker, node: Node) -> List[Segment]:
    segs = [self._walk_segment(c) for c in _named_children(node) if c.is_named]
    self._strip_leading_template_indent(segs)
    return segs


def _sole_inline_python_expr(node: Node | None) -> Node | None:
    """Return the sole inline expression in an initialization, assignment, or return RHS."""
    if node is None:
        return None
    if node.type == "inline_python_expr":
        return node
    children = _named_children(node)
    if len(children) == 1 and children[0].type == "inline_python_expr":
        return children[0]
    return None


def _strip_leading_template_indent(self: _Walker, segs: List[Segment]) -> None:
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


def _walk_segment(self: _Walker, node: Node) -> Segment:
    t = node.type
    if t in ("text_segment", "condition_text_segment"):
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
    if t == "inline_python_expr":
        return PythonExprSegment(
            code=self._walk_inline_python_expr(node),
            loc=self._span(node),
        )
    self._error(node, f"unexpected segment {t!r}")
    # Fallback so callers don't need None checks.
    return TextSegment(text="", loc=self._span(node))


def _walk_call_segment(self: _Walker, node: Node) -> CallSegment:
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


def _walk_call_arg(self: _Walker, node: Node) -> CallArg:
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
        lit = LiteralArg(value="__NATIVE_BACKTICK__", loc=self._span(node), native_backtick=code)
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


def _walk_segment_in_call(self: _Walker, node: Node) -> Segment:
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


def _walk_inline_python_expr(self: _Walker, node: Node) -> str:
    body = _field(node, "code")
    if body is None:
        return ""
    return _decode_inline_python_body(self.source, body)


def _dedent_fenced(self: _Walker, code_node: Node) -> str:
    """Extract and dedent a python_code node's body."""
    raw = _src_text(self.source, code_node)
    # Strip the trailing newline (if any) before splitting so we
    # don't get a phantom empty last line, then re-join.
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    dedented = dedent_block(lines)
    return "\n".join(dedented)
