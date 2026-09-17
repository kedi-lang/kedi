from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

from tree_sitter import Node

from kedi.lang.ast import Span
from kedi.lang.parser.diagnostics import KediDiagnostic
from kedi.lang.parser.escapes import dedent_block
from kedi.source_loading import CompositeSourceMap

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker


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


def _validate_block_comments(self: _Walker) -> None:
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


def _line_error(
    self: _Walker, line_index: int, message: str, *, code: Optional[str] = None
) -> None:
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


def _record_tree_error(self: _Walker, node: Node) -> None:
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
    self: _Walker,
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


def _parse_error_message(self: _Walker, node: Node, fallback: str) -> str:
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
