"""Named groups: ``> evolve:`` bodies, group names and the canonical text of a form.

A group body runs in its enclosing scope, so it cannot change what surrounds it: it cannot
return, define procedures or types, change directives, or nest another group. Its canonical
text drops comments and normalizes indentation by nesting depth; everything else, including
template whitespace, is kept, because it is what the model receives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from tree_sitter import Node

from kedi.lang.ast import GroupStmt, ProcedureDef, TemplateBlock
from kedi.lang.parser.cst_helpers import _field, _identifier_name, _named_children

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker

_COMMENTS = frozenset({"line_comment", "block_comment"})
_GROUPS = frozenset({"evolve_directive", "optimize_directive"})
_RETURNS = frozenset({"return_stmt", "return_block_stmt"})
_DEFINITIONS = {
    "procedure_def": "procedure definitions",
    "type_def_stmt": "type definitions",
    "module_import": "imports",
    "module_export": "exports",
    "validation_block": "test or eval blocks",
}


def walk_evolve_directive(
    self: _Walker, node: Node, proc: Optional[ProcedureDef]
) -> Optional[GroupStmt]:
    name_node = _field(node, "name")
    body_node = _field(node, "body")
    assert name_node is not None and body_node is not None
    name = _identifier_name(self.source, name_node)
    if not _valid_group_body(self, body_node):
        return None
    group = GroupStmt(
        kind="evolve",
        name=name,
        body=self._walk_block_statements(body_node, proc),
        canonical=canonical_block_text(self.source, body_node),
        name_loc=self._span(name_node),
        loc=self._span(node),
    )
    _register_group(self, group, proc, name_node)
    return group


def optimize_group(
    self: _Walker, node: Node, name: str, block: TemplateBlock, proc: ProcedureDef
) -> GroupStmt:
    """The group an ``> optimize:`` span forms; the optimizer keeps owning its template lines."""
    body_node = _field(node, "body")
    name_node = _field(node, "name")
    group = GroupStmt(
        kind="optimize",
        name=name,
        body=[block],
        canonical="" if body_node is None else canonical_block_text(self.source, body_node),
        name_loc=None if name_node is None else self._span(name_node),
        loc=block.loc,
    )
    _register_group(self, group, proc, node)
    return group


def _register_group(
    self: _Walker, group: GroupStmt, proc: Optional[ProcedureDef], node: Node
) -> None:
    """Groups and optimize spans share one name space per module, by qualified name."""
    qualified = group.name if proc is None else f"{proc.name}.{group.name}"
    if qualified in self.groups:
        self._error(
            node,
            f"Duplicate group name {qualified!r}: '> evolve:' groups and '> optimize:' spans "
            "share one name space",
            code="K-GROUP-DUP",
        )
    self.groups[qualified] = group


def _valid_group_body(self: _Walker, body: Node) -> bool:
    valid = True
    pending = list(_named_children(body))
    while pending:
        child = pending.pop()
        kind = child.type
        if kind in _GROUPS:
            self._error(
                child,
                "Groups cannot be nested: an '> evolve:' body cannot contain '> evolve:' or "
                "'> optimize:'",
                code="K-EVOLVE-NESTED",
            )
        elif kind in _RETURNS:
            self._error(
                child,
                "An '> evolve:' body cannot return; bind the values the program reads after it",
                code="K-EVOLVE-RETURN",
            )
        elif kind in _DEFINITIONS or kind.endswith("_directive"):
            what = _DEFINITIONS.get(kind) or f"'> {kind.removesuffix('_directive')}:' directives"
            self._error(child, f"An '> evolve:' body cannot contain {what}", code="K-EVOLVE-BODY")
        else:
            pending.extend(_named_children(child))
            continue
        valid = False
    return valid


def canonical_block_text(source: bytes, block: Node) -> str:
    """The comment-free text of a statement block, indented two spaces per nesting level."""
    lines: List[str] = []
    _append_block(source, block, 0, lines)
    return "\n".join(lines)


def _append_block(source: bytes, block: Node, depth: int, lines: List[str]) -> None:
    for statement in _named_children(block):
        if statement.type not in _COMMENTS:
            _append_statement(source, statement, depth, lines)


def _append_statement(source: bytes, statement: Node, depth: int, lines: List[str]) -> None:
    base = statement.start_point[1]
    cursor = statement.start_byte
    for block in _nested_blocks(statement):
        _append_text(source, statement, cursor, block.start_byte, base, depth, lines)
        _append_block(source, block, depth + 1, lines)
        cursor = block.end_byte
    _append_text(source, statement, cursor, statement.end_byte, base, depth, lines)


def _nested_blocks(node: Node) -> List[Node]:
    """The outermost statement blocks inside a statement, in source order."""
    blocks: List[Node] = []
    for child in _named_children(node):
        if child.type == "block":
            blocks.append(child)
        else:
            blocks.extend(_nested_blocks(child))
    return blocks


def _append_text(
    source: bytes,
    statement: Node,
    start: int,
    end: int,
    base: int,
    depth: int,
    lines: List[str],
) -> None:
    kept: List[bytes] = []
    cursor = start
    for comment in _comments_between(statement, start, end):
        comment_start, comment_end = max(cursor, comment.start_byte), min(comment.end_byte, end)
        kept.append(source[cursor:comment_start])
        # Keep the comment's line breaks so lines stay aligned with the source.
        kept.append(b"\n" * source[comment_start:comment_end].count(b"\n"))
        cursor = max(cursor, comment_end)
    kept.append(source[cursor:end])
    texts = b"".join(kept).decode().split("\n")
    for text, original in zip(texts, source[start:end].decode().split("\n"), strict=True):
        if not text.strip() and original.strip():
            continue  # the line held only a comment
        dedented = text[min(base, len(text) - len(text.lstrip(" "))) :]
        lines.append("  " * depth + dedented if dedented else "")
    while lines and not lines[-1].strip():
        lines.pop()


def _comments_between(node: Node, start: int, end: int) -> List[Node]:
    found: List[Node] = []
    for child in node.children:
        if child.end_byte <= start or child.start_byte >= end or child.type == "block":
            continue
        if child.type in _COMMENTS:
            found.append(child)
        else:
            found.extend(_comments_between(child, start, end))
    return sorted(found, key=lambda comment: comment.start_byte)
