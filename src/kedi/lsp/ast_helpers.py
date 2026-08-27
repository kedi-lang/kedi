"""Generic AST and source-text helpers for LSP features."""

from __future__ import annotations

from typing import Optional

from ..lang.ast import (
    AssignBlockStmt,
    AssignStmt,
    ReassignBlockStmt,
    ReassignStmt,
    Span,
)
from ..lang.parser.parser import parse as _ts_parse
from .ast_spans import name_token_span

__all__ = [
    "assignment_name_span",
    "extract_module_docstring",
    "is_public_export_name",
]


def assignment_name_span(
    stmt: AssignStmt | AssignBlockStmt | ReassignStmt | ReassignBlockStmt,
) -> Optional[Span]:
    if stmt.loc is None:
        return None
    return name_token_span(stmt.loc, stmt.name, prefix_len=1)


def is_public_export_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def extract_module_docstring(source: str) -> Optional[str]:
    tree = _ts_parse(source)
    named = [child for child in tree.root_node.children if child.is_named]
    if not named:
        return None

    first = named[0]
    if first.type == "block_comment":
        lines = source[first.start_byte : first.end_byte].splitlines()
        if len(lines) < 2:
            return None
        text = "\n".join(lines[1:-1]).strip()
        return text or None

    if first.type == "line_comment":
        body: list[str] = []
        for node in named:
            if node.type != "line_comment":
                break
            line = source[node.start_byte : node.end_byte]
            body.append(line[1:].lstrip())
        text = "\n".join(body).strip()
        return text or None

    return None
