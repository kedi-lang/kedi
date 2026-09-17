"""Shared embedded-Python ranges used by editor features."""

from __future__ import annotations

from typing import List, Optional

from lsprotocol import types as lsp
from tree_sitter import Node, Tree

from kedi.lang.parser.parser import parse as _ts_parse
from kedi.lsp.coordinates import tree_sitter_point_to_position, with_source_coordinates


@with_source_coordinates(None)
def compute_python_ranges(source: str) -> List[dict]:
    """Return byte / line ranges of every Python region in the document.

    Each entry is::

        {
          "kind": "fenced" | "inline",
          "range": {"start": {"line": ..., "character": ...},
                     "end":   {"line": ..., "character": ...}},
          "text":  "<the Python source (dedented for fenced, raw for inline)>"
        }

    Consumed by the VS Code extension to assemble a virtual Python
    document that mirrors the Kedi file's coordinate space so requests
    can be forwarded to a Python language server without translation.
    """
    return _collect_python_ranges(source)


@with_source_coordinates(None)
def compute_header_python_ranges(source: str) -> List[dict]:
    return _collect_python_ranges(source, include_regular_inline=False)


def _collect_python_ranges(
    source: str,
    *,
    include_header_inline: bool = True,
    include_regular_inline: bool = True,
    include_directive_inline: bool = True,
) -> List[dict]:
    tree: Tree = _ts_parse(source.encode("utf-8") if isinstance(source, str) else source)
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    out: List[dict] = []

    def visit(node: Node, parents: tuple[Node, ...] = ()) -> None:
        if node.type == "python_code":
            if not include_regular_inline:
                return
            text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")
            from kedi.lang.parser.escapes import dedent_block

            dedented = "\n".join(dedent_block(_python_code_lines(text)))
            out.append(
                {
                    "kind": "fenced",
                    "range": _node_range(node, source),
                    "text": dedented,
                }
            )
        elif node.type == "python_inline_body":
            if not include_directive_inline and any(
                parent.type.endswith("_directive") for parent in parents
            ):
                return
            header_owner = _header_inline_owner(parents)
            if header_owner is not None and not include_header_inline:
                return
            if header_owner is None and not include_regular_inline:
                return
            text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")
            item = {
                "kind": "inline",
                "range": _node_range(node, source),
                "text": text,
            }
            if header_owner is not None:
                item.update(header_owner)
            out.append(item)
        for c in node.children:
            visit(c, parents + (node,))

    visit(tree.root_node)
    return out


def _position_in_python_hover_region(source: str, pos: lsp.Position) -> bool:
    return any(
        _position_in_range(pos, region["range"])
        for region in _collect_python_ranges(source, include_directive_inline=False)
    )


def _position_in_range(pos: lsp.Position, range_: dict) -> bool:
    start = range_["start"]
    end = range_["end"]
    if pos.line < start["line"] or pos.line > end["line"]:
        return False
    if pos.line == start["line"] and pos.character < start["character"]:
        return False
    if pos.line == end["line"] and pos.character >= end["character"]:
        return False
    return True


def _node_range(node: Node, source: str) -> dict:
    start = tree_sitter_point_to_position(source, *node.start_point)
    end = tree_sitter_point_to_position(source, *node.end_point)
    return {
        "start": {"line": start.line, "character": start.character},
        "end": {"line": end.line, "character": end.character},
    }


def _header_inline_owner(parents: tuple[Node, ...]) -> Optional[dict]:
    is_type_expression = any(parent.type == "type_python" for parent in parents)
    owner = next(
        (parent for parent in reversed(parents) if parent.type in {"param", "type_field"}),
        None,
    )
    container = next(
        (parent for parent in reversed(parents) if parent.type in {"procedure_def", "type_def"}),
        None,
    )
    if owner is None:
        is_output_type = any(parent.type == "output_segment" for parent in parents)
        if not is_type_expression or is_output_type or container is None:
            return None
        owner_kind = "return_type"
    else:
        owner_kind = owner.type
    return {
        "ownerKind": owner_kind,
        "containerKind": container.type if container is not None else None,
        "containerLine": container.start_point[0] if container is not None else None,
        "expressionKind": "type" if is_type_expression else "default",
    }


def _python_code_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines
