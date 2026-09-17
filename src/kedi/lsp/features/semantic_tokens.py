from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import List, Optional, Tuple

from lsprotocol import types as lsp
from tree_sitter import Node, Tree
from tree_sitter import Query as _TSQuery
from tree_sitter import QueryCursor as _TSQueryCursor

from kedi.lang.parser.parser import parse as _ts_parse
from kedi.lsp.coordinates import byte_column_to_utf16, utf16_length, with_source_coordinates
from kedi.lsp.features.parser_state import (
    _KEDI_HIGHLIGHTS_QUERY,
    _KEDI_INJECTIONS_QUERY,
    _PY_HIGHLIGHTS_QUERY,
    _PY_PARSER,
)

SEMANTIC_TOKEN_TYPES: List[str] = [
    "namespace",  # 0
    "type",  # 1
    "class",  # 2
    "enum",  # 3
    "interface",  # 4
    "struct",  # 5
    "typeParameter",  # 6
    "parameter",  # 7
    "variable",  # 8
    "property",  # 9
    "enumMember",  # 10
    "event",  # 11
    "function",  # 12
    "method",  # 13
    "macro",  # 14
    "keyword",  # 15
    "modifier",  # 16
    "comment",  # 17
    "string",  # 18
    "number",  # 19
    "regexp",  # 20
    "operator",  # 21
    "decorator",  # 22
]


SEMANTIC_TOKEN_MODIFIERS: List[str] = [
    "declaration",  # bit 0  = 1
    "definition",  # bit 1  = 2
    "readonly",  # bit 2  = 4
    "static",  # bit 3  = 8
    "deprecated",  # bit 4  = 16
    "abstract",  # bit 5  = 32
    "async",  # bit 6  = 64
    "modification",  # bit 7  = 128
    "documentation",  # bit 8  = 256
    "defaultLibrary",  # bit 9  = 512
]


_TT = {name: i for i, name in enumerate(SEMANTIC_TOKEN_TYPES)}


_TM = {name: 1 << i for i, name in enumerate(SEMANTIC_TOKEN_MODIFIERS)}


_CAPTURE_TO_LSP: dict = {
    # Comments
    "comment": ("comment", 0),
    "comment.line": ("comment", 0),
    "comment.block": ("comment", 0),
    # Strings
    "string": ("string", 0),
    "string.special": ("string", 0),
    "string.escape": ("string", 0),
    "escape": ("string", 0),
    # Numbers / literals
    "number": ("number", 0),
    "boolean": ("keyword", 0),
    "constant": ("variable", 0),
    "constant.builtin": ("keyword", 0),
    # Operators / punctuation
    "operator": ("operator", 0),
    # Only punctuation.special is semantically meaningful in Kedi (`@`,
    # `~`, `>`, ``` and backticks). Delimiters stay uncoloured to avoid
    # noisy paren / colon / bracket highlighting.
    "punctuation.special": ("operator", 0),
    # We deliberately leave ``punctuation.delimiter`` out — colouring
    # every paren / colon / bracket looks noisy.
    # Keywords
    "keyword": ("keyword", 0),
    "keyword.return": ("keyword", 0),
    "keyword.function": ("keyword", 0),
    "keyword.operator": ("keyword", 0),
    "keyword.import": ("keyword", 0),
    "keyword.control": ("keyword", 0),
    # Functions / methods
    "function": ("function", 0),
    "function.call": ("function", 0),
    "function.method": ("method", 0),
    "function.builtin": ("function", 0),
    "function.macro": ("macro", 0),
    "method": ("method", 0),
    # Types
    "type": ("type", 0),
    "type.builtin": ("type", 0),
    "type.definition": ("type", 0),
    "constructor": ("class", 0),
    "class": ("class", 0),
    # Variables / params / properties
    "variable": ("variable", 0),
    "variable.parameter": ("parameter", 0),
    "variable.builtin": ("variable", 0),
    "parameter": ("parameter", 0),
    "property": ("property", 0),
    "attribute": ("property", 0),
    "field": ("property", 0),
    # Misc identifiers
    "decorator": ("decorator", 0),
    "namespace": ("namespace", 0),
    "module": ("namespace", 0),
    "label": ("enumMember", 0),
    "tag": ("variable", 0),
    "embedded": ("operator", 0),
}


def _capture_to_token(name: str) -> Optional[Tuple[int, int]]:
    """Resolve a tree-sitter highlight capture to (token_type_idx, mods).

    Falls back through dotted prefixes (``a.b.c`` → ``a.b`` → ``a``)
    when the exact capture name isn't mapped.
    """
    parts = name.split(".")
    while parts:
        key = ".".join(parts)
        hit = _CAPTURE_TO_LSP.get(key)
        if hit is not None:
            tt, mods = hit
            return _TT[tt], mods
        parts.pop()
    return None


def _procedure_docstring_nodes(root: Node) -> List[Node]:
    out: List[Node] = []

    def visit(node: Node) -> None:
        if node.type == "procedure_def":
            body = node.child_by_field_name("body")
            if body is not None:
                named = [child for child in body.children if child.is_named]
                if named and named[0].type == "block_comment":
                    out.append(named[0])
        for child in node.children:
            if child.is_named:
                visit(child)

    visit(root)
    return out


def _emit_token_spans_for_range(
    *,
    src_lines: Sequence[str],
    emit: Callable[[int, int, int, int, int], None],
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    token_type: int,
    token_mods: int,
) -> None:
    """Emit one or more LSP semantic-token spans for a possibly multi-line range."""
    if start_line == end_line:
        emit(start_line, start_col, end_col - start_col, token_type, token_mods)
        return
    if start_line < len(src_lines):
        emit(
            start_line,
            start_col,
            max(0, utf16_length(src_lines[start_line]) - start_col),
            token_type,
            token_mods,
        )
    for line_idx in range(start_line + 1, end_line):
        if line_idx < len(src_lines):
            emit(line_idx, 0, utf16_length(src_lines[line_idx]), token_type, token_mods)
    if end_col > 0 and end_line < len(src_lines):
        emit(end_line, 0, end_col, token_type, token_mods)


@with_source_coordinates(1)
def compute_semantic_tokens_full(source: str, source_path: Optional[str]) -> lsp.SemanticTokens:
    """Run tree-sitter highlight queries and emit LSP semantic tokens.

    The pipeline mirrors what editors like Neovim/Helix do natively
    with tree-sitter:

      1. Parse the document with tree-sitter-kedi.
      2. Run ``tree-sitter-kedi/queries/highlights.scm`` → Kedi tokens.
      3. Run ``tree-sitter-kedi/queries/injections.scm`` → list of
         (range, embedded language) injection sites.
      4. For each Python injection, parse the region's raw bytes with
         tree-sitter-python and run *its own*
         ``queries/highlights.scm`` against the sub-tree.
      5. Translate sub-tree positions back into the Kedi document's
         coordinate space; merge with the Kedi tokens.
      6. Sort + delta-encode per the LSP spec.

    There is no hand-rolled Python (or Kedi) classifier — the SCM
    query files are the single source of truth.
    """
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    tree: Tree = _ts_parse(src_bytes)
    src_lines = source.split("\n")
    raw: List[Tuple[int, int, int, int, int]] = []

    # Collect Python-injection byte ranges *first* so any Kedi-level
    # token that falls inside them (e.g. the fallback ``(python_code)
    # @string.special`` capture in kedi's highlights.scm) can be
    # punched out — the upstream tree-sitter-python tokens are finer
    # and the LSP spec forbids overlapping semantic tokens.
    python_byte_ranges: List[Tuple[int, int]] = []
    if _KEDI_INJECTIONS_QUERY is not None:
        cursor = _TSQueryCursor(_KEDI_INJECTIONS_QUERY)
        for _pat_idx, caps in cursor.matches(tree.root_node):
            for region_node in caps.get("injection.content", []):
                python_byte_ranges.append((region_node.start_byte, region_node.end_byte))
    docstring_nodes = _procedure_docstring_nodes(tree.root_node)
    docstring_byte_ranges = [(node.start_byte, node.end_byte) for node in docstring_nodes]

    def in_python_region(node: Node) -> bool:
        for s, e in python_byte_ranges:
            if node.start_byte >= s and node.end_byte <= e:
                return True
        return False

    def in_docstring_region(node: Node) -> bool:
        for s, e in docstring_byte_ranges:
            if node.start_byte >= s and node.end_byte <= e:
                return True
        return False

    def emit(line: int, col: int, length: int, ttype: int, mods: int) -> None:
        if length <= 0:  # pragma: no cover - tree-sitter captures are non-empty.
            return
        raw.append((line, col, length, ttype, mods))

    def emit_node(node: Node, ttype: int, mods: int, base_line: int = 0, base_col: int = 0) -> None:
        sl, sc = node.start_point
        el, ec = node.end_point
        # Translate from the embedded sub-tree's coordinates back to
        # Kedi-document coordinates. The Python sub-tree was parsed in
        # isolation, so its line 0 col 0 corresponds to (base_line,
        # base_col) in the Kedi document. Lines past line 0 in the
        # sub-tree always start at col 0 (newlines are 1:1).
        kedi_sl = base_line + sl
        kedi_sc = (base_col + sc) if sl == 0 else sc
        kedi_el = base_line + el
        kedi_ec = (base_col + ec) if el == 0 else ec
        _emit_token_spans_for_range(
            src_lines=src_lines,
            emit=emit,
            start_line=kedi_sl,
            start_col=byte_column_to_utf16(src_lines[kedi_sl], kedi_sc),
            end_line=kedi_el,
            end_col=byte_column_to_utf16(src_lines[kedi_el], kedi_ec),
            token_type=ttype,
            token_mods=mods,
        )

    # ---- 1. Kedi tokens via highlights.scm ----------------------------
    if _KEDI_HIGHLIGHTS_QUERY is not None:
        _emit_query_captures(
            _KEDI_HIGHLIGHTS_QUERY,
            tree.root_node,
            emit_node,
            base_line=0,
            base_col=0,
            skip_if=lambda node: in_python_region(node) or in_docstring_region(node),
        )
    docstring_token = _capture_to_token("variable")
    if docstring_token is not None:
        ttype, mods = docstring_token
        for node in docstring_nodes:
            emit_node(node, ttype, mods)

    # ---- 2. Embedded Python regions via injections.scm ---------------
    if python_byte_ranges and _PY_PARSER is not None and _PY_HIGHLIGHTS_QUERY is not None:
        assert _KEDI_INJECTIONS_QUERY is not None
        cursor = _TSQueryCursor(_KEDI_INJECTIONS_QUERY)
        for _pat_idx, caps in cursor.matches(tree.root_node):
            for region_node in caps.get("injection.content", []):
                py_bytes = region_node.text
                if not py_bytes:  # pragma: no cover - injection captures are non-empty.
                    continue
                try:
                    py_tree = _PY_PARSER.parse(py_bytes)
                except Exception:  # pragma: no cover — defensive
                    continue
                py_base_line, py_base_col = region_node.start_point
                _emit_query_captures(
                    _PY_HIGHLIGHTS_QUERY,
                    py_tree.root_node,
                    emit_node,
                    base_line=py_base_line,
                    base_col=py_base_col,
                )

    # ---- 3. Resolve overlaps + LSP delta-encode ----------------------
    # The LSP spec forbids overlapping semantic tokens. Highlights
    # queries routinely produce overlap — e.g. Python's f-string
    # ``(string)`` covers the whole literal while ``(interpolation)``
    # and identifier captures sit inside it, and pattern refinement
    # (``(identifier) @variable`` vs ``(call ... @function)``) emits
    # two tokens on the same node. Resolve by "paint with specificity":
    # for each character column, the *smallest* covering token wins.
    resolved = _resolve_overlaps(raw)
    resolved.sort(key=lambda t: (t[0], t[1]))

    data: List[int] = []
    prev_line = 0
    prev_col = 0
    for line, col, length, ttype, mods in resolved:
        d_line = line - prev_line
        d_col = col - prev_col if d_line == 0 else col
        data.extend([d_line, d_col, length, ttype, mods])
        prev_line = line
        prev_col = col
    return lsp.SemanticTokens(data=data)


def _resolve_overlaps(
    tokens: List[Tuple[int, int, int, int, int]],
) -> List[Tuple[int, int, int, int, int]]:
    """Collapse overlapping tokens into non-overlapping runs.

    For each character position, the token with the *smallest* length
    that covers it wins (smaller = more specific). Ties go to whichever
    was emitted first. Output runs of equal classification are merged
    so we don't fragment unnecessarily.
    """
    by_line: dict = {}
    for line, col, length, ttype, mods in tokens:
        by_line.setdefault(line, []).append((col, length, ttype, mods))

    out: List[Tuple[int, int, int, int, int]] = []
    for line, line_tokens in by_line.items():
        max_col = max(c + n for c, n, _, _ in line_tokens)
        # `paint[i]` = (length, ttype, mods) where length acts as the
        # specificity tag. Smaller length wins on tie-break.
        paint: List[Optional[Tuple[int, int, int]]] = [None] * max_col
        # Apply tokens longest-first so shorter ones overwrite.
        for col, length, ttype, mods in sorted(line_tokens, key=lambda t: -t[1]):
            for i in range(col, col + length):
                if 0 <= i < max_col:
                    paint[i] = (length, ttype, mods)
        # Collapse contiguous runs of identical (ttype, mods).
        i = 0
        while i < max_col:
            slot = paint[i]
            if slot is None:
                i += 1
                continue
            _, ttype, mods = slot
            j = i
            while j < max_col:
                next_slot = paint[j]
                if next_slot is None or next_slot[1:] != (ttype, mods):
                    break
                j += 1
            out.append((line, i, j - i, ttype, mods))
            i = j
    return out


def _emit_query_captures(
    query: _TSQuery,
    root: Node,
    emit_node,
    *,
    base_line: int,
    base_col: int,
    skip_if=None,
) -> None:
    """Run a highlights.scm-style query and emit tokens for each capture.

    Pattern order matters: tree-sitter returns matches in the order
    patterns appear in the .scm file, and the convention is that
    later patterns refine earlier ones (``(identifier) @variable``
    early, ``(function_definition name: (identifier) @function)``
    later). We feed all captures into the emit pipeline and let the
    dedup pass in :func:`compute_semantic_tokens_full` keep the last
    one for any identical (line, col, length).

    ``skip_if(node) -> bool`` lets the caller drop captures that fall
    inside an injection region (so an outer-language fallback
    capture doesn't overlap the inner-language injection tokens).
    """
    cursor = _TSQueryCursor(query)
    for _pat_idx, caps in cursor.matches(root):
        for cap_name, nodes_list in caps.items():
            mapped = _capture_to_token(cap_name)
            if mapped is None:
                continue
            ttype, mods = mapped
            for n in nodes_list:
                if skip_if is not None and skip_if(n):
                    continue
                emit_node(n, ttype, mods, base_line, base_col)
