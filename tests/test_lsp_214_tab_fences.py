"""Kedi #214 (6.8): tab-indented fenced Python blocks.

Visual indent width and source-character offsets are different units:
one tab is four visual columns but a single source character. The
virtual-Python builder must dedent by visual columns while keeping every
source map in character (UTF-16) offsets, or the emitted Python is
corrupted (``return`` became ``urn``).
"""

from __future__ import annotations

import ast

from kedi.lsp.python_virtual.builder import (
    PythonRegion,
    VirtualBuilder,
    _dedent_char_index,
)
from kedi.lsp.python_virtual.completion import _mapped_range
from kedi.lsp.python_virtual.document import compute_python_virtual_document

TAB_SRC = (
    "@calc(numbers: list[int]) -> int:\n"
    "  [total: int] = ```\n"
    "\treturn sum(numbers)\n"
    "\t```\n"
    "  = <`total`>\n"
)


def test_tab_fenced_block_virtual_document_parses() -> None:
    virtual = compute_python_virtual_document(TAB_SRC, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "return sum(numbers)" in virtual["text"]


def test_tab_fenced_block_source_map_uses_character_offsets() -> None:
    virtual = compute_python_virtual_document(TAB_SRC, "/tmp/x.kedi")
    mapping = next(m for m in virtual["mappings"] if m["sourceRange"]["start"]["line"] == 2)
    # The tab is one UTF-16 character; the code starts after it.
    assert mapping["sourceRange"]["start"] == {"line": 2, "character": 1}
    assert mapping["sourceRange"]["end"] == {"line": 2, "character": 20}
    virtual_line = virtual["text"].splitlines()[mapping["virtualRange"]["start"]["line"]]
    assert virtual_line.strip() == "return sum(numbers)"
    assert mapping["virtualRange"]["end"]["character"] == len(virtual_line)


def test_tab_fenced_completion_edit_maps_back_to_source() -> None:
    virtual = compute_python_virtual_document(TAB_SRC, "/tmp/x.kedi")
    mapping = next(m for m in virtual["mappings"] if m["sourceRange"]["start"]["line"] == 2)
    v_start = mapping["virtualRange"]["start"]["character"]
    # The ``numbers`` token: virtual 11..18 past the line start, source
    # 12..19 (one tab character precedes it on the source line).
    edit_range = {
        "start": {"line": mapping["virtualRange"]["start"]["line"], "character": v_start + 11},
        "end": {"line": mapping["virtualRange"]["start"]["line"], "character": v_start + 18},
    }
    mapped = _mapped_range(virtual, edit_range)
    assert mapped is not None
    assert mapped["start"] == {"line": 2, "character": 12}
    assert mapped["end"] == {"line": 2, "character": 19}


def test_builder_tab_dedent_keeps_full_line() -> None:
    builder = VirtualBuilder("\treturn x\n", None)
    region = PythonRegion(
        kind="fenced",
        range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 10}},
        text="return x",
    )
    builder.add_mapped_region(region)
    assert builder.lines == ["return x"]
    assert builder.mappings[0]["sourceRange"]["start"] == {"line": 0, "character": 1}


def test_builder_mixed_space_and_tab_dedent() -> None:
    source = "  return a\n\treturn b\n"
    builder = VirtualBuilder(source, None)
    region = PythonRegion(
        kind="fenced",
        range={"start": {"line": 0, "character": 0}, "end": {"line": 1, "character": 10}},
        text="return a\nreturn b",
    )
    builder.add_mapped_region(region)
    ast.parse("\n".join(builder.lines))
    assert builder.lines == ["return a", "return b"]


def test_dedent_char_index_counts_tabs_as_four_columns() -> None:
    assert _dedent_char_index("\treturn x", 4) == 1
    assert _dedent_char_index("    return x", 4) == 4
    assert _dedent_char_index(" \treturn x", 4) == 2
    assert _dedent_char_index("  return x", 4) == 2
