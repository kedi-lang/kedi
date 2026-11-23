"""Kedi #214 (6.6, 6.15/6.22): LSP coordinates must stay in their own units.

``Span`` columns are UTF-8 byte columns (tree-sitter convention), LSP
positions are UTF-16 code units, and Python string indexes are neither.
References, rename, and hover must keep working when non-ASCII or
non-BMP text precedes the identifier on the same line.
"""

from __future__ import annotations

from lsprotocol import types as lsp

from kedi.lang.ast import Span
from kedi.lsp.ast_spans import name_token_span
from kedi.lsp.features import hover as hover_module
from kedi.lsp.features import navigation as navigation_module
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.symbols import _name_token_span_on_line
from kedi.lsp.parsing import parse_program
from tests.lsp_helpers import references_at, rename_at

URI = "file:///tmp/x.kedi"
PATH = "/tmp/x.kedi"


def _utf16_position(source: str, needle: str, occurrence: int = 1, offset: int = 0) -> lsp.Position:
    """LSP position of ``needle`` measured in UTF-16 code units."""
    start = 0
    index = 0
    for _ in range(occurrence):
        index = source.index(needle, start)
        start = index + len(needle)
    index += offset
    before = source[:index]
    line_text = before.rsplit("\n", 1)[-1]
    return lsp.Position(
        line=before.count("\n"),
        character=len(line_text.encode("utf-16-le")) // 2,
    )


def test_references_include_use_after_non_ascii_text() -> None:
    src = "@helper(x):\n  >> H\xe9llo <x>\n  = <x>\n"
    refs = references_at(src, URI, PATH, "x", True)
    assert sorted({r.range.start.line for r in refs}) == [0, 1, 2]


def test_references_include_use_after_non_bmp_character() -> None:
    src = "@helper(x):\n  >> H\xe9llo \U0001f600 <x>\n  = <x>\n"
    refs = references_at(src, URI, PATH, "x", True)
    assert sorted({r.range.start.line for r in refs}) == [0, 1, 2]


def test_definition_from_unicode_line_use_resolves_to_declaration() -> None:
    src = "@helper(x):\n  >> H\xe9llo \U0001f600 <x>\n  = <x>\n"
    pos = _utf16_position(src, "<x>", occurrence=1, offset=1)
    loc = navigation_module.compute_definition(src, URI, PATH, pos)
    assert loc is not None
    assert loc.range.start.line == 0


def test_rename_from_unicode_line_use_renames_every_occurrence() -> None:
    src = "@helper(x):\n  >> H\xe9llo \U0001f600 <x>\n  = <x>\n"
    edit = rename_at(src, URI, PATH, "x", "name", occurrence=1)
    assert edit is not None
    changes = edit.changes or {}
    assert sorted({e.range.start.line for e in changes[URI]}) == [0, 1, 2]
    assert all(e.new_text == "name" for e in changes[URI])


def test_hover_on_unicode_line_use_names_the_parameter() -> None:
    src = "@helper(x):\n  >> H\xe9llo \U0001f600 <x>\n  = <x>\n"
    pos = _utf16_position(src, "<x>", occurrence=1, offset=1)
    hover = hover_module.compute_hover(src, PATH, pos)
    assert hover is not None
    assert isinstance(hover.contents, lsp.MarkupContent)
    assert "x" in hover.contents.value


def test_completion_after_non_ascii_text_offers_the_parameter() -> None:
    src = "@helper(x):\n  >> H\xe9llo <x\n  = <x>\n"
    pos = _utf16_position(src, "<x", occurrence=1, offset=2)
    items = compute_completion(src, PATH, pos)
    assert any(item.label == "x" for item in items)


def test_name_token_span_measures_names_in_utf8_bytes() -> None:
    loc = Span(
        start_byte=10,
        end_byte=20,
        start_line=1,
        start_col=5,
        end_line=1,
        end_col=15,
    )
    span = name_token_span(loc, "caf\xe9", prefix_len=1)
    assert span.start_byte == 11
    assert span.end_byte == 11 + len("caf\xe9".encode("utf-8"))
    assert span.start_col == 6
    assert span.end_col == 6 + len("caf\xe9".encode("utf-8"))


def test_synthesized_name_span_uses_byte_columns_after_non_ascii() -> None:
    src = "  # caf\xe9 tool\n@tool():\n  = ok\n"
    prog = parse_program(src, source_path=PATH)
    span = _name_token_span_on_line(prog, line_number=1, source_path=PATH, name="tool")
    assert span is not None
    prefix = "  # caf\xe9 "
    assert span.start_col == len(prefix.encode("utf-8")) + 1
    assert span.end_col == span.start_col + len("tool")
    assert src.encode("utf-8")[span.start_byte : span.end_byte] == b"tool"
