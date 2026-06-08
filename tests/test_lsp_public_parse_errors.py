"""LSP public API edge cases for unparseable sources."""

from __future__ import annotations

from tests.lsp_helpers import (
    definition_at,
    document_symbols_for,
    hover_at,
    inlay_hints_for_lines,
    position_of,
    prepare_rename_at,
    references_at,
    rename_at,
    signature_help_at,
)

_BAD = "@broken(:\n"
_URI = "file:///tmp/bad.kedi"
_PATH = "/tmp/bad.kedi"


def test_unparseable_source_returns_empty_lsp_results() -> None:
    pos = position_of(_BAD, "broken", offset=1)
    assert hover_at(_BAD, _PATH, "broken", offset=1) is None
    assert definition_at(_BAD, _URI, _PATH, "broken", offset=1) is None
    assert references_at(_BAD, _URI, _PATH, "broken", True, offset=1) == []
    assert prepare_rename_at(_BAD, _PATH, "broken", offset=1) is None
    assert rename_at(_BAD, _URI, _PATH, "broken", "x", offset=1) is None
    assert signature_help_at(_BAD, _PATH, "broken", offset=1) is None
    assert inlay_hints_for_lines(_BAD, _PATH, 0, 1) == []
    assert document_symbols_for(_BAD, _PATH) == []
    assert pos.line == 0
