"""Regression tests for shared LSP coordinate conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lsprotocol import types as lsp

from kedi.lang.ast import Span
from kedi.lsp import coordinates


def test_coordinate_conversions_handle_utf16_and_tree_sitter_edges() -> None:
    assert coordinates.utf16_to_index("🐈a", 0) == 0
    assert coordinates.utf16_to_index("🐈a", 2) == 1
    assert coordinates.utf16_to_index("🐈a", 3) == 2
    assert coordinates.byte_column_to_utf16("🐈a", 4) == 2

    source = coordinates.CoordinateSource("🐈\nİstanbul", None)
    assert source.byte_offset_to_position(-1) == lsp.Position(line=0, character=0)
    assert source.byte_offset_to_position(999) == lsp.Position(line=1, character=8)
    assert source.tree_sitter_point_to_position(-1, 0) == lsp.Position(line=0, character=0)
    assert source.tree_sitter_point_to_position(999, 9) == lsp.Position(line=1, character=8)


def test_coordinate_decorator_normalizes_source_path_inputs() -> None:
    @coordinates.with_source_coordinates(1)
    def position(source: str, source_path: Any = None) -> lsp.Range:
        return coordinates.span_to_range(
            Span(
                start_byte=0,
                end_byte=len(source.encode("utf-8")),
                start_line=1,
                start_col=1,
                end_line=1,
                end_col=len(source) + 1,
                source_path=None,
            )
        )

    assert position("🐈", 123).end.character == 2
    assert position(source="🐈", source_path=123).end.character == 2
    assert position(source="🐈", source_path="/tmp/source.kedi").end.character == 2
    assert position("🐈", "/tmp/source.kedi").end.character == 2


def test_external_coordinate_source_and_fallback_paths(tmp_path: Path, monkeypatch: Any) -> None:
    external = tmp_path / "external.kedi"
    external.write_text("İstanbul", encoding="utf-8")
    span = Span(
        start_byte=0,
        end_byte=len("İstanbul".encode("utf-8")),
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=9,
        source_path=str(external),
    )

    with coordinates.source_coordinates("main", str(tmp_path / "main.kedi")):
        assert coordinates.span_to_range(span).end.character == 8

    with coordinates.source_coordinates("main", None):
        assert coordinates.span_to_range(span).end.character == 8

    missing = Span(0, 1, 2, 3, 2, 4, source_path=str(tmp_path / "missing.kedi"))
    with coordinates.source_coordinates("main", str(tmp_path / "main.kedi")):
        assert coordinates.span_to_range(missing).start == lsp.Position(line=1, character=2)

    coordinates._read_source.cache_clear()
    monkeypatch.setattr(
        Path, "read_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )
    assert coordinates._read_source(str(external)) is None

    def fail_resolve(_path: Path) -> Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    same_path_span = Span(0, 1, 1, 1, 1, 2, source_path="same.kedi")
    with coordinates.source_coordinates("main", "same.kedi"):
        assert coordinates._span_source(same_path_span) is not None
    different_path_span = Span(0, 1, 1, 1, 1, 2, source_path="other.kedi")
    with coordinates.source_coordinates("main", "same.kedi"):
        assert coordinates._span_source(different_path_span) is None
