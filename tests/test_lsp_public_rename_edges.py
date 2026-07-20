"""LSP rename edge cases on public feature API."""

from __future__ import annotations

from typing import Any, cast

from kedi.lsp import features
from tests.lsp_helpers import prepare_rename_at, rename_at


def test_rename_returns_none_without_symbol(tmp_path) -> None:
    missing_src = "> import: missing\n= ok\n"
    main = tmp_path / "main.kedi"
    main.write_text(missing_src, encoding="utf-8")
    assert prepare_rename_at("= ok\n", "/tmp/x.kedi", "ok") is None
    assert rename_at("= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", "ok", "x") is None
    assert prepare_rename_at(missing_src, str(main), "missing") is None
    parameter_source = "@p(x: int):\n  = <x>\n"
    assert prepare_rename_at(parameter_source, "/tmp/x.kedi", "x") is not None
    parameter_rename = rename_at(
        parameter_source,
        "file:///tmp/x.kedi",
        "/tmp/x.kedi",
        "x",
        "y",
    )
    assert parameter_rename is not None and parameter_rename.changes is not None
    assert len(parameter_rename.changes["file:///tmp/x.kedi"]) == 2


def test_rename_rejects_external_imported_value(tmp_path, monkeypatch) -> None:
    external_span = features.Span(0, 1, 1, 1, 1, 2, source_path=str(tmp_path / "other.kedi"))
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef(
            "imported_value", "v", external_span, None, external_span
        ),
    )
    assert prepare_rename_at("= ok\n", "/tmp/x.kedi", "=") is None
    assert rename_at("= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", "=", "v2") is None


def test_rename_local_symbol_with_and_without_cursor_span(monkeypatch) -> None:
    rename_span = features.Span(0, 1, 1, 1, 1, 2)
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef("local", "x", rename_span, None, rename_span),
    )
    monkeypatch.setattr(
        features,
        "_find_all_references",
        lambda prog, sym, *, include_declaration: [rename_span],
    )
    rename_edit = rename_at("= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", "=", "y")
    assert rename_edit is not None and rename_edit.changes is not None
    assert len(rename_edit.changes["file:///tmp/x.kedi"]) == 1
    monkeypatch.setattr(
        features,
        "_resolve_symbol_at",
        lambda prog, pos: features.SymbolRef("local", "x", None, None, cast(Any, None)),
    )
    rename_without_cursor = rename_at("= ok\n", "file:///tmp/x.kedi", "/tmp/x.kedi", "=", "y")
    assert rename_without_cursor is not None and rename_without_cursor.changes is not None
    assert len(rename_without_cursor.changes["file:///tmp/x.kedi"]) == 1
