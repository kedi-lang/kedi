"""LSP hover edge cases on public feature API."""

from __future__ import annotations

import kedi.lang.ast as lang_ast_module
import kedi.lsp.features.hover as lsp_hover_module
import kedi.lsp.features.resolution as lsp_resolution_module
import kedi.lsp.features.symbols as lsp_symbols_module
from kedi.lsp import module_surface
from tests.lsp_helpers import definition_at, hover_text_at


def test_hover_reports_missing_import_target(tmp_path) -> None:
    main = tmp_path / "main.kedi"
    missing_src = "> import: missing\n= ok\n"
    main.write_text(missing_src, encoding="utf-8")
    hover = hover_text_at(missing_src, str(main), "missing")
    assert "could not be resolved" in hover
    assert "> import: missing" in hover


def test_hover_on_type_and_exported_value() -> None:
    module_src = "~Person(name)\n[x: int] = `1`\n> export:\n  Person\n  x\n"
    assert "Kedi type definition" in hover_text_at(
        module_src, "/tmp/mod.kedi", "Person", occurrence=2
    )
    assert "Module value" in hover_text_at(module_src, "/tmp/mod.kedi", "x", occurrence=3)


def test_hover_and_definition_on_imported_value(tmp_path) -> None:
    producer = tmp_path / "producer.kedi"
    producer.write_text("[value] = ok\n> export:\n  value\n", encoding="utf-8")
    consumer = tmp_path / "consumer.kedi"
    consumer_src = "> import: producer\n> export:\n  value\n= <`value`>\n"
    consumer.write_text(consumer_src, encoding="utf-8")
    hover = hover_text_at(consumer_src, str(consumer), "value")
    assert "Imported value" in hover
    module_src = "~Person(name)\n[x: int] = `1`\n> export:\n  Person\n  x\n"
    same_file_definition = definition_at(
        module_src,
        "file:///tmp/mod.kedi",
        "/tmp/mod.kedi",
        "Person",
        occurrence=2,
    )
    assert same_file_definition is not None and same_file_definition.uri == "file:///tmp/mod.kedi"


def test_hover_on_type_definition_name() -> None:
    module_src = "~Person(name)\n= ok\n"
    assert "Kedi type definition" in hover_text_at(module_src, "/tmp/mod.kedi", "Person")


def test_hover_on_unknown_call_local_and_field() -> None:
    unknown_call_src = "= <missing()>\n"
    assert "procedure not found" in hover_text_at(unknown_call_src, "/tmp/x.kedi", "missing")
    assert "Local" in hover_text_at("@p():\n  = <unknown>\n", "/tmp/x.kedi", "unknown")
    assert "Field" in hover_text_at("~Person(name)\n", "/tmp/x.kedi", "name")


def test_hover_on_plain_text_segment() -> None:
    generic = hover_text_at("@p():\n  >> plain prompt text\n", "/tmp/x.kedi", "plain")
    assert "Text segment" in generic


def test_hover_symbol_kind_branches(monkeypatch) -> None:
    monkeypatch.setattr(
        lsp_hover_module,
        "_resolve_symbol_at",
        lambda prog, pos: lsp_symbols_module.SymbolRef(
            kind="parameter",
            name="x",
            declaration_span=None,
            scope_proc=None,
            cursor_span=lang_ast_module.Span(0, 1, 1, 1, 1, 2),
        ),
    )
    assert "Parameter" in hover_text_at("= ok\n", "/tmp/x.kedi", "=")
    monkeypatch.setattr(
        lsp_hover_module,
        "_resolve_symbol_at",
        lambda prog, pos: lsp_symbols_module.SymbolRef(
            kind="type_field",
            name="field",
            declaration_span=None,
            scope_proc=None,
            cursor_span=lang_ast_module.Span(0, 5, 1, 1, 1, 6),
        ),
    )
    assert "Field" in hover_text_at("= ok\n", "/tmp/x.kedi", "=")
    monkeypatch.setattr(
        lsp_hover_module,
        "_resolve_symbol_at",
        lambda prog, pos: lsp_symbols_module.SymbolRef(
            kind="custom",
            name="thing",
            declaration_span=None,
            scope_proc=None,
            cursor_span=lang_ast_module.Span(0, 5, 1, 1, 1, 6),
        ),
    )
    assert "custom" in hover_text_at("= ok\n", "/tmp/x.kedi", "=")


def test_hover_imported_value_with_surface(monkeypatch) -> None:
    value_span = lang_ast_module.Span(0, 5, 1, 1, 1, 6)
    monkeypatch.setattr(
        lsp_hover_module,
        "_resolve_symbol_at",
        lambda prog, pos: lsp_symbols_module.SymbolRef(
            kind="imported_value",
            name="loose",
            declaration_span=value_span,
            scope_proc=None,
            cursor_span=value_span,
        ),
    )
    monkeypatch.setattr(
        lsp_resolution_module,
        "imported_surface_for_program",
        lambda prog: module_surface.ImportedSurface(
            values={"loose": module_surface.ImportedValueInfo("loose", value_span)}
        ),
    )
    assert "another Kedi module" in hover_text_at("= ok\n", "/tmp/x.kedi", "=")
