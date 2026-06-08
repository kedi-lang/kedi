"""LSP feature tests."""

from __future__ import annotations

from lsprotocol import types as lsp

from kedi.lsp import features

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_diagnostics_empty_for_valid_program():
    src = "@p():\n  = ok\n"
    assert features.compute_diagnostics(src, "example.kedi") == []


def test_diagnostics_reports_duplicate_type_field():
    src = "~Item(name, name)\n= ok\n"
    diags = features.compute_diagnostics(src, "example.kedi")
    assert diags
    d = diags[0]
    assert d.severity == lsp.DiagnosticSeverity.Error
    assert "duplicate field" in d.message.lower()
    assert d.source == "kedi"
    assert d.range.start.line == 0


def test_diagnostics_reports_unterminated_block_comment():
    diags = features.compute_diagnostics("###\nbody never closed\n", "example.kedi")
    assert any("unterminated" in d.message.lower() for d in diags)


def test_diagnostics_reports_legacy_optimize_empty_message():
    src = "@p():\n  > optimize: empty:\n\n  = done\n"
    diags = features.compute_diagnostics(src, "example.kedi")
    assert any("must have an indented block" in d.message for d in diags)


def test_diagnostics_accept_module_import_export_syntax(tmp_path):
    profiles = tmp_path / "profiles.kedi"
    profiles.write_text("> export:\n  xd\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    src = "> import: profiles (file name without .kedi)\n\n> export:\n  xd\n  get_proc\n"
    main.write_text(src, encoding="utf-8")
    assert features.compute_diagnostics(src, str(main)) == []


def test_diagnostics_report_missing_imported_module(tmp_path):
    main = tmp_path / "main.kedi"
    src = "> import: profiles\n\n= ok\n"
    main.write_text(src, encoding="utf-8")

    diags = features.compute_diagnostics(src, str(main))

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-IMPORT-NOT-FOUND"
    assert "profiles" in diag.message
    assert diag.range.start.line == 0


def test_diagnostics_accept_internal_registry_import(tmp_path):
    main = tmp_path / "main.kedi"
    src = "> import: this\n\n= ok\n"
    main.write_text(src, encoding="utf-8")

    assert features.compute_diagnostics(src, str(main)) == []
