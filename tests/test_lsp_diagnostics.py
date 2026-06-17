"""LSP feature tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lsprotocol import types as lsp

from kedi.lang.ast import Program, Span
from kedi.lang.parser.diagnostics import KediDiagnostic
from kedi.lsp import features

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_diagnostics_empty_for_valid_program():
    src = "@p():\n  = ok\n"
    assert features.compute_diagnostics(src, "example.kedi") == []


def test_diagnostics_accept_tools_fixture() -> None:
    path = Path("tests/files/tools.kedi").resolve()
    src = path.read_text(encoding="utf-8")
    assert features.compute_diagnostics(src, str(path)) == []


def test_diagnostics_surfaces_parser_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = KediDiagnostic(
        severity="warning",
        message="legacy warning",
        span=Span(0, 1, 1, 1, 1, 2),
        code="K-TEST-WARN",
    )
    prog = Program(parser_warnings=[warning])

    def _fake_parse(*_args: Any, **_kwargs: Any) -> Program:
        return prog

    monkeypatch.setattr(features, "parse_program", _fake_parse)
    diags = features.compute_diagnostics("= ok\n", "example.kedi")
    assert len(diags) == 1
    assert diags[0].code == "K-TEST-WARN"
    assert diags[0].severity == lsp.DiagnosticSeverity.Warning


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


def test_diagnostics_report_invalid_top_level_effort_value() -> None:
    diags = features.compute_diagnostics("> effort: extreme\n", "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-EFFORT-VALUE"
    assert "must be one of" in diag.message
    assert diag.range.start.line == 0


def test_diagnostics_report_invalid_profile_effort_value() -> None:
    src = "> profile: strict:\n    > effort: extreme\n"
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-EFFORT-VALUE"
    assert "must be one of" in diag.message
    assert diag.range.start.line == 1


def test_diagnostics_report_invalid_mcp_transport_value() -> None:
    src = "> mcp:\n    transport: websocket\n    url: http://localhost:8000\n"
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-MCP-TRANSPORT"
    assert "streamable-http" in diag.message
    assert diag.range.start.line == 1
    assert diag.range.start.character == 15


def test_diagnostics_accept_streamable_http_mcp_transport() -> None:
    src = "> mcp:\n    transport: streamable-http\n    url: http://localhost:8000/mcp\n"

    assert features.compute_diagnostics(src, "example.kedi") == []


def test_diagnostics_report_invalid_model_setting_key() -> None:
    src = "> settings:\n    imaginary: true\n"
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-SETTING-KEY"
    assert "model setting must be one of" in diag.message
    assert diag.range.start.line == 1


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
