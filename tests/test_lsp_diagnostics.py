"""LSP feature tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lsprotocol import types as lsp

from kedi.agent_adapter import AdapterCapabilities
from kedi.lang.ast import Program, Span
from kedi.lang.parser.diagnostics import KediDiagnostic
from kedi.lsp import features

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_diagnostics_empty_for_valid_program():
    src = "@p():\n  = ok\n"
    assert features.compute_diagnostics(src, "example.kedi") == []


def test_diagnostics_accept_annotated_output_description():
    src = '@p():\n  >> Capital is [capital: Annotated[str, "City name"]].\n'
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


def test_diagnostics_errors_when_adapter_lacks_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_structured_output=False),
    )
    src = "> agent: codex\n\n>> Capital is [capital].\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-CAPABILITY"
    assert diag.severity == lsp.DiagnosticSeverity.Error
    assert "structured template output fields" in diag.message
    assert diag.range.start.line == 2
    assert diag.range.start.character == len(">> Capital is ")


def test_diagnostics_warn_when_adapter_lacks_tool_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_tool_registration=False),
    )
    src = "> agent: codex\n\n@lookup():\n  = ok\n\n> use: lookup\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-CAPABILITY"
    assert diag.severity == lsp.DiagnosticSeverity.Warning
    assert "agent tool registration" in diag.message
    assert diag.range.start.line == 5
    assert diag.range.start.character == len("> use: ")


def test_diagnostics_warn_when_adapter_lacks_mcp_support() -> None:
    src = "> agent: codex\n\n> mcp:\n    command: fs-mcp\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-CAPABILITY"
    assert diag.severity == lsp.DiagnosticSeverity.Warning
    assert "MCP server directives" in diag.message
    assert diag.range.start.line == 2
    assert diag.range.start.character == 2


def test_diagnostics_warn_for_acp_unsupported_capabilities_from_registry() -> None:
    src = """
> agent: acp
> agent:
    acp: `["vsh", "run", "--acp"]`

@lookup():
  = ok

> use: lookup
> mcp:
    command: fs-mcp

>> Capital is [capital].
""".lstrip()

    diags = features.compute_diagnostics(src, "example.kedi")

    messages = [diag.message for diag in diags]
    assert len(diags) == 3
    assert any("agent tool registration" in message for message in messages)
    assert any("MCP server directives" in message for message in messages)
    assert any("structured template output fields" in message for message in messages)
    assert sorted(diag.severity for diag in diags) == [
        lsp.DiagnosticSeverity.Error,
        lsp.DiagnosticSeverity.Warning,
        lsp.DiagnosticSeverity.Warning,
    ]


def test_diagnostics_warn_for_nested_scoped_unsupported_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(
            supports_structured_output=False,
            supports_tool_registration=False,
            supports_mcp=False,
        ),
    )
    src = """
> profile: harnessed:
    > agent: codex
    > agent: codex
    > mcp:
        command: harness-mcp

> profile: blocked:
    > adapter: pydantic
    > adapter: pydantic
    > mcp:
        command: profile-mcp
    > use: listed_tool

> profile: mcp_only:
    > adapter: pydantic
    > mcp:
        command: mcp-only

> profile: tools_only:
    > adapter: pydantic
    > use: listed_tool

> agent: `agent_name`
> agent: codex

@listed_tool():
  = ok

@outer():
  > agent: `agent_name`
  > mcp:
      command: local-mcp
  > use:
      listed_tool

  @inner():
    >> Inner [value].
    = <value>

  >> Outer [result].
  = <result>

> adapter: `adapter_name`
> adapter: pydantic
> use: missing_profile
> use: blocked
> use: mcp_only
> use: tools_only
""".lstrip()

    diags = features.compute_diagnostics(src, "example.kedi")

    messages = [diag.message for diag in diags]
    assert any("MCP server directives" in message for message in messages)
    assert any("agent tool registration" in message for message in messages)
    assert any("structured template output fields" in message for message in messages)
    assert any(diag.severity == lsp.DiagnosticSeverity.Error for diag in diags)
    assert any(diag.severity == lsp.DiagnosticSeverity.Warning for diag in diags)
    assert any(diag.range.start.line == 3 and diag.range.start.character == 6 for diag in diags)
    assert any(diag.range.start.line == 9 and diag.range.start.character == 6 for diag in diags)
    assert any(diag.range.start.line == 33 and diag.range.start.character == 6 for diag in diags)
    assert any(diag.range.start.line == 36 for diag in diags)


def test_adapter_capability_diagnostic_helpers_cover_defensive_edges() -> None:
    profile = features.ProfileDefStmt(
        agent_members=[
            features.ProfileAgentEntry(plain_value="codex", loc=Span(0, 1, 1, 1, 1, 2)),
            features.ProfileAgentEntry(plain_value="codex", loc=Span(1, 2, 2, 1, 2, 2)),
        ],
        adapter_members=[
            features.ProfileAdapterEntry(plain_value="pydantic", loc=Span(2, 3, 3, 1, 3, 2)),
            features.ProfileAdapterEntry(plain_value="pydantic", loc=Span(3, 4, 4, 1, 4, 2)),
        ],
    )
    assert features._profile_adapter_state(profile).shortname == "pydantic"

    partial_profile = features.ProfileDefStmt(
        agent_members=[
            features.ProfileAgentEntry(plain_value="codex"),
            features.ProfileAgentEntry(plain_value="pydantic", loc=Span(4, 5, 5, 1, 5, 2)),
        ],
        adapter_members=[
            features.ProfileAdapterEntry(plain_value="pydantic"),
            features.ProfileAdapterEntry(plain_value=None, loc=Span(5, 6, 6, 1, 6, 2)),
        ],
    )
    assert features._profile_adapter_state(partial_profile).shortname is None
    assert features._adapter_state_from_plain_value(None, "agent-harness") is None
    assert features._adapter_state_from_plain_value("pydantic", "agent-harness") is None

    state = features._AdapterDiagnosticState(shortname="codex", kind="agent-harness")
    assert features._merge_adapter_state(state, features._AdapterDiagnosticState()) == state
    assert (
        features._use_capability_diagnostics(
            Program(),
            features.UseDirectiveStmt(),
            state,
            {},
            {},
            set(),
        )
        == []
    )
    assert features._directive_keyword_span(Program(), None, "mcp") is None
    assert (
        features._directive_keyword_span(
            Program(),
            Span(0, 1, 1, 1, 1, 2),
            "mcp",
        )
        is None
    )
    assert (
        features._directive_keyword_span(
            Program(source_text="> model: pydantic\n"),
            Span(0, 17, 1, 1, 1, 18),
            "mcp",
        )
        is None
    )


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
