"""LSP feature tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lsprotocol import types as lsp

from kedi.agent_adapter import AdapterCapabilities
from kedi.lang.ast import (
    OutputSegment,
    Program,
    Span,
    TemplateLine,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypeLiteral,
    TypePython,
    TypeRef,
    TypeUnion,
)
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


def test_diagnostics_warns_when_annotated_description_is_missing():
    src = "@p():\n  >> Capital is [capital: Annotated[str]].\n"
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    assert diags[0].code == "K-ANNOTATED-DESCRIPTION"
    assert diags[0].severity == lsp.DiagnosticSeverity.Warning
    assert 'Annotated[type, "description"]' in diags[0].message


def test_diagnostics_warns_for_annotated_inside_union_type():
    src = "@p():\n  >> Value: [value: Annotated[str] | int].\n"
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    assert diags[0].code == "K-ANNOTATED-DESCRIPTION"


def test_diagnostics_warns_when_annotated_description_is_not_string():
    src = "@p():\n  >> Capital is [capital: Annotated[str, int]].\n"
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    assert diags[0].code == "K-ANNOTATED-DESCRIPTION"
    assert "second argument should be a string literal" in diags[0].message


def test_diagnostics_warns_when_annotated_has_extra_metadata():
    src = '~Person(name: Annotated[str, "Full name", "unused"])\n= ok\n'
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    assert diags[0].code == "K-ANNOTATED-EXTRA"
    assert "extra Annotated metadata is ignored" in diags[0].message


def test_diagnostics_ignore_comments_inside_directive_bodies():
    mcp = "> mcp:\n    transport: http\n    # comment while editing\n    url: http://x\n"
    settings = (
        "> settings:\n    temperature: 0.2\n    ###\n    editing\n    ###\n    max_tokens: 5\n"
    )

    assert features.compute_diagnostics(mcp, "mcp.kedi") == []
    assert features.compute_diagnostics(settings, "settings.kedi") == []


def test_artifact_diagnostics_validate_values_and_adapter_capabilities() -> None:
    invalid = features.compute_diagnostics(
        "> artifacts:\n  threshold: invalid\n",
        "artifacts.kedi",
    )
    assert [diagnostic.code for diagnostic in invalid] == ["K-ARTIFACTS-VALUE"]
    assert invalid[0].range.start.line == 1
    assert invalid[0].range.start.character == len("  threshold: ")

    unsupported = features.compute_diagnostics(
        "> agent: acp\n> artifacts:\n  enabled: true\n",
        "artifacts.kedi",
    )
    assert {diagnostic.code for diagnostic in unsupported} == {"K-ADAPTER-CAPABILITY"}
    assert len(unsupported) == 1

    supported = features.compute_diagnostics(
        "> adapter: pydantic\n> artifacts:\n  enabled: true\n",
        "artifacts.kedi",
    )
    assert supported == []


def test_profile_artifact_diagnostics_follow_profile_adapter_capabilities() -> None:
    diagnostics = features.compute_diagnostics(
        "> profile: remote:\n  > agent: acp\n  > artifacts:\n    enabled: true\n",
        "artifacts.kedi",
    )

    assert len(diagnostics) == 1
    assert {diagnostic.code for diagnostic in diagnostics} == {"K-ADAPTER-CAPABILITY"}


def test_artifact_inline_python_is_exposed_as_python_range() -> None:
    source = "> artifacts:\n  threshold: `args.threshold`\n"

    ranges = features.compute_python_ranges(source)

    assert any(item["kind"] == "inline" and item["text"] == "args.threshold" for item in ranges)


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


def test_legacy_template_lines_receive_adapter_capability_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_structured_output=False),
    )
    output = OutputSegment(name="capital", loc=Span(0, 1, 1, 1, 1, 8))
    program = Program(source_text="Capital [capital]\n")

    diagnostics = features._scan_capability_scope(
        program,
        [TemplateLine(segments=[output])],
        features._AdapterDiagnosticState(shortname="codex", kind="agent-harness"),
        {},
        {},
        set(),
        {},
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "K-ADAPTER-CAPABILITY"
    assert "structured template output fields" in diagnostics[0].message


@pytest.mark.parametrize(
    ("src", "code", "message", "line", "character"),
    [
        (
            "> agent: foobarbaz\n",
            "K-ADAPTER-NOT-FOUND",
            "Unknown agent harness adapter 'foobarbaz'",
            0,
            len("> agent: "),
        ),
        (
            "> adapter: foobarbaz\n",
            "K-ADAPTER-NOT-FOUND",
            "Unknown agent framework adapter 'foobarbaz'",
            0,
            len("> adapter: "),
        ),
        (
            "> agent: pydantic\n",
            "K-ADAPTER-KIND",
            "cannot select 'pydantic'",
            0,
            len("> agent: "),
        ),
        (
            "> profile: invalid:\n    > adapter: codex\n",
            "K-ADAPTER-KIND",
            "cannot select 'codex'",
            1,
            len("    > adapter: "),
        ),
    ],
)
def test_diagnostics_report_invalid_agent_adapter_selection(
    src: str,
    code: str,
    message: str,
    line: int,
    character: int,
) -> None:
    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == code
    assert diag.severity == lsp.DiagnosticSeverity.Error
    assert message in diag.message
    assert diag.range.start.line == line
    assert diag.range.start.character == character


def test_diagnostics_error_for_codex_unsupported_output_format() -> None:
    src = "> agent: codex\n\n>> Return [file: FileUrl].\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-SCHEMA-FORMAT"
    assert diag.severity == lsp.DiagnosticSeverity.Error
    assert "FileUrl" in diag.message
    assert "uri" in diag.message
    assert "Annotated[str" in diag.message
    assert diag.range.start.line == 2
    assert diag.range.start.character == len(">> Return [file: ")


def test_diagnostics_error_for_nested_codex_unsupported_output_format() -> None:
    src = "~Artifact(file: FileUrl)\n\n> agent: codex\n\n>> Return [artifact: Artifact].\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-SCHEMA-FORMAT"
    assert "FileUrl" in diag.message
    assert diag.range.start.line == 4
    assert diag.range.start.character == len(">> Return [artifact: ")


def test_diagnostics_accept_provider_supported_output_formats() -> None:
    codex_src = "> agent: codex\n\n>> Return [email: Email] on [day: date].\n"
    claude_src = "> agent: claude\n\n>> Return [file: FileUrl] matching [pattern: Regex].\n"

    assert features.compute_diagnostics(codex_src, "codex.kedi") == []
    assert features.compute_diagnostics(claude_src, "claude.kedi") == []


def test_adapter_schema_format_diagnostics_cover_nested_type_expressions() -> None:
    supported_formats = frozenset({"email"})
    loop = TypeDef(
        name="Loop",
        fields=[TypeField(name="next", type_expr=TypeRef(name="Loop"))],
    )
    type_defs = {"Loop": loop}

    assert features._unsupported_type_formats(None, type_defs, supported_formats) == []
    assert (
        features._unsupported_type_formats(
            TypeLiteral(value="description"), type_defs, supported_formats
        )
        == []
    )
    assert (
        features._unsupported_type_formats(
            TypePython(code="dynamic_type"), type_defs, supported_formats
        )
        == []
    )
    assert (
        features._unsupported_type_formats(TypeRef(name="Missing"), type_defs, supported_formats)
        == []
    )
    assert (
        features._unsupported_type_formats(TypeRef(name="Loop"), type_defs, supported_formats) == []
    )
    assert features._unsupported_type_formats(TypeExpr(), type_defs, supported_formats) == []
    assert features._unsupported_type_formats(
        TypeApply(name="list", args=[TypeRef(name="FileUrl")]),
        type_defs,
        supported_formats,
    ) == [("FileUrl", "uri")]
    assert features._unsupported_type_formats(
        TypeUnion(left=TypeRef(name="Email"), right=TypeRef(name="Regex")),
        type_defs,
        supported_formats,
    ) == [("Regex", "regex")]

    diagnostics = features._template_line_capability_diagnostics(
        features._AdapterDiagnosticState(shortname="codex", kind="agent-harness"),
        TemplateLine(segments=[OutputSegment(name="file", type_expr=TypeRef(name="FileUrl"))]),
        type_defs,
    )
    assert diagnostics == []


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


def test_diagnostics_warn_when_adapter_lacks_skill_tool_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_tool_registration=False),
    )
    src = "> agent: codex\n\n> use: skills\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-CAPABILITY"
    assert diag.severity == lsp.DiagnosticSeverity.Warning
    assert "skill tool registration" in diag.message
    assert diag.range.start.line == 2
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


def test_diagnostics_warn_when_adapter_cannot_apply_native_approval_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_native_approvals=False),
    )
    src = "> adapter: dspy\n> approval: deny\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-NATIVE-APPROVAL"
    assert diag.severity == lsp.DiagnosticSeverity.Warning
    assert "builtin or MCP tools" in diag.message
    assert "Kedi-registered tools remain protected" in diag.message
    assert diag.range.start.line == 1
    assert diag.range.start.character == len("> approval: ")


def test_diagnostics_track_approval_policy_across_later_adapter_and_profile_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_native_approvals=False),
    )
    src = (
        "> approval: allow\n"
        "> adapter: dspy\n\n"
        "> profile: guarded:\n"
        "    > approval: deny\n\n"
        "> use: guarded\n"
    )

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 2
    assert {diag.code for diag in diags} == {"K-ADAPTER-NATIVE-APPROVAL"}
    assert {(diag.range.start.line, diag.range.start.character) for diag in diags} == {
        (0, len("> approval: ")),
        (4, len("    > approval: ")),
    }


def test_native_approval_warning_targets_an_inline_handler_value() -> None:
    src = "> agent: acp\n> approval: `approval_handler`\n"

    diags = features.compute_diagnostics(src, "example.kedi")

    assert len(diags) == 1
    diag = diags[0]
    assert diag.code == "K-ADAPTER-NATIVE-APPROVAL"
    assert diag.range.start.line == 1
    assert diag.range.start.character == len("> approval: ")
    assert diag.range.end.character == len("> approval: `approval_handler`")


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
        approval_members=[features.ProfileApprovalEntry(plain_value="deny")],
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


def test_native_approval_diagnostic_helpers_cover_suppression_and_value_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "> approval: deny   \n"
    loc = Span(
        start_byte=0,
        end_byte=len(source.encode("utf-8")),
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=len(source),
        source_path="example.kedi",
    )
    prog = Program(source_text=source)
    state = features._AdapterDiagnosticState(
        shortname="dspy",
        kind="agent-framework",
        approval_loc=loc,
    )

    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_native_approvals=False),
    )
    warned_spans: set[tuple[str | None, int, int]] = set()
    diagnostics = features._native_approval_capability_diagnostics(prog, state, warned_spans)
    assert len(diagnostics) == 1
    assert diagnostics[0].range.end.character == len("> approval: deny")
    assert features._native_approval_capability_diagnostics(prog, state, warned_spans) == []

    monkeypatch.setattr(
        features,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(supports_native_approvals=True),
    )
    assert features._native_approval_capability_diagnostics(prog, state, set()) == []

    assert features._directive_value_span(Program(), loc) is None
    assert features._directive_value_span(Program(source_text="> approval deny\n"), loc) is None
    assert features._directive_value_span(Program(source_text="> approval:   \n"), loc) is None


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
    src = "> import: profiles\n\n> export:\n  xd\n  get_proc\n"
    main.write_text(src, encoding="utf-8")
    assert features.compute_diagnostics(src, str(main)) == []


def test_diagnostics_accept_selective_submodule_import(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "profiles.kedi").write_text(
        "@get_profile() -> str:\n  = ok\n> export:\n  get_profile\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = "> import: package/profiles:\n  get_profile\n\n= <get_profile()>\n"
    main.write_text(src, encoding="utf-8")

    assert features.compute_diagnostics(src, str(main)) == []


def test_diagnostics_report_unexported_selective_import_name(tmp_path):
    module = tmp_path / "profiles.kedi"
    module.write_text("[visible: str] = yes\n> export:\n  visible\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    src = "> import: profiles:\n  missing\n\n= ok\n"
    main.write_text(src, encoding="utf-8")

    diagnostics = features.compute_diagnostics(src, str(main))

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "K-IMPORT-NAME"
    assert diagnostics[0].range.start.line == 1
    assert diagnostics[0].message == "module 'profiles' does not export 'missing'"


def test_diagnostics_limit_recovered_parse_errors_to_the_faulting_line() -> None:
    src = (
        "> import: profiles (module fixture)\n> profile: local:\n    > model: openai:gpt-4o-mini\n"
    )

    diagnostics = features.compute_diagnostics(src, "example.kedi")

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code == "K-PARSE"
    assert diagnostic.message == "unexpected syntax"
    assert diagnostic.range.start.line == 0
    assert diagnostic.range.end.line == 0
    assert diagnostic.range.start.character == 19
    assert diagnostic.range.end.character == 35


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


def test_diagnostics_report_corrupt_installed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "kedi-home" / "registry" / "profiles"
    package.mkdir(parents=True)
    (package / "package.kedi").write_text("# invalid manifest\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    source = "> import: profiles\n"
    main.write_text(source, encoding="utf-8")
    monkeypatch.setenv("KEDI_HOME", str(tmp_path / "kedi-home"))

    diagnostics = features.compute_diagnostics(source, str(main))

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code == "K-IMPORT-CORRUPT"
    assert "invalid package manifest" in diagnostic.message


def test_diagnostics_accept_internal_registry_import(tmp_path):
    main = tmp_path / "main.kedi"
    src = "> import: this\n\n= ok\n"
    main.write_text(src, encoding="utf-8")

    assert features.compute_diagnostics(src, str(main)) == []


@pytest.mark.parametrize(
    "source",
    [
        "> mcp:\n    transport http\n",
        "> settings:\n    temperature 0.2\n",
    ],
)
def test_diagnostics_handle_incomplete_directive_fields(source: str) -> None:
    diagnostics = features.compute_diagnostics(source, "example.kedi")

    assert diagnostics
    assert all(diagnostic.code == "K-PARSE" for diagnostic in diagnostics)


def test_diagnostics_report_invalid_template_escape() -> None:
    diagnostics = features.compute_diagnostics(">> value=a [result]\n", "example.kedi")

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "K-TEXT-ESCAPE"
    assert "Literal '=' must be escaped" in diagnostics[0].message


def test_missing_import_diagnostics_ignore_corrupt_import_without_a_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = Program()
    program.imports = [SimpleNamespace(module="broken", loc=None, name_locs={}, names=[])]
    monkeypatch.setattr(features, "import_resolution_error", lambda *_args: "invalid package")

    assert features._missing_import_diagnostics(program) == []
