from __future__ import annotations

from pathlib import Path

import pytest
from lsprotocol import types as lsp

import kedi.lsp.features.directive_help as lsp_directive_help_module
import kedi.lsp.features.document as lsp_document_module
import kedi.lsp.features.resolution as lsp_resolution_module
import kedi.lsp.features.semantic_tokens as lsp_semantic_tokens_module
import kedi.lsp.features.symbols as lsp_symbols_module
from kedi.lang import parse_program
from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    InputSegment,
    ProfileAdapterEntry,
    ProfileAgentEntry,
    ProfileArtifactsEntry,
    ProfileDefStmt,
    ProfileModelEntry,
    ProfileOutputEntry,
    ProfileSettingsEntry,
    ProfileSystemEntry,
    SettingsField,
    Span,
    TemplateLine,
    TextSegment,
    TypeRef,
    UseDirectiveStmt,
)
from tests.lsp_helpers import (
    decode_semantic_tokens,
    definition_at,
    hover_text_at,
    position_of,
    references_at,
)


def test_semantic_tokens_cover_agent_profile_directives() -> None:
    src = (
        "> agent: codex\n"
        "> adapter: pydantic\n"
        "> model: haiku\n"
        "> effort: low\n"
        "> instructions: Be concise.\n"
        "> profile: light:\n"
        "    > agent: codex\n"
        "    > adapter: pydantic\n"
        "    > model: opus\n"
        "    > effort: high\n"
        "    > instructions: Keep answers short.\n"
        "    > mcp:\n"
        "        command: fs-mcp\n"
        "    > settings:\n"
        "        temperature: 0.2\n"
        "> use: light\n"
    )
    tokens = lsp_semantic_tokens_module.compute_semantic_tokens_full(src, "/tmp/profile.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    keyword_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("keyword")
    label_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("enumMember")
    string_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("string")

    assert any(line == 0 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 0 and token_type == label_type for line, _, _, token_type, _ in decoded)
    assert any(line == 1 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 1 and token_type == label_type for line, _, _, token_type, _ in decoded)
    assert any(line == 2 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 2 and token_type == string_type for line, _, _, token_type, _ in decoded)
    assert any(line == 3 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 3 and token_type == string_type for line, _, _, token_type, _ in decoded)
    assert any(line == 4 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 4 and token_type == string_type for line, _, _, token_type, _ in decoded)
    assert any(line == 5 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 5 and token_type == label_type for line, _, _, token_type, _ in decoded)
    assert any(line == 6 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 6 and token_type == label_type for line, _, _, token_type, _ in decoded)
    assert any(line == 7 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 7 and token_type == label_type for line, _, _, token_type, _ in decoded)
    assert any(line == 9 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 10 and token_type == string_type for line, _, _, token_type, _ in decoded)
    property_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("property")
    assert any(line == 11 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 12 and token_type == property_type for line, _, _, token_type, _ in decoded)
    assert any(line == 12 and token_type == string_type for line, _, _, token_type, _ in decoded)
    assert any(line == 13 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 14 and token_type == property_type for line, _, _, token_type, _ in decoded)
    assert any(line == 14 and token_type == string_type for line, _, _, token_type, _ in decoded)
    assert any(line == 15 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 15 and token_type == label_type for line, _, _, token_type, _ in decoded)


def test_semantic_tokens_cover_multiline_acp_agent_command() -> None:
    src = "> agent:\n    acp: uv run acp server\n"
    tokens = lsp_semantic_tokens_module.compute_semantic_tokens_full(src, "/tmp/acp.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    label_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("enumMember")
    string_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("string")

    assert (1, 4, 3, label_type, 0) in decoded
    assert (1, 9, len("uv run acp server"), string_type, 0) in decoded


def test_semantic_tokens_and_hover_cover_typed_agent_connections() -> None:
    src = (
        "> agent: a2a:\n"
        "    endpoint: https://agents.example.com/research\n"
        "    auth:\n"
        "        scheme: bearer\n"
        "        token_env: RESEARCH_A2A_TOKEN\n"
        "    request_timeout: `5.0`\n"
        "    task_timeout: 120\n"
        "> agent: acp:\n"
        "    command: `('uv', 'run', 'agent')`\n"
    )
    tokens = lsp_semantic_tokens_module.compute_semantic_tokens_full(src, "/tmp/remote.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    label_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("enumMember")
    property_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("property")
    assert (0, 9, len("a2a"), label_type, 0) in decoded
    assert (7, 9, len("acp"), label_type, 0) in decoded
    assert any(line == 1 and token_type == property_type for line, _, _, token_type, _ in decoded)
    assert any(line == 4 and token_type == property_type for line, _, _, token_type, _ in decoded)

    directive_hover = hover_text_at(src, "remote.kedi", "a2a")
    assert "structured-output extension" in directive_hover
    assert "Remote endpoints require HTTPS" in hover_text_at(src, "remote.kedi", "endpoint")
    assert "Bearer token" in hover_text_at(src, "remote.kedi", "token_env")
    assert "does not cancel remote work" in hover_text_at(src, "remote.kedi", "task_timeout")
    assert "ACP command" in hover_text_at(src, "remote.kedi", "command")


def test_semantic_tokens_and_hover_cover_raw_invoke() -> None:
    src = "[answer] << Hello <name>\n"
    tokens = lsp_semantic_tokens_module.compute_semantic_tokens_full(src, "/tmp/raw.kedi")
    decoded = decode_semantic_tokens(tokens.data)

    keyword_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("keyword")
    variable_type = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("variable")

    assert any(line == 0 and token_type == keyword_type for line, _, _, token_type, _ in decoded)
    assert any(line == 0 and token_type == variable_type for line, _, _, token_type, _ in decoded)
    assert "Raw model invoke" in hover_text_at(src, "raw.kedi", "<<")


def test_hover_on_model_profile_and_use_directives() -> None:
    src = """
> model: haiku
> agent: `agent_name`
> adapter: `adapter_name`
> effort: low
> instructions: Be concise.
> settings:
    parallel_tool_calls: true
> instructions: Be <tone>.
> profile: light:
    > agent: codex
    > adapter: pydantic
    > model: opus
    > effort: high
    > instructions: Keep answers short.
    > mcp:
        command: profile-mcp
    > instructions:
        Prefer <tone> answers.
    > settings:
        temperature: 0.2
        num_retries: 2
> mcp:
    transport: stdio
    command: vsh
    args: `["run", "--mcp"]`
> use: light
@foo():
    > agent: `agent_name`
    > adapter: `adapter_name`
    > mcp:
        command: local-mcp
    >> answer [x]
"""
    assert "> model:" in hover_text_at(src, "profile.kedi", "model", occurrence=1)
    profile_model_hover = hover_text_at(src, "profile.kedi", "model", occurrence=2)
    assert "Model override stored in this profile" in profile_model_hover
    assert "`opus`" in profile_model_hover
    assert "> agent:" in hover_text_at(src, "profile.kedi", "agent", occurrence=1)
    assert "`agent_name`" in hover_text_at(src, "profile.kedi", "agent_name", occurrence=1)
    assert "> adapter:" in hover_text_at(src, "profile.kedi", "adapter", occurrence=1)
    assert "`adapter_name`" in hover_text_at(src, "profile.kedi", "adapter_name", occurrence=1)
    assert "> effort:" in hover_text_at(src, "profile.kedi", "effort", occurrence=1)
    assert "> instructions:" in hover_text_at(src, "profile.kedi", "instructions", occurrence=1)
    assert "> settings:" in hover_text_at(src, "profile.kedi", "settings", occurrence=1)
    parallel_tool_calls_hover = hover_text_at(src, "profile.kedi", "parallel_tool_calls")
    assert "Pydantic AI `ModelSettings`" in parallel_tool_calls_hover
    assert "LangChain model settings" in parallel_tool_calls_hover
    assert "Input segment" in hover_text_at(src, "profile.kedi", "tone", occurrence=1)
    assert "Keep answers short." in hover_text_at(src, "profile.kedi", "Keep")
    assert "MCP server configuration stored in this profile" in hover_text_at(
        src, "profile.kedi", "mcp", occurrence=1
    )
    assert "Command used" in hover_text_at(src, "profile.kedi", "profile-mcp")
    assert "Text segment" in hover_text_at(src, "profile.kedi", "Prefer")
    assert "Model configuration stored in this profile" in hover_text_at(
        src, "profile.kedi", "settings", occurrence=2
    )
    temperature_hover = hover_text_at(src, "profile.kedi", "temperature")
    assert "Pydantic AI `ModelSettings`" in temperature_hover
    assert "DSPy `LM`" in temperature_hover
    assert "LangChain model settings" in temperature_hover
    assert "DSPy `LM` only" in hover_text_at(src, "profile.kedi", "num_retries")
    assert "> mcp:" in hover_text_at(src, "profile.kedi", "mcp", occurrence=3)
    transport_hover = hover_text_at(src, "profile.kedi", "transport")
    assert "Connection transport for this MCP server" in transport_hover
    assert "streamable-http" in transport_hover
    assert "stdio" in hover_text_at(src, "profile.kedi", "stdio")
    assert "Command used" in hover_text_at(src, "profile.kedi", "vsh")
    assert "String argument list" in hover_text_at(src, "profile.kedi", "args")
    assert "local-mcp" in hover_text_at(src, "profile.kedi", "local-mcp")
    assert "haiku" in hover_text_at(src, "profile.kedi", "haiku")
    assert "Agent profile" in hover_text_at(src, "profile.kedi", "light", occurrence=1)
    assert "> use:" in hover_text_at(src, "profile.kedi", "light", occurrence=2)
    assert "Agent harness stored in this profile" in hover_text_at(src, "profile.kedi", "codex")
    assert "Model framework stored in this profile" in hover_text_at(
        src, "profile.kedi", "pydantic"
    )
    assert "> agent:" in hover_text_at(src, "profile.kedi", "agent_name", occurrence=2)
    assert "> adapter:" in hover_text_at(src, "profile.kedi", "adapter_name", occurrence=2)


def test_document_symbols_cover_agent_and_adapter_directives() -> None:
    src = """
> profile: light:
    > agent: `agent_name`
    > adapter: `adapter_name`

@foo():
    > agent: `agent_name`
    > adapter: `adapter_name`
    >> answer [x]
"""
    symbols = lsp_document_module.compute_document_symbols(src, "profile.kedi")
    profile = next(symbol for symbol in symbols if symbol.name == "> profile: light")
    proc = next(symbol for symbol in symbols if symbol.name == "foo")

    assert profile.children is not None
    assert [child.name for child in profile.children[:2]] == [
        "> agent: `agent_name`",
        "> adapter: `adapter_name`",
    ]
    assert proc.children is not None
    assert [child.name for child in proc.children[:2]] == [
        "> agent: `agent_name`",
        "> adapter: `adapter_name`",
    ]


def test_profile_symbol_skips_agent_and_adapter_members_without_locations() -> None:
    profile = ProfileDefStmt(
        name="synthetic",
        agent_members=[ProfileAgentEntry(plain_value="codex")],
        adapter_members=[ProfileAdapterEntry(plain_value="pydantic")],
    )
    symbol = lsp_document_module._profile_def_symbol(profile)

    assert symbol.children == []


def test_directive_markdown_helpers_for_synthetic_nodes() -> None:
    assert "`agent_name`" in lsp_directive_help_module._agent_directive_markdown(
        AgentDirectiveStmt(backtick_code="agent_name")
    )
    assert "command" in lsp_directive_help_module._agent_directive_markdown(
        AgentDirectiveStmt(plain_value="acp", command_backtick_code='["uv", "run"]')
    )
    assert "`adapter_name`" in lsp_directive_help_module._adapter_directive_markdown(
        AdapterDirectiveStmt(backtick_code="adapter_name")
    )
    assert "`agent_name`" in lsp_directive_help_module._profile_agent_entry_markdown(
        ProfileAgentEntry(backtick_code="agent_name")
    )
    assert "uv run acp" in lsp_directive_help_module._profile_agent_entry_markdown(
        ProfileAgentEntry(plain_value="acp", command_plain_value="uv run acp")
    )
    assert "`adapter_name`" in lsp_directive_help_module._profile_adapter_entry_markdown(
        ProfileAdapterEntry(backtick_code="adapter_name")
    )


def test_hover_on_backtick_setting_and_settings_helpers() -> None:
    src = '> settings:\n    stop_sequences: `["END"]`\n'

    assert '`["END"]`' in hover_text_at(src, "profile.kedi", "stop_sequences")
    assert "no adapter; valid keys are" in lsp_directive_help_module._setting_adapter_label(
        "imaginary"
    )
    assert (
        lsp_directive_help_module._settings_field_value_preview(
            SettingsField(name="preload_tools", plain_values=["search", "fetch"])
        )
        == "`[search, fetch]`"
    )
    cwd_hover = lsp_directive_help_module._setting_adapter_label("cwd")
    assert "Claude Agent SDK settings" in cwd_hover
    assert "Codex App Server settings" in cwd_hover
    assert "ACP agent settings" in cwd_hover
    profile = ProfileDefStmt(
        name="empty_settings",
        settings_members=[ProfileSettingsEntry(fields=[])],
    )
    assert "Reusable agent profile." in lsp_directive_help_module._profile_signature_markdown(
        profile
    )


def test_profile_docstring_replaces_default_profile_hover() -> None:
    src = """
> profile: documented:
    ###
    Use this profile for short answers.
    ###
    > model: opus
    > instructions: Keep it tight.

> profile: fallback:
    > model: haiku
"""

    documented_hover = hover_text_at(src, "profile.kedi", "documented")
    assert "Use this profile for short answers." in documented_hover
    assert "Reusable agent settings" not in documented_hover
    assert "`> model:` members" not in documented_hover

    doc_comment_hover = hover_text_at(src, "profile.kedi", "short")
    assert "Use this profile for short answers." in doc_comment_hover
    assert "Reusable agent settings" not in doc_comment_hover

    fallback_hover = hover_text_at(src, "profile.kedi", "fallback")
    assert "Reusable agent profile." in fallback_hover
    assert "Model: `haiku`" in fallback_hover


def test_use_hover_when_module_has_no_exported_profiles(tmp_path: Path) -> None:
    (tmp_path / "utils.kedi").write_text(
        "@foo() -> str:\n  = ok\n\n> export:\n  foo\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = "> import: utils\n\n> use: utils\n"
    body = hover_text_at(src, str(main), "utils", occurrence=2)
    assert "no exported agent profiles" in body


def test_use_hover_hints_when_module_name_used_as_profile(tmp_path: Path) -> None:
    (tmp_path / "model_profile.kedi").write_text(
        "> profile: advanced:\n    > model: opus\n\n> export:\n  advanced\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    src = "> import: model_profile\n\n> use: model_profile\n"
    body = hover_text_at(src, str(main), "model_profile", occurrence=2)
    assert "imported **module**" in body
    assert "> use: advanced" in body


def test_resolve_symbol_skips_empty_toplevel_use_directive() -> None:
    src = "@main():\n    >> answer [x]\n"
    prog = parse_program(src)
    prog.ordered_toplevel.insert(
        0,
        UseDirectiveStmt(
            loc=Span(
                start_byte=0,
                end_byte=8,
                start_line=1,
                start_col=1,
                end_line=1,
                end_col=8,
            )
        ),
    )
    assert lsp_resolution_module._resolve_symbol_at(prog, lsp.Position(line=0, character=0)) is None


def test_resolve_symbol_skips_unknown_multiline_tool() -> None:
    src = """
> use:
    missing_tool

@main():
    >> answer [x]
"""
    prog = parse_program(src)
    pos = position_of(src, "missing_tool")
    sym = lsp_resolution_module._resolve_symbol_at(prog, pos)
    assert sym is not None
    assert sym.kind == "agent_tool"
    assert sym.name == "missing_tool"


def test_resolve_symbol_skips_unknown_tool_inside_procedure() -> None:
    src = """
@agent():
    > use:
        missing_tool
    >> answer [x]
"""
    prog = parse_program(src)
    pos = position_of(src, "missing_tool")
    sym = lsp_resolution_module._resolve_symbol_at(prog, pos)
    assert sym is not None
    assert sym.kind == "agent_tool"
    assert sym.name == "missing_tool"


def test_resolve_in_stmt_skips_empty_use_directive() -> None:
    src = "@agent():\n    >> answer [x]\n"
    prog = parse_program(src)
    proc = prog.procedures[0]
    loc = Span(
        start_byte=10,
        end_byte=16,
        start_line=2,
        start_col=5,
        end_line=2,
        end_col=11,
    )
    stmt = UseDirectiveStmt(loc=loc)
    pos = lsp.Position(line=1, character=5)
    proc_by_name = {p.name: p for p in prog.procedures}
    assert (
        lsp_resolution_module._resolve_in_stmt(
            prog,
            stmt,
            pos,
            proc,
            proc_by_name,
            {},
            {},
        )
        is None
    )


def test_resolve_in_stmt_use_name_off_cursor() -> None:
    src = """
@foo():
    = ok

@agent():
    > use: foo
    >> answer [x]
"""
    prog = parse_program(src)
    proc = next(p for p in prog.procedures if p.name == "agent")
    use_stmt = next(stmt for stmt in proc.body if isinstance(stmt, UseDirectiveStmt))
    pos = position_of(src, "answer")
    proc_by_name = {p.name: p for p in prog.procedures}
    assert (
        lsp_resolution_module._resolve_in_stmt(
            prog,
            use_stmt,
            pos,
            proc,
            proc_by_name,
            {},
            {},
        )
        is None
    )


def test_use_tool_entry_spans_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    prog = parse_program("@x():\n    = 1\n")
    assert lsp_symbols_module._use_tool_entry_spans(prog, UseDirectiveStmt()) == []
    assert lsp_symbols_module._line_start_byte(prog, 999) is None
    prog.source_text = None
    assert lsp_symbols_module._line_start_byte(prog, 1) is None
    assert lsp_symbols_module._use_single_target_span(prog, UseDirectiveStmt(target="x")) is None
    stmt = UseDirectiveStmt(
        tool_names=["missing"],
        loc=prog.procedures[0].loc,
    )
    monkeypatch.setattr(lsp_symbols_module, "_program_line_text", lambda *_args, **_kwargs: None)
    assert lsp_symbols_module._use_tool_entry_spans(prog, stmt) == []
    assert (
        lsp_symbols_module._use_single_target_span(prog, UseDirectiveStmt(target="x", loc=stmt.loc))
        is None
    )


def test_use_tool_entry_spans_deduplicates_same_line() -> None:
    src = "> use:\n    tool\n"
    prog = parse_program(src)
    use_stmt = prog.ordered_toplevel[0]
    assert isinstance(use_stmt, UseDirectiveStmt)
    use_stmt.tool_names = ["tool", "tool"]
    spans = lsp_symbols_module._use_tool_entry_spans(prog, use_stmt)
    assert len(spans) == 1


def test_hover_on_missing_agent_tool() -> None:
    src = "> use:\n    missing\n@main():\n    >> x [y]\n"
    body = hover_text_at(src, "main.kedi", "missing")
    assert "Agent tool" in body
    assert "No procedure named `missing`" in body


def test_profile_def_symbol_skips_members_without_loc() -> None:
    loc = Span(
        start_byte=0,
        end_byte=10,
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=10,
    )
    symbol = lsp_document_module._profile_def_symbol(
        ProfileDefStmt(
            name="p",
            loc=loc,
            model_members=[ProfileModelEntry(plain_value="haiku")],
            tool_names=["tool"],
        )
    )
    assert symbol.name == "> profile: p"
    child_names = [child.name for child in symbol.children or ()]
    assert not any(name.startswith("> model:") for name in child_names)
    assert any(name.startswith("> use:") for name in child_names)


def test_hover_on_profile_name_in_multiline_use_is_tool_entry() -> None:
    src = """
> profile: light:
    > model: opus

> use:
    light

@main():
    >> answer [x]
"""
    body = hover_text_at(src, "main.kedi", "light", occurrence=2)
    assert "Agent tool" in body
    assert "No procedure named `light`" in body


def test_hover_on_profile_composed_inside_profile_resolves_profile() -> None:
    src = """
> profile: readonlyfs:
    > instructions: Read files only.

> profile: filesystem:
    > use: readonlyfs

> use: filesystem
"""
    body = hover_text_at(src, "main.kedi", "readonlyfs", occurrence=2)
    assert "**Agent profile** `> profile: readonlyfs:`" in body
    assert "System instructions: configured" in body


def test_hover_on_unknown_profile_use_target_reports_missing_tool() -> None:
    src = """
> profile: filesystem:
    > use: missing
"""
    body = hover_text_at(src, "main.kedi", "missing")
    assert "Agent tool" in body
    assert "No procedure named `missing`" in body


def test_hover_on_tool_inside_procedure_multiline_use() -> None:
    src = """
@tool():
    = ok

@agent():
    > use:
        tool
    >> answer [x]
"""
    body = hover_text_at(src, "main.kedi", "tool", occurrence=2)
    assert "`@tool()`" in body


def test_hover_on_toplevel_procedure_used_as_single_line_tool() -> None:
    src = """
@foo():
    = ok

> use: foo
@agent():
    >> answer [x]
"""
    body = hover_text_at(src, "main.kedi", "foo", occurrence=2)
    assert "Agent tool" in body


def test_hover_on_single_line_use_explains_tools_and_profiles() -> None:
    body = hover_text_at("> use: profile_or_tool\n", "main.kedi", "use")
    assert "procedure target" in body
    assert "profile target" in body


def test_hover_on_skills_explains_explicit_skill_discovery() -> None:
    body = hover_text_at("> skills: enabled\n", "main.kedi", "skills")

    assert "skill discovery" in body
    assert "list_skills" in body
    assert "read_skill" in body


def test_hover_on_procedure_local_skills_explains_explicit_skill_discovery() -> None:
    source = "@ask():\n    > skills: enabled\n    >> Return [answer].\n"
    body = hover_text_at(source, "main.kedi", "skills")

    assert "skill discovery" in body


def test_hover_covers_expanded_top_level_and_profile_skill_settings() -> None:
    source = (
        "> skills:\n"
        "    enabled: true\n"
        "    include_registry: false\n\n"
        "> profile: researcher:\n"
        "    > skills:\n"
        "        enabled: true\n"
        "        max_skills: 5\n"
    )

    assert "Kedi skill registry" in hover_text_at(source, "main.kedi", "include_registry")
    assert "Skill discovery policy stored in this profile" in hover_text_at(
        source, "main.kedi", "skills", occurrence=2
    )
    assert "Maximum number of skill names" in hover_text_at(source, "main.kedi", "max_skills")


def test_hover_on_procedure_used_as_single_line_tool() -> None:
    src = """
@foo():
    = ok

@agent():
    > use: foo
    >> answer [x]
"""
    body = hover_text_at(src, "main.kedi", "foo", occurrence=2)
    assert "Agent tool" in body


def test_hover_on_single_line_backtick_use_tool() -> None:
    src = """
@web_fetch(url: str) -> str:
    = `url`

> use: `web_fetch`

@agent():
    >> answer [x]
"""
    body = hover_text_at(src, "main.kedi", "web_fetch", occurrence=2)
    assert "Agent tool" in body


def test_hover_on_profile_used_inside_procedure() -> None:
    src = """
> profile: light:
    > model: opus

@agent():
    > use: light
    >> answer [x]
"""
    body = hover_text_at(src, "main.kedi", "light", occurrence=2)
    assert "Agent profile" in body


def test_hover_on_profile_name_inside_procedure_multiline_use_is_tool_entry() -> None:
    src = """
> profile: light:
    > model: opus

@agent():
    > use:
        light
    >> answer [x]
"""
    body = hover_text_at(src, "main.kedi", "light", occurrence=2)
    assert "Agent tool" in body
    assert "No procedure named `light`" in body


def test_hover_on_tool_name_in_multiline_use() -> None:
    src = """
@web_fetch(url: str) -> str:
    = `url`

> use:
    web_fetch

@agent():
    >> answer [x]
"""
    body = hover_text_at(src, "tools.kedi", "web_fetch", occurrence=2)
    assert "Agent tool" in body
    assert "web_fetch" in body


def test_hover_on_multiline_use_tools() -> None:
    src = """
@web_fetch(url: str) -> str:
    = `url`

> use:
    web_fetch

@agent():
    >> answer [x]
"""
    body = hover_text_at(src, "tools.kedi", "use")
    assert "Registers Kedi procedures as agent tools" in body
    assert "web_fetch" in body


def test_document_symbols_include_profile_definition() -> None:
    src = """
@web_search():
    = ok

> profile: light:
    > model: opus
    > instructions: Be concise.
    > use: web_search
@foo():
    > model: haiku
    > instructions: Keep answers short.
    > use:
        web_search
    >> x [y]
"""
    symbols = lsp_document_module.compute_document_symbols(src, "profile.kedi")
    names = [symbol.name for symbol in symbols]
    assert "> profile: light" in names
    profile = next(symbol for symbol in symbols if symbol.name == "> profile: light")
    profile_children = [child.name for child in profile.children or ()]
    assert any(child.startswith("> instructions:") for child in profile_children)
    assert any(child.startswith("> use:") for child in profile_children)
    proc = next(symbol for symbol in symbols if symbol.name == "foo")
    child_names = [child.name for child in proc.children or ()]
    assert any(name.startswith("> model:") for name in child_names)
    assert any(name.startswith("> instructions:") for name in child_names)
    assert any(name.startswith("> use:") and "web_search" in name for name in child_names)


def test_profile_def_symbol_without_tools() -> None:
    loc = Span(
        start_byte=0,
        end_byte=10,
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=10,
    )
    symbol = lsp_document_module._profile_def_symbol(
        ProfileDefStmt(
            name="light",
            loc=loc,
            model_members=[
                ProfileModelEntry(plain_value="opus", loc=loc),
            ],
            system_members=[
                ProfileSystemEntry(
                    template_lines=[
                        TemplateLine(
                            segments=[
                                TextSegment(text="Use "),
                                InputSegment(name="tone"),
                                TextSegment(text=" tone."),
                            ]
                        )
                    ],
                    loc=loc,
                ),
                ProfileSystemEntry(plain_value="Skip me."),
            ],
            artifacts_members=[ProfileArtifactsEntry()],
        )
    )
    child_names = [child.name for child in symbol.children or ()]
    assert any(name.startswith("> model:") for name in child_names)
    assert any(name == "> instructions: Use <tone> tone." for name in child_names)
    assert not any("Skip me" in name for name in child_names)
    assert not any(name.startswith("> use:") for name in child_names)

    signature = lsp_directive_help_module._profile_signature_markdown(
        ProfileDefStmt(name="empty", system_members=[ProfileSystemEntry()])
    )
    assert "Reusable agent profile." in signature


def test_artifact_directive_and_profile_field_hovers() -> None:
    source = (
        "> artifacts:\n"
        "  enabled: true\n"
        "  query_artifacts: enabled\n"
        "  threshold: `args.threshold`\n"
        "\n"
        "> profile: compact:\n"
        "  > artifacts:\n"
        "    store: memory\n"
    )

    assert "large-value storage" in hover_text_at(
        source, "artifacts.kedi", "artifacts", occurrence=1
    )
    assert "Minimum payload size" in hover_text_at(source, "artifacts.kedi", "threshold")
    assert "bounded lexical" in hover_text_at(source, "artifacts.kedi", "query_artifacts")
    assert "Artifact policy stored in this profile" in hover_text_at(
        source, "artifacts.kedi", "artifacts", occurrence=3
    )
    assert "Payload backend" in hover_text_at(source, "artifacts.kedi", "store")


def test_profile_artifacts_appear_in_profile_hover_and_document_symbols() -> None:
    source = "> profile: compact:\n  > artifacts:\n    enabled: true\n    threshold: 1kb\n"

    profile_hover = hover_text_at(source, "artifacts.kedi", "compact")
    assert "Artifact settings: 2" in profile_hover

    symbols = lsp_document_module.compute_document_symbols(source, "artifacts.kedi")
    profile = next(symbol for symbol in symbols if symbol.name == "> profile: compact")
    assert any(child.name == "> artifacts: enabled, threshold" for child in profile.children or ())


def test_profile_output_type_supports_navigation_and_document_symbols() -> None:
    source = "~Result(value: int)\n> profile: worker:\n  > output: Result\n"
    path = "/tmp/profile-output.kedi"
    uri = f"file://{path}"

    location = definition_at(source, uri, path, "Result", occurrence=2)
    assert location is not None
    assert location.range.start.line == 0

    references = references_at(source, uri, path, "Result", include_declaration=True)
    assert len(references) == 2

    symbols = lsp_document_module.compute_document_symbols(source, path)
    profile = next(symbol for symbol in symbols if symbol.name == "> profile: worker")
    assert any(child.name == "> output: Result" for child in profile.children or ())


def test_profile_output_symbol_skips_synthetic_entry_without_location() -> None:
    profile = ProfileDefStmt(
        name="synthetic",
        output_members=[ProfileOutputEntry(type_annotation="str", type_expr=TypeRef(name="str"))],
    )

    assert lsp_document_module._profile_def_symbol(profile).children == []
