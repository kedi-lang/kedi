from __future__ import annotations

import warnings
from pathlib import Path
from textwrap import indent

import pytest
from lsprotocol import types as lsp

import kedi
from kedi.artifacts import artifact_instructions
from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.diagnostics import compute_diagnostics
from kedi.lsp.features.semantic_tokens import (
    SEMANTIC_TOKEN_TYPES,
    compute_semantic_tokens_full,
)
from tests.lsp_helpers import decode_semantic_tokens, hover_text_at
from tests.mock_adapter import MockAdapter


@pytest.mark.parametrize("scope", ["top", "procedure", "profile"])
@pytest.mark.parametrize("multiline", [False, True])
def test_instructions_preserve_scoped_rendering(scope, multiline):
    text = "Use a <tone> tone."
    directive_source = (
        f"> instructions:\n  {text}\n  Keep it short.\n"
        if multiline
        else f"> instructions: {text}\n"
    )
    prompt = ">> The answer is [answer: str].\n= <answer>\n"
    prelude = "[tone] = direct\n"
    if scope == "procedure":
        source = prelude + "@reply():\n" + indent(directive_source + prompt, "  ")
        source += "\n= <reply()>\n"
    elif scope == "profile":
        source = prelude + "> profile: reviewer:\n" + indent(directive_source, "  ")
        source += "\n> use: reviewer\n" + prompt
    else:
        source = prelude + directive_source + prompt

    program = parse_program(source, source_path="review.kedi")
    assert program.parser_warnings == []
    adapter = MockAdapter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime = compile_program(program, adapter=adapter)
        try:
            runtime.run_main()
        finally:
            runtime.artifacts.close()
    assert caught == []
    expected = "Use a direct tone." + ("\nKeep it short." if multiline else "")
    assert adapter.instruction_calls == [f"{expected}\n\n{artifact_instructions()}"]


@pytest.mark.parametrize("scope", ["top", "procedure", "profile"])
@pytest.mark.parametrize("multiline", [False, True])
def test_removed_system_directive_is_rejected_without_warnings(scope, multiline):
    source = "> system:\n  Be concise.\n" if multiline else "> system: Be concise.\n"
    if scope == "procedure":
        source = "@reply():\n" + indent(source, "  ")
    elif scope == "profile":
        source = "> profile: reviewer:\n" + indent(source, "  ")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(KediParseError):
            parse_program(source)
    assert caught == []


def test_interactive_session_rejects_system_and_accepts_instructions():
    adapter = MockAdapter()
    with kedi.interactive(adapter=adapter, model=None) as session:
        with pytest.raises(KediParseError):
            session.execute("> system: Legacy instructions.\n")
        session.execute("> instructions: Updated instructions.\n")
        session.execute(">> The result is [result: str].\n")
    assert adapter.instruction_calls == [f"Updated instructions.\n\n{artifact_instructions()}"]


def test_later_instructions_replace_earlier_profile_instructions():
    source = (
        "> profile: reviewer:\n  > instructions: Earlier.\n  > instructions: Later.\n"
        "> use: reviewer\n>> The result is [result: str].\n= <result>\n"
    )
    adapter = MockAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        runtime.run_main()
    finally:
        runtime.artifacts.close()
    assert adapter.instruction_calls == [f"Later.\n\n{artifact_instructions()}"]


def test_instructions_in_imported_modules(tmp_path: Path):
    module = tmp_path / "reviewer.kedi"
    module.write_text(
        "> instructions: Imported instructions.\n@review():\n  = done\n> export:\n  review\n"
    )
    source = "> import: reviewer\n= <review()>\n"
    runtime = compile_program(parse_program(source, source_path=str(tmp_path / "main.kedi")))
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()


def test_compilation_does_not_emit_unrelated_editor_warnings():
    program = parse_program("[result: str] << Describe the supplied evidence.\n")
    assert [d.code for d in program.parser_warnings] == ["K-RAW-INVOKE-REDUNDANT-STR"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime = compile_program(program)
    runtime.artifacts.close()
    assert caught == []


def test_editor_warning_does_not_hide_a_real_parse_error():
    with pytest.raises(KediParseError, match="effort directive must be one of") as error:
        parse_program("[result: str] << Describe the evidence.\n> effort: invalid\n")
    assert error.value.diagnostics[0].code == "K-EFFORT-VALUE"
    assert error.value.diagnostics[-1].code == "K-RAW-INVOKE-REDUNDANT-STR"


def test_editor_recommends_instructions_and_rejects_the_removed_keyword():
    items = compute_completion("> ins", "review.kedi", lsp.Position(line=0, character=5))
    assert any(item.label == "instructions" for item in items)
    assert not any(
        item.label == "system"
        for item in compute_completion("> ", "review.kedi", lsp.Position(line=0, character=2))
    )
    assert compute_diagnostics("> instructions: Be concise.\n", "review.kedi") == []
    diagnostics = compute_diagnostics("> system: Be concise.\n", "review.kedi")
    assert diagnostics
    assert all(d.severity == lsp.DiagnosticSeverity.Error for d in diagnostics)
    assert all(not d.tags for d in diagnostics)
    assert "> instructions:" in hover_text_at(
        "> instructions: Be concise.\n", "review.kedi", "instructions"
    )


def test_instructions_are_highlighted():
    directive = "instructions"
    tokens = decode_semantic_tokens(
        compute_semantic_tokens_full(f"> {directive}: Be concise.\n", "review.kedi").data
    )
    keyword = SEMANTIC_TOKEN_TYPES.index("keyword")
    assert (0, 2, len(directive), keyword, 0) in tokens
