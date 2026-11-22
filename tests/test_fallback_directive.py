from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest
from lsprotocol import types as lsp

from kedi.agent_profile import AgentProfile
from kedi.agent_profile.compile import profile_stmt_to_definition
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.core import KediRuntime
from kedi.explain import explain_program
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import FallbackDirectiveStmt
from kedi.lang.parser import KediParseError
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.diagnostics import compute_diagnostics
from kedi.lsp.features.document import compute_document_symbols
from tests.lsp_helpers import hover_text_at
from tests.mock_adapter import MockAdapter


def evaluate(source: str, **env) -> AgentProfile:
    program = parse_program(source)
    definitions = {
        name: profile_stmt_to_definition(stmt) for name, stmt in program.profile_defs.items()
    }
    runtime = KediRuntime()
    try:
        return eval_profile_definition(runtime, definitions["selected"], env, registry=definitions)
    finally:
        runtime.artifacts.close()


def test_inline_and_block_fallback_keep_model_ids_and_expressions():
    program = parse_program("""> fallback: openai:first
> fallback:
    openai:second
    `alternative`
""")
    first, second = program.toplevel_body
    assert isinstance(first, FallbackDirectiveStmt)
    assert first.values[0].plain_value == "openai:first"
    assert isinstance(second, FallbackDirectiveStmt)
    assert second.values[0].plain_value == "openai:second"
    assert second.values[1].backtick_code == "alternative"


@pytest.mark.parametrize("flag, expected", [(True, ("first", "second")), (False, ("third",))])
def test_conditional_fallback_members_replace_in_source_order(flag, expected):
    profile = evaluate(
        """> profile: selected:
    > fallback: initial
    > if: `flag`:
        > fallback:
            first
            `"second"`
    > else:
        > fallback: third
    > model: primary
""",
        flag=flag,
    )
    assert profile.model == "primary"
    assert profile.fallback_models == expected


def test_profile_only_fallback_and_computed_list_clear():
    profile = evaluate(
        """> profile: earlier:
    > model: primary
    > fallback: original
> profile: selected:
    > use: earlier
    > fallback: `alternatives`
""",
        alternatives=[],
    )
    assert profile.model == "primary"
    assert profile.fallback_models == ()
    assert AgentProfile(fallback_models=("base",)).merge(profile).fallback_models == ()


@pytest.mark.parametrize("value", [None, 3, True, {}, ["valid", 4], [""], " "])
def test_invalid_computed_fallback_values_fail_before_model_io(value):
    with pytest.raises((TypeError, ValueError)):
        evaluate("> profile: selected:\n    > fallback: `value`\n", value=value)


@pytest.mark.parametrize("literal", ['"openai:model"', "['one']", "{'one': 1}"])
def test_plain_python_values_require_backticks(literal):
    with pytest.raises(KediParseError):
        parse_program(f"> fallback: {literal}\n")


def test_capability_cannot_configure_fallback_even_in_an_unselected_branch():
    with pytest.raises(KediParseError, match="not allowed in a capability"):
        parse_program("""> capability: unsupported:
    > if: `False`:
        > fallback: alternative
""")


def test_profile_merge_retains_independent_fallback_and_detaches_input_list():
    alternatives = ["second"]
    base = AgentProfile(model="first", fallback_models=alternatives)
    alternatives.append("third")
    changed = base.merge(AgentProfile(model="other"))
    assert changed.fallback_models == ("second",)
    assert base.fallback_models == ("second",)
    assert changed.needs_adapter_override("other")
    assert base.merge(AgentProfile(fallback_models=())).fallback_models == ()
    with pytest.raises(TypeError):
        AgentProfile(fallback_models="invalid")


def test_procedure_local_fallback_does_not_mutate_captured_outer_profile():
    class Recorder(MockAdapter):
        capabilities = replace(MockAdapter.capabilities, supports_model_fallback=True)

        def __init__(self):
            super().__init__(default_model="primary")
            self.fallbacks = []

        @contextmanager
        def profile_override(self, profile):
            self.fallbacks.append(profile.fallback_models)
            with super().profile_override(profile):
                yield

    adapter = Recorder()
    runtime = compile_program(
        parse_program("""> fallback: outer
@local():
    > fallback: inner
    >> The value is [value].
    = `value`
@caller():
    >> The value is [value].
    = `value`
> show: `local()`
= `caller()`
"""),
        adapter=adapter,
    )
    try:
        runtime.run_main()
        assert adapter.fallbacks == [("inner",), ("outer",)]
    finally:
        runtime.artifacts.close()


def test_unsupported_adapter_fails_without_provider_call():
    adapter = MockAdapter(default_model="primary")
    runtime = compile_program(
        parse_program("> fallback: alternative\n>> The value is [value].\n"), adapter=adapter
    )
    try:
        with pytest.raises(Exception, match="model_fallback"):
            runtime.run_main()
        assert adapter.request_calls == []
    finally:
        runtime.artifacts.close()


def test_fallback_completion_hover_symbols_and_unsupported_diagnostics():
    source = "> fal"
    completion = compute_completion(
        source, "/tmp/fallback.kedi", lsp.Position(line=0, character=len(source))
    )
    assert any(item.label == "fallback" and "alternatives" in item.detail for item in completion)
    source = "> agent: codex\n> fallback: openai:backup\n"
    diagnostics = compute_diagnostics(source, "/tmp/fallback.kedi")
    assert any(d.code == "K-ADAPTER-MODEL-FALLBACK" for d in diagnostics)
    assert "request budget" in hover_text_at(source, "/tmp/fallback.kedi", "fallback")
    symbols = compute_document_symbols(
        """> profile: selected:
    > fallback: backup
@local():
    > fallback: another
    = done
""",
        "/tmp/fallback.kedi",
    )
    assert (
        sum(child.name == "> fallback:" for symbol in symbols for child in symbol.children or [])
        == 2
    )


def test_explain_records_fallback_without_running_dynamic_expressions():
    result = explain_program(parse_program("> fallback: `missing()`\n>> The value is [value].\n"))
    assert "fallback_models" in repr(result)
    assert "missing()" in repr(result)


def test_profile_fallback_has_hover_and_unsupported_adapter_diagnostics():
    source = """> profile: resilient:
    > agent: codex
    > fallback: openai:backup
> use: resilient
>> The value is [value].
"""
    diagnostics = compute_diagnostics(source, "/tmp/fallback.kedi")
    assert any(d.code == "K-ADAPTER-MODEL-FALLBACK" for d in diagnostics)
    assert "ordered" in hover_text_at(source, "/tmp/fallback.kedi", "fallback").lower()


def test_computed_profile_fallback_is_not_evaluated_by_the_editor():
    source = """> profile: resilient:
    > fallback: `missing()`
> use: resilient
>> The value is [value].
"""
    diagnostics = compute_diagnostics(source, "/tmp/fallback.kedi")
    assert not any(d.code == "K-ADAPTER-MODEL-FALLBACK" for d in diagnostics)
    assert "fallback" in hover_text_at(source, "/tmp/fallback.kedi", "fallback")


def test_spanless_profile_fallback_is_omitted_from_document_symbols():
    from kedi.lsp.features.capability_state import _profile_adapter_state
    from kedi.lsp.features.document import _profile_def_symbol

    profile = parse_program("> profile: resilient:\n    > fallback: backup\n").profile_defs[
        "resilient"
    ]
    profile.fallback_members[0].loc = None
    assert _profile_def_symbol(profile).children == []
    state = _profile_adapter_state(profile)
    assert state.fallback_enabled is True
    assert state.fallback_loc is None
