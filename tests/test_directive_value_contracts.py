"""DSL literals, computed values, and shared editor value contracts."""

from __future__ import annotations

import pytest
from lsprotocol import types as lsp

from kedi.agent_profile.directives import (
    eval_artifacts_directive,
    eval_codemode_directive,
    eval_history_directive,
    eval_hooks_directive,
    eval_settings_directive,
    eval_skills_directive,
)
from kedi.core import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.parser import KediParseError
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.diagnostics import compute_diagnostics


@pytest.mark.parametrize(
    ("directive", "evaluate", "attribute"),
    [
        ("history", eval_history_directive, "history"),
        ("skills", eval_skills_directive, "skills"),
        ("codemode", eval_codemode_directive, "codemode"),
        ("hooks", eval_hooks_directive, "hooks"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_computed_activation_short_and_block(directive, evaluate, attribute, enabled):
    forms = [f"> {directive}: `flag`\n"]
    if directive != "hooks":
        forms.append(f"> {directive}:\n  enabled: `flag`\n")
    runtime = KediRuntime()
    for source in forms:
        statement = parse_program(source).ordered_toplevel[0]
        profile = evaluate(runtime, statement, {"flag": enabled})
        actual = getattr(profile, attribute)
        assert (actual if directive == "history" else actual.enabled) is enabled
        with pytest.raises(ValueError, match="boolean"):
            evaluate(runtime, statement, {"flag": "false"})


def test_dynamic_activation_survives_profile_capture_and_restores_outer_scope():
    runtime = compile_program(
        parse_program("""
> history: enabled
> profile: quiet:
  > history: `False`
  > skills:
    enabled: `False`
  > codemode: `False`

@inspect() -> bool:
  > use: quiet
  = `True`

= `inspect()`
""")
    )
    try:
        profile = runtime._evaluate_profile_definition(runtime._profile_registry["quiet"])
        assert profile.history is False
        assert profile.skills.enabled is False
        assert profile.codemode.enabled is False
        assert runtime.run_main() is True
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("scope", ["history", "skills", "codemode", "artifacts"])
@pytest.mark.parametrize("value", ["True", "TRUE", "enabled", '"true"', "none", "null", "1"])
def test_boolean_literals_are_strict_and_diagnostics_target_value(scope, value):
    source = f"> {scope}:\n  enabled: {value}\n"
    with pytest.raises(KediParseError):
        parse_program(source)
    diagnostics = compute_diagnostics(source, None)
    assert diagnostics and diagnostics[0].severity == lsp.DiagnosticSeverity.Error
    assert diagnostics[0].range.start == lsp.Position(line=1, character=11)


@pytest.mark.parametrize(
    "source",
    [
        "> settings:\n  parallel_tool_calls: maybe\n",
        '> settings:\n  stop_sequences: ["END"]\n',
        '> mcp:\n  transport: stdio\n  command: uvx\n  args: ["server"]\n',
        '> artifacts:\n  store: "file"\n',
        "> history:\n  enabled: true\n  processor: callback\n",
        "> agent: a2a:\n  endpoint: https://example.com\n  auth:\n    scheme: mystery\n",
        "> adapter: pydantic\n> settings:\n  tool_choice: unknown\n",
    ],
)
def test_invalid_static_fields_fail_parser_and_editor(source):
    with pytest.raises(KediParseError):
        parse_program(source)
    assert any(
        item.severity == lsp.DiagnosticSeverity.Error for item in compute_diagnostics(source, None)
    )


def test_none_is_a_string_tool_option_and_containers_require_python():
    source = '> adapter: pydantic\n> settings:\n  tool_choice: none\n  stop_sequences: `["END"]`\n'
    statement = parse_program(source).ordered_toplevel[1]
    profile = eval_settings_directive(KediRuntime(), statement, {})
    assert profile.settings == {"tool_choice": "none", "stop_sequences": ["END"]}


@pytest.mark.parametrize("value", [True, False])
def test_query_activation_uses_bool_expression_without_changing_python_policy(value):
    statement = parse_program("> artifacts:\n  query_artifacts: `flag`\n").ordered_toplevel[0]
    profile = eval_artifacts_directive(KediRuntime(), statement, {"flag": value})
    assert profile.artifacts.queries_enabled() is value
    assert profile.artifacts.query_artifacts == ("enabled" if value else "disabled")


@pytest.mark.parametrize(
    ("source", "options"),
    [
        ("> effort: h", {"high"}),
        ("> history: e", {"enabled"}),
        ("> history:\n  enabled: t", {"true"}),
        ("> settings:\n  tool_reason: d", {"disabled"}),
        ("> settings:\n  parallel_tool_calls: f", {"false"}),
        ("> artifacts:\n  query_artifacts: e", {"enabled"}),
        ("> artifacts:\n  store: f", {"file"}),
        ("> mcp:\n  transport: st", {"stdio", "streamable-http"}),
        (
            "> agent: a2a:\n  endpoint: https://example.com\n  auth:\n    scheme: b",
            {"basic", "bearer"},
        ),
        ("> profile: worker:\n  > workflow: d", {"delegate", "dynamic"}),
        ("> capability: search:\n  > defer: f", {"false"}),
        ("> adapter: p", {"pydantic"}),
        ("> agent: a", {"acp", "a2a"}),
        ("> adapter: pydantic\n> settings:\n  tool_choice: n", {"none"}),
        ("> agent: codex\n> settings:\n  sandbox: workspace-w", {"workspace-write"}),
        ("> adapter: pydantic\n> settings:\n  thinking: t", {"true"}),
        ("> adapter: langchain\n> settings:\n  disable_streaming: t", {"true", "tool_calling"}),
        ("> requires: deferred_t", {"deferred_tools"}),
        ("> requires:\n  deferred_t", {"deferred_tools"}),
    ],
)
def test_shared_lsp_completes_only_matching_values(source, options):
    position = lsp.Position(line=source.count("\n"), character=len(source.split("\n")[-1]))
    items = compute_completion(source, None, position)
    assert {item.label for item in items} == options
    assert all(item.detail and isinstance(item.text_edit, lsp.TextEdit) for item in items)


def test_empty_value_requires_explicit_completion_and_preserves_enter():
    source = "> history:\n  enabled: "
    position = lsp.Position(line=1, character=11)
    assert compute_completion(source, None, position) == []
    items = compute_completion(source, None, position, explicit=True)
    assert {item.label for item in items} == {"true", "false"}


def test_completion_replaces_remaining_value_without_touching_syntax():
    source = "> artifacts:\n  store: memory"
    item = compute_completion(source, None, lsp.Position(line=1, character=10))[0]
    edit = item.text_edit
    assert isinstance(edit, lsp.TextEdit)
    line = source.split("\n")[-1]
    assert (
        line[: edit.range.start.character] + edit.new_text + line[edit.range.end.character :]
        == "  store: memory"
    )


def test_adapter_completion_ignores_python_text_and_sibling_profiles():
    source = """
> adapter: pydantic
> profile: remote:
  > agent: codex
```
text = "> agent: codex"
```
> settings:
  tool_choice: n"""
    position = lsp.Position(line=source.count("\n"), character=16)
    assert {item.label for item in compute_completion(source, None, position)} == {"none"}


def test_lsp_validates_settings_selected_through_profile_without_running_code():
    source = """
> profile: local:
  > adapter: pydantic
> use: local
> settings:
  tool_choice: unknown
"""
    diagnostics = compute_diagnostics(source, None)
    assert any(item.code == "K-DIRECTIVE-VALUE" for item in diagnostics)


def test_dynamic_values_are_not_executed_by_editor():
    source = "> artifacts:\n  query_artifacts: `raise_if_called()`\n"
    assert not [
        item
        for item in compute_diagnostics(source, None)
        if item.severity == lsp.DiagnosticSeverity.Error
    ]


def test_model_receives_none_as_tool_choice_string():
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter import PydanticAdapter

    seen = []

    def respond(messages, info):
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart("ready")])

    runtime = compile_program(
        parse_program(
            "> settings:\n  tool_choice: none\n[result] << Confirm readiness.\n= <result>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    try:
        assert runtime.run_main() == "ready"
        assert seen[0]["tool_choice"] == "none"
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("field", ["parallel_tool_calls", "tool_choice", "service_tier"])
def test_python_none_cannot_reset_a_dsl_field(field):
    runtime = compile_program(parse_program(f"> settings:\n  {field}: `None`\n"))
    try:
        with pytest.raises(ValueError):
            runtime.run_main()
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("client", ["vscode", "zed"])
def test_editor_protocol_value_completions_and_diagnostic_updates(client, monkeypatch):
    from types import SimpleNamespace
    from typing import Any, cast

    from kedi.lsp.server import _publish_diagnostics, make_server

    server = make_server()
    document = SimpleNamespace(source="> settings:\n  tool_reason: en")
    uri = f"file:///tmp/{client}-options.kedi"
    cast(Any, server.protocol)._workspace = SimpleNamespace(get_text_document=lambda _: document)
    handler = server.protocol.fm.features[lsp.TEXT_DOCUMENT_COMPLETION]
    items = handler(
        lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=lsp.Position(line=1, character=17),
            context=lsp.CompletionContext(trigger_kind=lsp.CompletionTriggerKind.Invoked),
        )
    )
    assert {item.label for item in items} == {"enabled"}
    notifications = []
    monkeypatch.setattr(server, "text_document_publish_diagnostics", notifications.append)
    document.source = "> settings:\n  tool_reason: maybe\n"
    _publish_diagnostics(server, uri)
    assert notifications[-1].diagnostics[0].severity == lsp.DiagnosticSeverity.Error
    document.source = "> settings:\n  tool_reason: enabled\n"
    _publish_diagnostics(server, uri)
    assert notifications[-1].diagnostics == []


@pytest.mark.parametrize(
    ("scope", "name", "value", "adapter"),
    [
        ("history", "processor", 42, None),
        ("settings", "callbacks", None, None),
        ("mcp", "args", "server", None),
        ("mcp", "args", 42, None),
        ("settings", "max_tokens", True, None),
        ("history", "compaction_threshold", 0, None),
        ("settings", "temperature", "warm", None),
        ("settings", "api_key", 42, None),
        ("artifacts", "path", 42, None),
        ("settings", "tool_choice", None, "pydantic"),
        ("settings", "tool_choice", True, "pydantic"),
        ("settings", "tool_choice", "unknown", "pydantic"),
        ("settings", "thinking", "unknown", "pydantic"),
    ],
)
def test_computed_values_obey_their_declared_contract(scope, name, value, adapter):
    from kedi.directive_values import validate_field

    with pytest.raises(ValueError):
        validate_field(scope, name, value, adapter=adapter)


def test_native_objects_optional_python_values_and_sandbox_aliases():
    from pathlib import Path

    from kedi.agent_profile.settings import pydantic_model_settings
    from kedi.directive_values import parse_field, parse_scalar, validate_field

    def callback(ctx):
        return ctx.messages

    assert validate_field("history", "processor", callback) is callback
    assert validate_field("settings", "callbacks", callback) is callback
    assert validate_field("artifacts", "path", Path("artifacts")) == Path("artifacts")
    assert validate_field("settings", "tool_choice", ["lookup"], adapter="pydantic") == ["lookup"]
    assert validate_field("settings", "thinking", False, adapter="pydantic") is False
    assert validate_field("settings", "thinking", "high", adapter="pydantic") == "high"
    assert parse_field("settings", "sandbox", "readOnly", adapter="codex") == "readOnly"
    assert parse_field("settings", "api_key", "123") == "123"
    assert parse_field("settings", "max_tokens", "512", adapter="pydantic") == 512
    assert parse_field("settings", "max_tokens", "0", adapter="pydantic") == 0
    assert pydantic_model_settings({"temperature": None}) == {"temperature": None}
    assert parse_scalar("false") is False
    with pytest.raises(ValueError, match="inline Python"):
        parse_field("settings", "callbacks", "callback")


@pytest.mark.parametrize(
    "source",
    [
        "> requires:\n\n  # Available at runtime\n  stream_e",
        "> requires:\n  structured_output\n  stream_e",
    ],
)
def test_requirement_completion_skips_comments_and_existing_entries(source):
    position = lsp.Position(line=source.count("\n"), character=len(source.split("\n")[-1]))
    assert {item.label for item in compute_completion(source, None, position)} == {"stream_events"}


def test_value_completion_handles_stale_positions_comments_and_unknown_fields():
    from kedi.lsp.features.directive_options import option_completions, value_context

    assert value_context("", lsp.Position(line=1, character=0)) is None
    assert value_context("> history:\n  enabled", lsp.Position(line=1, character=9)) is None
    assert option_completions("> settings:\n  unknown: x", lsp.Position(line=1, character=12)) == []
    source = "###\n> effort: h\n###"
    assert option_completions(source, lsp.Position(line=1, character=11)) == []
    source = "> agent: a2a:\n  credentials:\n    scheme: b"
    assert value_context(source, lsp.Position(line=2, character=13)) is None


def test_dynamic_adapter_selection_does_not_inherit_an_old_option_domain():
    source = "> adapter: pydantic\n> adapter: `select_adapter()`\n> settings:\n  tool_choice: n"
    position = lsp.Position(line=3, character=16)
    assert compute_completion(source, None, position) == []


@pytest.mark.parametrize("directive", ["history", "skills", "codemode", "hooks"])
def test_computed_activation_hover_and_diagnostics_do_not_assume_enabled(directive):
    from tests.lsp_helpers import hover_text_at

    source = f"> {directive}: `is_enabled`\n"
    assert "`is_enabled`" in hover_text_at(source, "options.kedi", directive, offset=2)
    assert not [
        item
        for item in compute_diagnostics(source, None)
        if item.severity == lsp.DiagnosticSeverity.Error
    ]
    source = f"> profile: local:\n  > {directive}: `is_enabled`\n> use: local\n"
    assert not [
        item
        for item in compute_diagnostics(source, None)
        if item.severity == lsp.DiagnosticSeverity.Error
    ]


def test_public_option_predicates_remain_available():
    from kedi.effort import is_valid_effort
    from kedi.mcp import is_valid_mcp_transport

    assert is_valid_effort(" high ")
    assert not is_valid_effort("unknown")
    assert is_valid_mcp_transport("streamable-http")
    assert not is_valid_mcp_transport("websocket")


def test_profile_setting_diagnostics_do_not_evaluate_computed_values():
    source = '> profile: local:\n  > adapter: pydantic\n  > settings:\n    tool_choice: `"none"`\n'
    assert not [
        item
        for item in compute_diagnostics(source, None)
        if item.severity == lsp.DiagnosticSeverity.Error
    ]


def test_malformed_configuration_text_does_not_offer_unrelated_symbols():
    source = "> history:\n  invalid?"
    assert compute_completion(source, None, lsp.Position(line=1, character=10)) == []


@pytest.mark.parametrize("members", ["> adapter: pydantic", "> use: framework"])
def test_option_completion_follows_composed_profiles(members):
    source = (
        "> profile: framework:\n  > adapter: pydantic\n"
        "> profile: selected:\n  " + members + "\n"
        "> use: selected\n> settings:\n  tool_choice: n"
    )
    position = lsp.Position(line=source.count("\n"), character=16)
    assert {item.label for item in compute_completion(source, None, position)} == {"none"}


@pytest.mark.parametrize(
    "prefix",
    [
        "> adapter: pydantic\n> capability: guidance:\n  > instructions: Be concise.\n> use: guidance\n",
        "> adapter: pydantic\n@lookup():\n  = ready\n> use: lookup\n",
        "> adapter: pydantic\n> profile: guidance:\n  > instructions: Be concise.\n> use: guidance\n",
        "> adapter: pydantic\n> capability: guidance:\n  > instructions: Be concise.\n> profile: local:\n  > adapter: pydantic\n  > use: guidance\n> use: local\n",
    ],
)
def test_tools_and_capabilities_preserve_the_adapter_option_domain(prefix):
    source = prefix + "> settings:\n  tool_choice: n"
    position = lsp.Position(line=source.count("\n"), character=16)
    assert {item.label for item in compute_completion(source, None, position)} == {"none"}


@pytest.mark.parametrize(
    "prefix",
    [
        "> adapter: pydantic\n> use: imported_profile\n",
        "> adapter: pydantic\n> profile: dynamic:\n  > adapter: `select_adapter()`\n> use: dynamic\n",
        "> profile: first:\n  > use: second\n> profile: second:\n  > use: first\n> use: first\n",
    ],
)
def test_unknown_and_cyclic_profile_selections_do_not_guess_an_sdk_domain(prefix):
    source = prefix + "> settings:\n  tool_choice: n"
    position = lsp.Position(line=source.count("\n"), character=16)
    assert compute_completion(source, None, position) == []


def test_last_profile_in_a_use_block_selects_the_option_domain():
    source = (
        "> profile: framework:\n  > adapter: pydantic\n"
        "> profile: harness:\n  > agent: codex\n"
        "> use:\n  framework\n  harness\n> settings:\n  sandbox: w"
    )
    position = lsp.Position(line=source.count("\n"), character=12)
    assert {item.label for item in compute_completion(source, None, position)} == {
        "workspace-write"
    }


@pytest.mark.parametrize(
    "selection",
    [
        '> adapter: `"langchain"`\n',
        '> profile: local:\n  > adapter: `"langchain"`\n> use: local\n',
        "> profile: framework:\n  > adapter: langchain\n"
        "> profile: local:\n  > use: framework\n> use: local\n",
    ],
)
def test_setting_diagnostics_do_not_reuse_a_previous_adapter_domain(selection):
    source = "> adapter: pydantic\n" + selection + "> settings:\n  tool_choice: lookup\n"
    assert not [
        diagnostic
        for diagnostic in compute_diagnostics(source, None)
        if diagnostic.severity == lsp.DiagnosticSeverity.Error
    ]


@pytest.mark.parametrize("selection", ["", "  > adapter: langchain\n"])
def test_profile_settings_do_not_inherit_the_declaration_sites_adapter(selection):
    source = (
        "> adapter: pydantic\n> profile: local:\n"
        "  > settings:\n    tool_choice: lookup\n"
        + selection
        + "> adapter: langchain\n> use: local\n= ready\n"
    )
    runtime = compile_program(parse_program(source))
    try:
        assert runtime.run_main() == "ready"
    finally:
        runtime.artifacts.close()
    assert not [
        diagnostic
        for diagnostic in compute_diagnostics(source, None)
        if diagnostic.severity == lsp.DiagnosticSeverity.Error
    ]


def test_profile_settings_use_the_profiles_final_explicit_adapter():
    source = "> profile: local:\n  > settings:\n    tool_choice: invalid\n  > adapter: pydantic\n"
    with pytest.raises(KediParseError, match="tool_choice"):
        parse_program(source)


def test_procedure_use_takes_precedence_over_a_profile_with_the_same_name():
    source = (
        "> adapter: langchain\n@lookup():\n  = ready\n"
        "> profile: lookup:\n  > adapter: pydantic\n"
        "> use: lookup\n> settings:\n  tool_choice: lookup\n"
    )
    assert not [
        diagnostic
        for diagnostic in compute_diagnostics(source, None)
        if diagnostic.severity == lsp.DiagnosticSeverity.Error
    ]


@pytest.mark.parametrize("directive", ["hooks", "codemode", "skills", "history"])
@pytest.mark.parametrize("nested", [False, True])
def test_computed_activation_does_not_report_an_unsupported_static_feature(directive, nested):
    activation = f"> {directive}: `False`\n"
    source = "> adapter: dspy\n" + (
        "> profile: local:\n  " + activation + "> use: local\n" if nested else activation
    )
    assert not [
        diagnostic
        for diagnostic in compute_diagnostics(source, None)
        if diagnostic.severity == lsp.DiagnosticSeverity.Error
    ]


@pytest.mark.parametrize("directive", ["history", "codemode", "hooks"])
def test_profile_computed_activation_replaces_a_previously_known_activation(directive):
    from kedi.lsp.features.capability_state import (
        _AdapterDiagnosticState,
        _merge_adapter_state,
        _profile_adapter_state,
    )

    profile = parse_program(f"> profile: local:\n  > {directive}: `False`\n").profile_defs["local"]
    inner = _profile_adapter_state(profile)
    outer = _AdapterDiagnosticState(**{f"{directive}_enabled": True})
    merged = _merge_adapter_state(outer, inner)
    assert getattr(merged, f"{directive}_enabled") is None
    assert getattr(merged, f"{directive}_loc") == getattr(inner, f"{directive}_loc")


@pytest.mark.parametrize("directive", ["history", "codemode", "skills", "hooks"])
def test_profile_activation_hover_shows_the_expression(directive):
    from tests.lsp_helpers import hover_text_at

    source = f"> profile: local:\n  > {directive}: `is_enabled`\n"
    assert "`is_enabled`" in hover_text_at(source, "options.kedi", directive, offset=2)


def test_native_sdk_settings_preserve_valid_objects():
    from pydantic import SecretStr

    from kedi.agent_profile.settings import langchain_model_settings, pydantic_model_settings

    secret = SecretStr("example-key")
    assert langchain_model_settings({"api_key": secret})["api_key"] is secret
    assert pydantic_model_settings({"temperature": None}) == {"temperature": None}


def test_requirement_completion_does_not_activate_inside_a_block_comment():
    source = "###\n> requires:\n  stream_e\n###\n"
    assert compute_completion(source, None, lsp.Position(line=2, character=10)) == []


@pytest.mark.parametrize("value", ["0", "-1"])
def test_codemode_timeout_rejects_nonpositive_literals_in_the_editor(value):
    source = f"> codemode:\n  request_timeout: {value}\n"
    with pytest.raises(KediParseError, match="request_timeout"):
        parse_program(source)
    assert any(
        diagnostic.code == "K-DIRECTIVE-VALUE" for diagnostic in compute_diagnostics(source, None)
    )


def test_positive_codemode_timeout_can_be_computed():
    source = "> codemode:\n  request_timeout: `0.25`\n"
    statement = parse_program(source).ordered_toplevel[0]
    profile = eval_codemode_directive(KediRuntime(), statement, {})
    assert profile.codemode.request_timeout == 0.25


@pytest.mark.parametrize(
    ("field", "value"), [("tool_choice", '"invalid"'), ("thinking", '"invalid"')]
)
def test_sdk_option_expressions_are_validated_before_model_io(field, value):
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter import PydanticAdapter

    def respond(messages, info):
        pytest.fail("Invalid SDK options must fail before model I/O")

    runtime = compile_program(
        parse_program(
            f"> settings:\n  {field}: `{value}`\n[result] << Confirm readiness.\n= <result>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    try:
        with pytest.raises(ValueError, match=field):
            runtime.run_main()
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize(
    "prelude",
    [
        'def worker():\n  return "ready"',
        'async def worker():\n  return "ready"',
        "class worker:\n  pass",
        'worker = lambda: "ready"',
        "from helpers import worker",
        "import helpers as worker",
        "from helpers import *",
        'exec("worker = lambda: 1")',
        "worker =",
    ],
)
def test_python_bindings_do_not_get_mistaken_for_same_named_profiles(prelude):
    source = (
        f"```\n{prelude}\n```\n> adapter: langchain\n"
        "> profile: worker:\n  > adapter: pydantic\n"
        "> use: worker\n> settings:\n  tool_choice: lookup\n"
    )
    # Parsing and editor inspection must never import or execute the prelude.
    parse_program(source)
    assert not [
        diagnostic
        for diagnostic in compute_diagnostics(source, None)
        if diagnostic.code == "K-DIRECTIVE-VALUE"
    ]


def test_unrelated_python_bindings_preserve_provable_profile_selections():
    source = (
        "```\nfrom pathlib import Path\nvalue = 1\n```\n"
        "> profile: worker:\n  > adapter: pydantic\n"
        "> use: worker\n> settings:\n  tool_choice: n"
    )
    position = lsp.Position(line=source.count("\n"), character=16)
    assert {item.label for item in compute_completion(source, None, position)} == {"none"}
