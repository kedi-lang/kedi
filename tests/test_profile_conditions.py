from __future__ import annotations

from copy import deepcopy
from textwrap import indent

import pytest
from lsprotocol import types as lsp

from kedi.agent_profile.capabilities import deferred_capabilities
from kedi.agent_profile.compile import profile_stmt_to_definition
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.explain import explain_program
from kedi.lang import compile_program, parse_program
from kedi.lang.parser import KediParseError
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.diagnostics import compute_diagnostics
from tests.mock_adapter import MockAdapter


def evaluate(source: str, **env):
    program = parse_program(source)
    definitions = {
        name: profile_stmt_to_definition(stmt) for name, stmt in program.profile_defs.items()
    }
    runtime = KediRuntime()
    try:
        return eval_profile_definition(runtime, definitions["selected"], env, registry=definitions)
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize(
    ("flag", "model", "effort"), [(True, "fast-model", "high"), (False, "smart-model", None)]
)
def test_selected_branch_contributes_grouped_settings(flag, model, effort):
    result = evaluate(
        """> profile: selected:
  > if: `flag`:
    > model: fast-model
    > effort: high
  > else:
    > model: smart-model
""",
        flag=flag,
    )
    assert result.model == model
    assert result.effort == effort


def test_conditions_are_lazy_nested_and_evaluated_once_in_source_order():
    calls = []

    def choose():
        calls.append("choose")
        return False

    def forbidden():
        raise AssertionError("unselected member executed")

    result = evaluate(
        """> profile: selected:
  > model: initial
  > if: `choose()`:
    > if: `forbidden()`:
      > model: `forbidden()`
  > else:
    > model: fallback
    > if: `True`:
      > effort: medium
    > else:
      > effort: `forbidden()`
  > model: final
""",
        choose=choose,
        forbidden=forbidden,
    )
    assert calls == ["choose"]
    assert result.model == "final"
    assert result.effort == "medium"


def test_false_without_else_preserves_prior_members():
    result = evaluate("""> profile: selected:
  > model: unchanged
  > if: `False`:
    > model: missing
""")
    assert result.model == "unchanged"


@pytest.mark.parametrize("value", [0, 1, "", "true", [], None])
def test_guard_requires_exact_bool(value):
    with pytest.raises(TypeError, match="condition must evaluate to a bool"):
        evaluate("> profile: selected:\n  > if: `value`:\n    > model: one\n", value=value)


@pytest.mark.parametrize(
    "body",
    [
        "> if: A plausible claim\n  > model: one\n",
        "> if: `True`:\n  >> The result is [result].\n",
        "> if: `True`:\n  = result\n",
        "> else:\n  > model: one\n",
        "> if: `True`:\n  > model: one\n> else:\n  > model: two\n> else:\n  > model: three\n",
    ],
)
def test_configuration_guards_do_not_admit_executable_bodies_or_orphan_else(body):
    with pytest.raises(KediParseError):
        parse_program("> profile: selected:\n" + indent(body, "  "))


@pytest.mark.parametrize(
    "member", ["model: cheap", "settings:\n  temperature: 0.2", "history: false", "output: str"]
)
def test_capability_ownership_rules_apply_even_to_unselected_branches(member):
    with pytest.raises(KediParseError, match="not allowed in a capability"):
        parse_program(
            "> capability: selected:\n  > if: `False`:\n" + indent("> " + member + "\n", "    ")
        )


def test_capability_conditional_instructions_and_defer():
    source = """> capability: selected:
  > if: `deferred`:
    > defer: true
    > instructions: Check provenance.
  > else:
    > instructions: Keep it brief.
"""
    delayed = evaluate(source, deferred=True)
    assert deferred_capabilities(delayed)[0].profile.system == "Check provenance."
    eager = evaluate(source, deferred=False)
    assert not deferred_capabilities(eager)
    assert eager.capability_instructions[0].content == "Keep it brief."


def test_conditional_use_keeps_profile_and_capability_composition():
    result = evaluate("""> capability: evidence:
  > instructions: Cite sources.
> profile: extra:
  > effort: low
> profile: selected:
  > instructions: Review the report.
  > if: `True`:
    > use:
      extra
      evidence
  > else:
    > use: missing
""")
    assert result.system == "Review the report."
    assert result.effort == "low"
    assert result.capability_instructions[0].content == "Cite sources."


@pytest.mark.parametrize("kind, value", [("output", "str"), ("workflow", "delegate")])
def test_singleton_members_checked_on_selected_path(kind, value):
    source = f"""> profile: selected:
  > {kind}: {value}
  > if: `flag`:
    > {kind}: {value}
"""
    evaluate(source, flag=False)
    with pytest.raises(KediExecutionError, match="only one"):
        evaluate(source, flag=True)


def test_exclusive_output_types_are_legal():
    result = evaluate("""> profile: selected:
  > if: `False`:
    > output: str
  > else:
    > output: int
""")
    assert result.output_type is int


def test_workflow_validation_occurs_after_selected_members_merge():
    source = """> profile: worker:
  > instructions: Check the document.
> profile: selected:
  > workflow: dynamic
  > if: `flag`:
    > subagent: worker
"""
    assert evaluate(source, flag=True).subagents == ("worker",)
    with pytest.raises(KediExecutionError, match="requires at least one"):
        evaluate(source, flag=False)


def test_adapter_selection_is_exclusive_per_active_path():
    source = """> profile: selected:
  > if: `flag`:
    > adapter: pydantic
  > else:
    > agent: codex
"""
    assert evaluate(source, flag=True).adapter_shortname == "pydantic"
    assert evaluate(source, flag=False).adapter_shortname == "codex"
    with pytest.raises(KediExecutionError, match="cannot contain both"):
        evaluate(
            "> profile: selected:\n  > adapter: pydantic\n  > if: `True`:\n    > agent: codex\n"
        )


def test_compiled_program_captures_profile_at_procedure_definition():
    adapter = MockAdapter(default_model="default")
    runtime = compile_program(
        parse_program("""[quality: bool] = `True`
> profile: selected:
  > if: `quality`:
    > model: quality-model
  > else:
    > model: fast-model
> use: selected
@answer() -> str:
  >> The city is [city].
  = <city>
`quality = False`
= <answer()>
"""),
        adapter=adapter,
    )
    try:
        runtime.run_main()
        assert adapter.model_calls == ["quality-model"]
    finally:
        runtime.artifacts.close()


def test_explain_never_runs_profile_conditions():
    program = parse_program("""> model: initial
> profile: selected:
  > if: `should_never_execute()`:
    > model: alternate
> use: selected
>> The city is [city].
""")
    report = explain_program(program)
    assert report.calls[0].model.value is None
    assert "conditional" in report.calls[0].model.unresolved


def test_condition_definition_deepcopy_is_independent():
    definition = profile_stmt_to_definition(
        parse_program("> profile: selected:\n  > if: `True`:\n    > model: first\n").profile_defs[
            "selected"
        ]
    )
    copied = deepcopy(definition)
    assert copied == definition
    assert copied.members[0].consequence is not definition.members[0].consequence


def test_nested_capability_completions_remain_ownership_aware():
    source = "> capability: selected:\n  > if: `True`:\n    > "
    items = compute_completion(source, "/tmp/conditional.kedi", lsp.Position(line=2, character=6))
    labels = {item.label for item in items}
    assert {"instructions", "use", "if", "else", "defer"} <= labels
    assert not {"model", "adapter", "profile", "show", "task"} & labels


def test_conditional_sdk_options_validate_against_branch_backend():
    source = """> profile: selected:
  > if: `choose()`:
    > adapter: pydantic
    > settings:
      parallel_tool_calls: invalid
  > else:
    > adapter: langchain
    > settings:
      parallel_tool_calls: true
"""
    errors = compute_diagnostics(source, "/tmp/conditional.kedi")
    assert any("true" in error.message and error.range.start.line == 4 for error in errors)


def test_capability_forbidden_hooks_inside_condition_are_rejected():
    with pytest.raises(KediParseError, match="owned tool hooks"):
        parse_program("""> capability: selected:
  > if: `False`:
    > hooks: disabled
""")


def test_pydantic_model_flow_uses_only_selected_capability_instructions():
    from pydantic_ai import models
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter

    prompts = []

    def respond(messages, info):
        prompts.extend(str(message) for message in messages)
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {"city": "Istanbul"})])

    with models.override_allow_model_requests(False):
        runtime = compile_program(
            parse_program("""> capability: style:
  > if: `True`:
    > instructions: Use exact city names.
  > else:
    > instructions: Never use city names.
> use: style
>> The city of cats is [city].
= <city>
"""),
            adapter=PydanticAdapter(model=FunctionModel(respond)),
        )
        try:
            assert runtime.run_main() == "Istanbul"
        finally:
            runtime.artifacts.close()
    assert "Use exact city names." in "\n".join(prompts)
    assert "Never use city names." not in "\n".join(prompts)


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("flag", [True, False])
def test_selected_capability_registers_only_its_owned_tool(backend, flag):
    from tests.test_capability_tool_binding import scripted_adapter

    requests = []
    executed = []

    def access() -> str:
        executed.append("access")
        return "source evidence"

    def forbidden() -> str:
        raise AssertionError("unselected tool ran")

    runtime = compile_program(
        parse_program(
            """> approval: deny
> capability: selected:
  > if: `flag`:
    > approval: allow
    > use: access
  > else:
    > approval: allow
    > use: forbidden
> use: selected
"""
            + ("[reply] << Read the document.\n= <reply>\n" if flag else "= done\n")
        ),
        adapter=scripted_adapter(backend, False, requests),
        runtime_globals={"flag": flag, "access": access, "forbidden": forbidden},
    )
    try:
        runtime.run_main()
        assert executed == (["access"] if flag else [])
        if flag:
            assert "forbidden" not in str(requests)
    finally:
        runtime.artifacts.close()


def test_imported_conditional_bundle_retains_private_profile_and_tool(tmp_path):
    from tests.test_capability_tool_binding import scripted_adapter

    library = tmp_path / "library.kedi"
    library.write_text("""@access() -> str:
  = private evidence
> capability: private_bundle:
  > approval: allow
  > use: access
> capability: exported:
  > if: `True`:
    > use: private_bundle
> export:
  exported
""")
    requests = []
    runtime = compile_program(
        parse_program(
            """> import: library
> use: exported
[reply] << Read the document.
= <reply>
""",
            source_path=str(tmp_path / "main.kedi"),
        ),
        adapter=scripted_adapter("pydantic", False, requests),
    )
    try:
        assert runtime.run_main() == "done"
        assert "private evidence" in str(requests[-1])
        assert "access" not in runtime._procedures
    finally:
        runtime.artifacts.close()


def test_conditional_capability_capture_does_not_run_during_compilation():
    from tests.test_capability_capture import CapabilityAdapter

    evaluations = []

    def choose():
        evaluations.append("guard")
        return True

    runtime = compile_program(
        parse_program("""> capability: style:
  > instructions: Read carefully.
> profile: selected:
  > if: `choose()`:
    > use: style
> use: selected
@answer() -> str:
  >> The city is [city].
  = <city>
= <answer()>
"""),
        adapter=CapabilityAdapter(),
        runtime_globals={"choose": choose},
    )
    try:
        assert evaluations == []
        runtime.run_main()
        assert evaluations == ["guard"]
    finally:
        runtime.artifacts.close()


def test_lsp_conditional_symbols_hover_navigation_and_python_projection():
    from kedi.lsp.features.document import compute_document_symbols
    from kedi.lsp.python_virtual.document import compute_python_virtual_document
    from tests.lsp_helpers import definition_at, hover_text_at, references_at

    source = """```
prefer_quality = True
```
~Review(recommendation: str)
> profile: worker:
  > instructions: Check the report.
> capability: citations:
  > instructions: Cite sources.
> profile: selected:
  > if: `prefer_quality`:
    > model: quality-model
    > output: Review
    > subagent: worker
    > use: citations
  > else:
    > model: fast-model
> use: selected
"""
    path = "/tmp/conditional.kedi"
    assert "quality-model" in hover_text_at(source, path, "model", occurrence=1)
    assert "Conditional members" in hover_text_at(source, path, "selected")
    for needle in ("Review", "worker", "citations"):
        definition = definition_at(source, "file://" + path, path, needle, occurrence=2)
        assert definition is not None
        refs = references_at(source, "file://" + path, path, needle, True)
        assert len(refs) == 2
    selected = next(
        symbol
        for symbol in compute_document_symbols(source, path)
        if symbol.name == "> profile: selected"
    )
    branch = selected.children[0]
    assert branch.name == "> if: `prefer_quality`"
    assert any(child.name == "> else:" for child in branch.children)
    virtual = compute_python_virtual_document(source, path)
    assert "prefer_quality" in str(virtual)


def test_lsp_conditional_unknown_subagent_and_cycle():
    unknown = compute_diagnostics(
        "> profile: selected:\n  > if: `False`:\n    > subagent: missing\n", None
    )
    assert any(error.code == "K-SUBAGENT-UNKNOWN" for error in unknown)
    cyclic = compute_diagnostics(
        "> profile: selected:\n  > if: `False`:\n    > subagent: selected\n", None
    )
    assert any(error.code == "K-SUBAGENT-CYCLE" for error in cyclic)


@pytest.mark.parametrize(
    "file", ["tree-sitter-kedi/queries/highlights.scm", "kedi-zed/languages/kedi/highlights.scm"]
)
def test_profile_condition_keywords_are_highlighted(file):
    from pathlib import Path

    import tree_sitter_kedi
    from tree_sitter import Language, Parser, Query, QueryCursor

    source = (
        b"> profile: selected:\n  > if: `True`:\n    > model: one\n  > else:\n    > model: two\n"
    )
    language = Language(tree_sitter_kedi.language())
    captures = QueryCursor(Query(language, Path(file).read_text())).captures(
        Parser(language).parse(source).root_node
    )
    assert {"if", "else"} <= {
        source[node.start_byte : node.end_byte].decode() for node in captures["keyword"]
    }


def test_unknown_guard_does_not_select_a_backend_in_caller_diagnostics():
    source = """> agent: codex
> profile: selected:
  > if: `choose()`:
    > adapter: pydantic
  > else:
    > adapter: langchain
> use: selected
> codemode: enabled
"""
    assert not compute_diagnostics(source, None)


def test_deeply_nested_members_keep_one_owner_and_evaluate_each_guard_once():
    calls = []

    def choose():
        calls.append(True)
        return True

    body = "> model: nested-model\n"
    for _ in range(20):
        body = "> if: `choose()`:\n" + indent(body, "  ")
    result = evaluate("> profile: selected:\n" + indent(body, "  "), choose=choose)
    assert result.model == "nested-model"
    assert len(calls) == 20


def test_shared_definition_has_no_cross_thread_branch_state():
    from concurrent.futures import ThreadPoolExecutor

    definition = profile_stmt_to_definition(
        parse_program("""> profile: selected:
  > if: `flag`:
    > model: first
  > else:
    > model: second
""").profile_defs["selected"]
    )
    runtime = KediRuntime()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            profiles = list(
                pool.map(
                    lambda flag: eval_profile_definition(runtime, definition, {"flag": flag}),
                    [True, False] * 20,
                )
            )
        assert [profile.model for profile in profiles] == ["first", "second"] * 20
    finally:
        runtime.artifacts.close()


def test_guard_python_projection_declares_kedi_inputs_before_using_them():
    from kedi.lsp.python_virtual.document import compute_python_virtual_document

    virtual = compute_python_virtual_document("""[prefer_quality: bool] = `True`
> profile: selected:
  > if: `prefer_quality`:
    > model: quality-model
""")
    assert virtual["text"].index("prefer_quality: bool") < virtual["text"].index(
        "    prefer_quality\n"
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("> if: `flag`:\n  > instructions: Cite the evidence.\n", "pydantic"),
        ("> if: `flag`:\n  > adapter: pydantic\n", None),
        ("> if: `flag`:\n  > adapter: pydantic\n> else:\n  > adapter: pydantic\n", "pydantic"),
    ],
)
def test_conditional_adapter_resolution_outside_profile_never_runs_guards(body, expected):
    import tree_sitter_kedi
    from tree_sitter import Language, Parser

    from kedi.lang.parser.directive_context import selected_adapter

    source = (
        "> adapter: pydantic\n> profile: selected:\n"
        + indent(body, "  ")
        + "> use: selected\n> settings:\n  parallel_tool_calls: true\n"
    ).encode()
    tree = Parser(Language(tree_sitter_kedi.language())).parse(source)
    field = tree.root_node.named_children[-1].child_by_field_name("body").named_children[0]
    adapter = selected_adapter(source, field)
    assert adapter == expected


def test_unconditional_backend_after_a_guard_is_statically_known():
    from kedi.lsp.features.capability_state import _profile_adapter_state

    source = """> profile: selected:
  > if: `flag`:
    > agent: codex
  > agent: claude
  > codemode: enabled
> use: selected
"""
    state = _profile_adapter_state(parse_program(source).profile_defs["selected"])
    assert state.shortname == "claude"
    assert state.unknown_selection is False


def test_synthetic_conditional_outline_can_omit_source_locations():
    from kedi.lang.ast import ProfileConditionalEntry, ProfileDefStmt, ProfileModelEntry
    from kedi.lsp.features.document import _profile_def_symbol

    profile = ProfileDefStmt(
        name="selected",
        conditional_members=[
            ProfileConditionalEntry(
                backtick_code="True",
                consequence=ProfileDefStmt(
                    name="selected",
                    model_members=[ProfileModelEntry(plain_value="one")],
                ),
            ),
        ],
    )
    assert _profile_def_symbol(profile).children[0].name == "> if: `True`"


@pytest.mark.parametrize(
    "member",
    [
        "model: final",
        "adapter: pydantic",
        "agent: codex",
        "effort: high",
        "instructions: Finish the review.",
        "settings:\n  temperature: 0.2",
        "output: str",
        "use: another",
        "if: `False`:\n  > model: nested",
    ],
)
def test_profile_guard_without_else_does_not_consume_the_next_member(member):
    source = "> profile: selected:\n  > if: `True`:\n    > model: first\n" + indent(
        "> " + member + "\n", "  "
    )
    profile = parse_program(source).profile_defs["selected"]
    first = profile.conditional_members[0]
    assert first.alternative is None
    assert len(first.consequence.model_members) == 1
    assert first.consequence.model_members[0].plain_value == "first"


def test_capability_guard_without_else_preserves_following_owned_policy():
    result = evaluate("""> capability: selected:
  > if: `False`:
    > instructions: Skip this contribution.
  > approval: allow
  > instructions: Keep this contribution.
""")
    assert result.capability_instructions[0].content == "Keep this contribution."
