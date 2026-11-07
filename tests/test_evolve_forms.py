from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.cli import main as cli_main
from kedi.forms import Form, FormActivation, FormError, load_forms
from kedi.groups import form_fingerprint
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import TemplateBlock
from kedi.telemetry import TelemetryAsset
from kedi.tests_runner import run_tests
from tests.telemetry_helpers import RecordingTelemetryBackend, installed_recording_backend

PLANNER = """@solve(problem: str) -> str:
  [hint] = be brief
  > evolve: planner:
    >> Break <problem> into steps: [steps: list[str]]
    [plan] << Turn <steps> into a short numbered plan.

  = <plan>

= <solve(`"ship it"`)>
"""

DIRECT = "[plan] << Write a numbered plan for <problem>, <hint>."
PARENT = "a" * 64


class _Model:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: Any, info: AgentInfo) -> ModelResponse:
        self.prompts.append(str(messages[-1].parts[-1].content))
        if info.output_tools:
            tool = info.output_tools[0]
            properties = tool.parameters_json_schema.get("properties", {})
            value: Any = ["a", "b"] if "steps" in properties else "Ada"
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {name: value for name in properties})]
            )
        return ModelResponse(parts=[TextPart("1. a 2. b")])


def _run(
    source: str, forms: dict[str, Form], *, source_path: str | None = None
) -> tuple[Any, list[TelemetryAsset], _Model]:
    model = _Model()
    backend = RecordingTelemetryBackend(capture=frozenset({"content"}))
    with installed_recording_backend(backend):
        runtime = compile_program(
            parse_program(source, source_path=source_path),
            adapter=PydanticAdapter(FunctionModel(model)),
            forms=forms,
        )
        try:
            result = runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())
    return result, backend.assets, model


def _rejection(source: str, forms: dict[str, Form]) -> str:
    model = _Model()
    with pytest.raises(FormError) as caught:
        compile_program(
            parse_program(source), adapter=PydanticAdapter(FunctionModel(model)), forms=forms
        )
    assert model.prompts == []
    return str(caught.value)


def test_a_supplied_form_runs_in_place_of_the_source_body() -> None:
    form = Form(DIRECT, parent=PARENT, note="one call instead of two")
    result, assets, model = _run(PLANNER, {"solve.planner": form})

    assert result == "1. a 2. b"
    assert len(model.prompts) == 1
    assert "Write a numbered plan for ship it, be brief." in model.prompts[0]
    (recorded,) = [asset for asset in assets if asset.kind == "form"]
    assert recorded.source_locator == "kedi:group:main:solve.planner"
    assert recorded.content == DIRECT
    assert recorded.content_fingerprint == form_fingerprint(DIRECT)
    assert recorded.parent_fingerprint == PARENT
    assert dict(recorded.metadata) == {
        "group": "solve.planner",
        "group_kind": "evolve",
        "module": "main",
        "producer": "supplied",
        "note": "one call instead of two",
    }
    assert [asset.source_locator for asset in assets if asset.kind == "template"] == [
        "kedi:template:main:solve.planner#0"
    ]


def test_form_identity_comes_from_content_alone() -> None:
    _, source_run, _ = _run(PLANNER, {})
    canonical = parse_program(PLANNER).groups["solve.planner"].canonical
    indented = "\n".join("    " + line for line in canonical.splitlines())
    _, supplied_run, _ = _run(PLANNER, {"solve.planner": Form(f"# same body\n{indented}\n")})

    def form(assets: list[TelemetryAsset]) -> TelemetryAsset:
        return next(asset for asset in assets if asset.kind == "form")

    assert form(supplied_run).content_fingerprint == form(source_run).content_fingerprint
    assert form(supplied_run).metadata["producer"] == "supplied"


def test_forms_reach_imported_modules_by_module_name(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "helpers.kedi").write_text(
        "@helper(x: str) -> str:\n"
        "  > evolve: draft:\n"
        "    [answer] << Answer <x>.\n"
        "  = <answer>\n"
        "\n"
        "> export:\n"
        "  helper\n"
    )
    root = tmp_path / "root.kedi"
    root.write_text('> import: lib/helpers\n\n= <helper(`"q"`)>\n')
    forms = {"lib/helpers:helper.draft": Form("[answer] << Answer <x> in one word.")}

    _, assets, model = _run(root.read_text(), forms, source_path=str(root))

    assert "Answer q in one word." in model.prompts[0]
    (recorded,) = [asset for asset in assets if asset.kind == "form"]
    assert recorded.metadata["producer"] == "supplied"


def test_an_optimize_span_form_replaces_the_span_lines() -> None:
    source = "@f(x: str) -> str:\n  > optimize: draft:\n    >> Do <x> as [y]\n  = <y>\n"
    program = parse_program(source)
    compile_program(program, forms={"f.draft": Form(">> Do <x> carefully as [y]")})

    form = program.groups["f.draft"].form
    assert form is not None
    block = form.body[0]
    assert isinstance(block, TemplateBlock)
    assert program.procedures[0].optimize_spans[0].template_lines is block.lines


def test_an_optimizer_result_runs_as_an_optimizer_form() -> None:
    source = '@f(x: str) -> str:\n  > optimize: draft:\n    >> Name someone for <x>: [y]\n  = <y>\n= <f(`"x"`)>\n'
    program = parse_program(source)
    span_canonical = program.groups["f.draft"].canonical
    program.procedures[0].optimize_spans[0].template_lines[0].optimized_prompt_prefix = "Be kind."
    model = _Model()
    backend = RecordingTelemetryBackend(capture=frozenset({"content"}))
    with installed_recording_backend(backend):
        runtime = compile_program(program, adapter=PydanticAdapter(FunctionModel(model)))
        try:
            runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())

    (recorded,) = [asset for asset in backend.assets if asset.kind == "form"]
    assert recorded.content == f"Be kind.\n\n{span_canonical}"
    assert recorded.metadata["producer"] == "optimizer"
    assert recorded.parent_fingerprint == form_fingerprint(span_canonical)

    with pytest.raises(FormError, match=r"rejected \(optimizer\)"):
        compile_program(program, forms={"f.draft": Form(">> Other <x>: [y]")})


@pytest.mark.parametrize(
    ("text", "rule", "detail"),
    [
        ("= <problem>", "parse", "line 1: An '> evolve:' body cannot return"),
        ("[plan] << p\n> model: openai:gpt-6-luna", "parse", "line 2: An '> evolve:' body cannot"),
        ("> evolve: inner:\n  [plan] << p", "parse", "Groups cannot be nested"),
        ("   \n", "parse", "the form is empty"),
        ("[plan] << Plan `problem.upper()`.", "python", "a backtick expression"),
        ("```\nx = 1\n```\n[plan] << p", "python", "Python block"),
        ("> if: `True`:\n  >> x\n[plan] << p", "python", "a Python condition"),
        ("> loop [x]: `[1]`:\n  >> hi\n[plan] << p", "python", "a loop"),
        ("[plan: `str`] << p", "python", "a backtick type"),
        (">> Think about <problem>: [notes]", "interface", "must bind 'plan'"),
        ("[plan: int] << Count <problem>.", "interface", "str in the source but int"),
        ("[plan] << Plan <nothing>.", "interface", "cannot see: nothing"),
        ("[hint] = other\n[plan] << p", "interface", "the source does not: hint"),
        ("> if: <problem> is long\n  [hint] := short\n[plan] << p", "interface", "hint"),
    ],
    ids=[
        "return",
        "directive",
        "nested-group",
        "empty",
        "backtick",
        "python-block",
        "python-condition",
        "loop",
        "backtick-type",
        "missing-binding",
        "changed-type",
        "unknown-name",
        "rebinding",
        "reassignment",
    ],
)
def test_the_gate_rejects_a_form_that_breaks_a_rule(text: str, rule: str, detail: str) -> None:
    message = _rejection(PLANNER, {"solve.planner": Form(text)})
    assert f"Form for group 'solve.planner' rejected ({rule}): " in message
    assert detail in message


def test_an_optimize_span_form_is_one_template_block() -> None:
    source = "@f(x: str) -> str:\n  > optimize: draft:\n    >> Do <x> as [y]\n  = <y>\n"
    message = _rejection(source, {"f.draft": Form("[y] << Do <x>.")})
    assert "rejected (parse): an '> optimize:' form is exactly one template block" in message


def test_unknown_groups_and_modules_are_errors() -> None:
    assert "Form for unknown group 'solve.plan': module 'main' has groups solve.planner" in (
        _rejection(PLANNER, {"solve.plan": Form(DIRECT)})
    )
    assert "modules the program does not import: lib/x:g" in _rejection(
        PLANNER, {"lib/x:g": Form(DIRECT)}
    )


def test_forms_files_name_a_body_file_or_inline_text(tmp_path: Path) -> None:
    (tmp_path / "forms").mkdir()
    (tmp_path / "forms" / "planner-2.kedi").write_text(DIRECT + "\n")
    path = tmp_path / "forms.json"
    path.write_text(
        json.dumps(
            {
                "solve.planner": {"body": "forms/planner-2.kedi", "parent": PARENT, "note": "n"},
                "greeting": {"text": "[line] = hi"},
            }
        )
    )

    forms = load_forms(path)

    assert forms["solve.planner"] == Form(
        DIRECT + "\n", PARENT, "n", str(tmp_path / "forms" / "planner-2.kedi")
    )
    assert forms["greeting"] == Form("[line] = hi", None, None, "<form greeting>")


@pytest.mark.parametrize(
    ("content", "detail"),
    [
        ("[]", "must hold an object keyed by group name"),
        ("{", "Cannot read forms file"),
        ('{"g": "text"}', "group 'g': expected an object"),
        ('{"g": {"text": "x", "author": "me"}}', "unknown fields author"),
        ('{"g": {"text": "x", "body": "y"}}', "give exactly one of 'body' and 'text'"),
        ('{"g": {"note": "n"}}', "give exactly one of 'body' and 'text'"),
        ('{"g": {"text": 1}}', "every field must be a string"),
        ('{"g": {"body": "missing.kedi"}}', "cannot read body"),
        ('{"g": {"text": "x", "parent": "abc"}}', "64-character lowercase hex fingerprint"),
    ],
)
def test_invalid_forms_files_are_errors(tmp_path: Path, content: str, detail: str) -> None:
    path = tmp_path / "forms.json"
    path.write_text(content)
    with pytest.raises(FormError, match=detail):
        load_forms(path)


def test_tests_run_against_the_forms(tmp_path: Path) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text(
        "@label() -> str:\n"
        "  > evolve: pick:\n"
        "    [name] = first\n"
        "  = <name>\n"
        "\n"
        "@test: label:\n"
        "  > case: picks:\n"
        '    `assert label() == "first"`\n'
    )

    def outcomes(forms: dict[str, Form] | None) -> list[bool]:
        return [
            case.success for suite in run_tests(str(source), forms=forms) for case in suite.cases
        ]

    assert outcomes(None) == [True]
    assert outcomes({"label.pick": Form("[name] = second")}) == [False]


def test_the_cli_runs_forms_and_reports_rejections(tmp_path: Path) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text(
        "> evolve: greeting:\n  [word] = hello\n  [line] = <word> world\n\n= <line>\n"
    )
    accepted = tmp_path / "forms.json"
    accepted.write_text(json.dumps({"greeting": {"text": "[line] = goodbye world"}}))
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps({"greeting": {"text": "[line] = <nope>"}}))
    runner = CliRunner()

    assert runner.invoke(cli_main, [str(source)]).output == "hello world\n"
    assert runner.invoke(cli_main, [str(source), "--forms", str(accepted)]).output == (
        "goodbye world\n"
    )
    result = runner.invoke(cli_main, [str(source), "--forms", str(rejected)])
    assert result.exit_code == 1
    assert "rejected (interface): the form reads names the group cannot see: nope" in (
        result.output
    )
    combined = runner.invoke(cli_main, [str(source), "--forms", str(accepted), "--parse"])
    assert combined.exit_code == 2
    assert "--forms cannot be combined with --optimize or --parse." in combined.output


RICH = """> import: lib/things:
  thing

@helper(x: str) -> str:
  = <x>

@outer(topic: str) -> str:
  @inner(item: str) -> str:
    > evolve: nested:
      [label] << Label <item>.
    = <label>

  > loop [item]: `[topic]`:
    > evolve: looped:
      [note] << Note <item> with <inner(<topic>)>.
    >> Use <note> [kept]
  > map:
    > show: <kept>
  > loop: `False`:
    `topic = topic`
  > if: `True`:
    > evolve: branch:
      [pick] << Pick <topic>.
    = <pick>
  > else:
    = <topic>

> evolve: top:
  [answer] << Answer.
> evolve: second:
  [extra] << More.
```
this is not ( Python
```
> task [job]: reviewer:
  >> Review <answer>.
> await [review]: job
> task_group:
  > await [again]: job
  > process:
    > show: <again>
> show: <`answer`>
= <outer(<answer>)>
"""


def _apply(source: str, forms: dict[str, Form]) -> None:
    activation = FormActivation(forms)
    activation.apply(parse_program(source), module="main", root=True, module_names=())
    activation.finish()


def test_the_gate_sees_scopes_through_every_kind_of_block() -> None:
    _apply(
        RICH,
        {
            "top": Form("[answer] << Answer briefly."),
            "second": Form(">> Nothing the program reads later."),
            "inner.nested": Form("[label] << Name <item>."),
            "outer.looped": Form("[note] << Jot <item> via <helper(<inner(<topic>)>)>."),
            "outer.branch": Form("[pick] << Choose <topic> like <thing>."),
        },
    )


@pytest.mark.parametrize(
    ("key", "text", "detail"),
    [
        ("top", "[other] << x", "must bind 'answer'"),
        ("outer.looped", "[note] << <kept>", "cannot see: kept"),
        ("outer.branch", "[pick] << <item>", "cannot see: item"),
        ("outer.looped", "[note] << <helper(<`1`>)>", "a backtick expression"),
        ("outer.looped", "[note] << <helper(`1`)>", "a backtick expression"),
        ("outer.looped", ">> p [note: list[`str`] | None]", "a backtick type"),
        ("outer.looped", "[item] = other\n[note] << p", "the source does not: item"),
    ],
)
def test_scope_rules_follow_the_enclosing_blocks(key: str, text: str, detail: str) -> None:
    with pytest.raises(FormError, match=re.escape(detail)):
        _apply(RICH, {key: Form(text)})


def _runtime(source: str, model: _Model, **kwargs: Any) -> Any:
    return compile_program(
        parse_program(source), adapter=PydanticAdapter(FunctionModel(model)), **kwargs
    )


def _forms_used(backend: RecordingTelemetryBackend) -> list[tuple[str, Any, Any]]:
    return [
        (asset.metadata["producer"], asset.content, asset.parent_fingerprint)
        for asset in backend.assets
        if asset.kind == "form"
    ]


def test_a_proposed_form_runs_from_the_next_execution() -> None:
    model = _Model()
    backend = RecordingTelemetryBackend(capture=frozenset({"content"}))
    with installed_recording_backend(backend):
        runtime = _runtime(PLANNER, model)
        try:
            runtime.run_main()
            canonical = runtime._program.groups["solve.planner"].canonical
            fingerprint = runtime.propose_form(
                "solve.planner", DIRECT, parent=form_fingerprint(canonical), note="shorter"
            )
            runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())

    assert fingerprint == form_fingerprint(DIRECT)
    assert _forms_used(backend) == [
        ("source", canonical, None),
        ("runtime", DIRECT, form_fingerprint(canonical)),
    ]
    assert len(model.prompts) == 3
    assert "Write a numbered plan for ship it, be brief." in model.prompts[-1]


def test_a_rejected_proposal_keeps_the_active_form() -> None:
    model = _Model()
    backend = RecordingTelemetryBackend(capture=frozenset({"content"}))
    with installed_recording_backend(backend):
        runtime = _runtime(PLANNER, model, forms={"solve.planner": Form(DIRECT)})
        try:
            with pytest.raises(FormError, match=r"rejected \(interface\): .*cannot see: nothing"):
                runtime.propose_form("solve.planner", "[plan] << Plan <nothing>.")
            with pytest.raises(FormError, match="unknown group 'solve.plan'"):
                runtime.propose_form("solve.plan", DIRECT)
            with pytest.raises(FormError, match="the program has no module 'lib/x'"):
                runtime.propose_form("lib/x:g", DIRECT)
            runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())

    assert _forms_used(backend) == [("supplied", DIRECT, None)]


def test_an_execution_runs_one_form_from_start_to_end() -> None:
    source = (
        "> evolve: survey:\n"
        "  [first] << Name a city.\n"
        "  `propose()`\n"
        "  [second] << Name a river.\n"
        "= <first> <second>\n"
    )
    model = _Model()
    backend = RecordingTelemetryBackend(capture=frozenset({"content"}))
    proposals: list[str] = []
    with installed_recording_backend(backend):
        runtime: Any = None

        def propose() -> None:
            proposals.append(
                runtime.propose_form("survey", "[first] << Name a sea.\n[second] << Name a lake.")
            )

        runtime = _runtime(source, model, runtime_globals={"propose": propose})
        try:
            runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())

    assert len(proposals) == 1
    # Independent calls run concurrently, so they may reach the model in either order.
    assert sorted(model.prompts) == ["Name a city.", "Name a river."]
    assert [producer for producer, _, _ in _forms_used(backend)] == ["source"]


def test_proposals_reach_imported_modules(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "helpers.kedi").write_text(
        "@helper(x: str) -> str:\n"
        "  > evolve: draft:\n"
        "    [answer] << Answer <x>.\n"
        "  = <answer>\n"
        "\n"
        "> export:\n"
        "  helper\n"
    )
    root = tmp_path / "root.kedi"
    root.write_text('> import: lib/helpers\n\n= <helper(`"q"`)>\n')
    model = _Model()
    runtime = compile_program(
        parse_program(root.read_text(), source_path=str(root)),
        adapter=PydanticAdapter(FunctionModel(model)),
    )
    try:
        runtime.propose_form("lib/helpers:helper.draft", "[answer] << Answer <x> tersely.")
        runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())

    assert "Answer q tersely." in model.prompts[0]
