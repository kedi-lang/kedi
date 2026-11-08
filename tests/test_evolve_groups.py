from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI
from lsprotocol import types as lsp
from openai import AsyncOpenAI
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.explain import explain_program
from kedi.groups import form_fingerprint, module_name
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import GroupStmt, IfStmt, RawInvokeStmt, TemplateBlock
from kedi.lang.parser import KediParseError
from kedi.lsp.features.hover import compute_hover
from kedi.lsp.features.navigation import compute_definition
from kedi.telemetry import TelemetryAsset
from tests.telemetry_helpers import RecordingTelemetryBackend, installed_recording_backend

PLANNER = """@solve(problem: str) -> str:
  > evolve: planner:
    >> Break <problem> into steps: [steps: list[str]]
    [plan] << Turn <steps> into a short numbered plan.

  = <plan>

= <solve(`"ship it"`)>
"""


def _respond(messages: Any, info: AgentInfo) -> ModelResponse:
    if info.output_tools:
        tool = info.output_tools[0]
        properties = tool.parameters_json_schema.get("properties", {})
        value: Any = ["a", "b"] if "steps" in properties else "Ada"
        return ModelResponse(parts=[ToolCallPart(tool.name, {name: value for name in properties})])
    return ModelResponse(parts=[TextPart("1. a 2. b")])


def _run(source: str, *, source_path: str | None = None) -> tuple[Any, list[TelemetryAsset]]:
    backend = RecordingTelemetryBackend(capture=frozenset({"content"}))
    with installed_recording_backend(backend):
        runtime = compile_program(
            parse_program(source, source_path=source_path),
            adapter=PydanticAdapter(FunctionModel(_respond)),
        )
        try:
            result = runtime.run_main()
        finally:
            asyncio.run(runtime.aclose())
    return result, backend.assets


def _identities(assets: list[TelemetryAsset]) -> list[tuple[str, str, str | None, str]]:
    return sorted(
        (asset.kind, asset.source_locator, asset.definition_locator, asset.content_fingerprint)
        for asset in assets
        if asset.kind in {"form", "template"}
    )


def _parse_error_codes(source: str) -> list[str | None]:
    with pytest.raises(KediParseError) as caught:
        parse_program(source)
    return [diagnostic.code for diagnostic in caught.value.diagnostics]


def test_evolve_groups_parse_in_procedures_and_at_top_level() -> None:
    program = parse_program("> evolve: greeting:\n  >> Say hello to [name].\n" + PLANNER)

    group = program.procedures[0].body[0]
    assert isinstance(group, GroupStmt)
    assert (group.kind, group.name) == ("evolve", "planner")
    assert [type(statement) for statement in group.body] == [TemplateBlock, RawInvokeStmt]
    assert isinstance(program.toplevel_body[0], GroupStmt)
    assert sorted(program.groups) == ["greeting", "solve.planner"]


def test_optimize_spans_are_groups_sharing_the_name_space() -> None:
    program = parse_program(
        "@f(x: str) -> str:\n  > optimize: draft:\n    >> Do <x> [y]\n  = <y>\n"
    )

    group = program.groups["f.draft"]
    assert group.kind == "optimize"
    # The optimizer still owns the span's template lines, shared with the group's block.
    assert group.body[0].lines[0] is program.procedures[0].optimize_spans[0].template_lines[0]
    duplicate = (
        "@f() -> str:\n  > optimize: draft:\n    >> Do [y]\n  > evolve: draft:\n    >> More [z]\n"
        "  = <z>\n"
    )
    assert _parse_error_codes(duplicate) == ["K-GROUP-DUP"]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("  > evolve: inner:\n    >> x [y]\n", "K-EVOLVE-NESTED"),
        ("  > optimize: inner:\n    >> x [y]\n", "K-EVOLVE-NESTED"),
        ("  >> x [y]\n  = <y>\n", "K-EVOLVE-RETURN"),
        ("  > model: openai/gpt-6\n  >> x [y]\n", "K-EVOLVE-BODY"),
        ("  @helper():\n    = x\n", "K-EVOLVE-BODY"),
        ("  > if: `True`:\n    = done\n", "K-EVOLVE-RETURN"),
    ],
    ids=["nested-evolve", "nested-optimize", "return", "directive", "procedure", "nested-return"],
)
def test_group_bodies_cannot_change_their_surroundings(body: str, code: str) -> None:
    assert code in _parse_error_codes("> evolve: group:\n" + body)


def test_canonical_text_ignores_comments_and_indentation_but_keeps_template_text() -> None:
    source = (
        "@f(problem: str) -> str:\n"
        "  > evolve: g:\n"
        "    # a comment line\n"
        "    >> Break <problem> into steps:   [steps: list[str]]   # trailing note\n"
        "    > if: `len(steps) > 3`:\n"
        "        [plan] << Shorten <steps>.\n"
        "  = done\n"
    )
    canonical = parse_program(source).groups["f.g"].canonical
    assert canonical == (
        ">> Break <problem> into steps:   [steps: list[str]]   \n"
        "> if: `len(steps) > 3`:\n"
        "  [plan] << Shorten <steps>."
    )
    relaid = source.replace("    # a comment line\n", "").replace("        [plan]", "      [plan]")
    assert parse_program(relaid).groups["f.g"].canonical == canonical
    reworded = source.replace("Shorten", "Condense")
    assert parse_program(reworded).groups["f.g"].canonical != canonical


def test_canonical_text_drops_comment_lines_inside_a_statement() -> None:
    source = "@f() -> str:\n  > evolve: g:\n    >> First \\\n      # c\n      more [x]\n  = <x>\n"

    assert parse_program(source).groups["f.g"].canonical == ">> First \\\n  more [x]"


def test_group_bindings_stay_visible_after_the_group() -> None:
    result, _ = _run(PLANNER)
    assert result == "1. a 2. b"

    top_level, _ = _run("> evolve: greeting:\n  >> Name a person: [name: str]\n= Hello <name>\n")
    assert top_level == "Hello Ada"


def test_group_model_calls_belong_to_the_form_not_to_source_lines() -> None:
    _, assets = _run(PLANNER)

    (form,) = [asset for asset in assets if asset.kind == "form"]
    assert form.source_locator == "kedi:group:main:solve.planner"
    assert form.representation == "definition"
    canonical = parse_program(PLANNER).groups["solve.planner"].canonical
    assert form.content == canonical
    assert form.content_fingerprint == form_fingerprint(canonical)
    assert dict(form.metadata) == {
        "group": "solve.planner",
        "group_kind": "evolve",
        "module": "main",
        "producer": "source",
    }
    templates = sorted(
        (asset.source_locator, asset.definition_locator, asset.metadata["call_kind"])
        for asset in assets
        if asset.kind == "template"
    )
    assert templates == [
        ("kedi:template:main:solve.planner#0", "kedi:group:main:solve.planner", "produce"),
        ("kedi:template:main:solve.planner#1", "kedi:group:main:solve.planner", "invoke"),
    ]
    assert not [asset for asset in assets if asset.kind == "source_identity"]

    _, shifted = _run("# A new line above the group\n\n" + PLANNER)
    assert _identities(shifted) == _identities(assets)


def test_nested_statements_and_optimize_spans_have_group_ordinals() -> None:
    source = (
        "@f(x: str) -> str:\n"
        "  > optimize: draft:\n"
        "    >> Name someone for <x>: [name: str]\n"
        "  > evolve: refine:\n"
        "    > if: `True`:\n"
        "      >> Name another for <name>: [other: str]\n"
        "  = <name>\n"
        '= <f(`"x"`)>\n'
    )
    _, assets = _run(source)

    locators = {asset.source_locator for asset in assets if asset.kind in {"form", "template"}}
    assert locators == {
        "kedi:group:main:f.draft",
        "kedi:template:main:f.draft#0",
        "kedi:group:main:f.refine",
        "kedi:template:main:f.refine#0",
    }
    refine = parse_program(source).groups["f.refine"]
    assert isinstance(refine.body[0], IfStmt)


def test_loop_bodies_number_their_model_calls_in_source_order() -> None:
    source = (
        "> evolve: survey:\n"
        "  > loop [item]: `[1]`:\n"
        "    >> Describe <item>: [about: str]\n"
        "  > map:\n"
        "    >> Summarize: [short: str]\n"
        "  > loop: `False`:\n"
        "    >> Never asked: [never: str]\n"
        "  >> Name someone: [name: str]\n"
        "= <name>\n"
    )
    _, assets = _run(source)

    assert sorted({asset.source_locator for asset in assets if asset.kind == "template"}) == [
        "kedi:template:main:survey#0",
        "kedi:template:main:survey#1",
        "kedi:template:main:survey#3",
    ]


def test_auto_implementation_is_a_codegen_form() -> None:
    source = (
        "@shout(text: str) -> str:\n"
        "  > auto:\n"
        "    Return the text shouted.\n"
        "\n"
        "@shout(text: str) -> str:\n"
        "  [loud] << Shout <text>.\n"
        "  = <loud>\n"
        "\n"
        '= <shout(`"hi"`)>\n'
    )
    result, assets = _run(source)

    assert result == "1. a 2. b"
    (form,) = [asset for asset in assets if asset.kind == "form"]
    assert form.source_locator == "kedi:group:main:shout"
    assert form.content == "[loud] << Shout <text>.\n= <loud>"
    assert form.metadata["producer"] == "codegen"
    assert {asset.source_locator for asset in assets if asset.kind == "template"} == {
        "kedi:template:main:shout#0"
    }


def test_imported_modules_scope_groups_by_their_path_from_the_root(tmp_path: Path) -> None:
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

    _, assets = _run(root.read_text(), source_path=str(root))

    assert {asset.source_locator for asset in assets if asset.kind == "form"} == {
        "kedi:group:lib/helpers:helper.draft"
    }


def test_names_a_top_level_group_binds_are_star_exports(tmp_path: Path) -> None:
    (tmp_path / "values.kedi").write_text("> evolve: g:\n  [greeting: str] = hello\n> export: *\n")
    root = tmp_path / "root.kedi"
    root.write_text("> import: values:\n  greeting\n= <greeting>\n")

    result, _ = _run(root.read_text(), source_path=str(root))

    assert result == "hello"


def test_explain_walks_a_group_as_part_of_its_scope() -> None:
    (call,) = explain_program(parse_program("> evolve: g:\n  >> Answer is [answer: str].\n")).calls

    assert (call.scope, call.source_location.line) == ("top-level (group g)", 2)


def test_module_names_never_use_absolute_paths(tmp_path: Path) -> None:
    root = tmp_path / "app" / "root.kedi"
    assert module_name(root, root=root, spec=None) == "root"
    assert module_name(tmp_path / "app" / "lib" / "x.kedi", root=root, spec="lib/x") == "lib/x"
    assert module_name(tmp_path / "registry" / "main.kedi", root=root, spec="kedi_http") == (
        "kedi_http"
    )
    assert module_name(None, root=None, spec=None) == "main"
    assert module_name(tmp_path / "solo.kedi", root=None, spec=None) == "solo"


def _responses(bodies: list[bytes]) -> Any:
    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        body = json.loads(request.content)
        if any(tool.get("name") == "final_result" for tool in body.get("tools", [])):
            output: list[dict[str, Any]] = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "final_result",
                    "arguments": json.dumps({"steps": "a, b"}),
                    "status": "completed",
                }
            ]
        else:
            structured = body.get("text", {}).get("format", {}).get("type") == "json_schema"
            output = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"steps": "a, b"}) if structured else "1. a 2. b",
                            "annotations": [],
                        }
                    ],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(bodies)}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.2",
                "output": output,
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    return respond


async def _requests(source: str, adapter_kind: str) -> list[bytes]:
    bodies: list[bytes] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_responses(bodies))) as http:
        if adapter_kind == "pydantic":
            client = AsyncOpenAI(api_key="offline-fixture", http_client=http, max_retries=0)
            adapter: Any = PydanticAdapter(
                OpenAIResponsesModel("gpt-5.2", provider=OpenAIProvider(openai_client=client))
            )
        else:
            adapter = LangChainAdapter(
                ChatOpenAI(
                    model="gpt-5.2",
                    api_key="offline-fixture",
                    use_responses_api=True,
                    max_retries=0,
                    http_async_client=http,
                )
            )
        with use_async_owner_loop():
            runtime = compile_program(parse_program(source), adapter=adapter)
            try:
                await asyncio.to_thread(runtime.run_main)
            finally:
                await runtime.aclose()
    return bodies


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_kind", ["pydantic", "langchain"])
async def test_groups_without_forms_send_byte_identical_model_requests(adapter_kind: str) -> None:
    plain = (
        "@solve(problem: str) -> str:\n"
        "  >> Break <problem> into [steps: str].\n"
        "  [plan] << Plan <steps>.\n"
        "  = <plan>\n"
        '= <solve(`"ship it"`)>\n'
    )
    grouped = plain.replace(
        "  >> Break <problem> into [steps: str].\n  [plan] << Plan <steps>.\n",
        "  > evolve: planner:\n    >> Break <problem> into [steps: str].\n"
        "    [plan] << Plan <steps>.\n",
    )
    assert grouped != plain

    expected = await _requests(plain, adapter_kind)
    assert len(expected) == 2
    assert await _requests(grouped, adapter_kind) == expected


def test_editor_features_see_group_statements_in_their_enclosing_scope() -> None:
    source = (
        "@solve(problem: str) -> str:\n"
        "  > evolve: planner:\n"
        "    [plan: str] = `problem.upper()`\n"
        "  [x: str] = <plan>\n"
        "  = <x>\n"
    )
    hover = compute_hover(source, "solve.kedi", lsp.Position(line=1, character=13))
    assert hover is not None
    assert isinstance(hover.contents, lsp.MarkupContent)
    assert "group `solve.planner`" in hover.contents.value
    elsewhere = compute_hover(source, "solve.kedi", lsp.Position(line=3, character=2))
    assert elsewhere is not None
    assert "group `solve.planner`" not in str(elsewhere.contents)
    definition = compute_definition(
        source, "file:///solve.kedi", "solve.kedi", lsp.Position(line=3, character=15)
    )
    assert definition is not None
    assert (definition.range.start.line, definition.range.start.character) == (2, 5)
