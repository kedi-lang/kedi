"""Execute the README's Kedi source, including its actual decision criteria."""

import re
from io import StringIO
from pathlib import Path

import pydantic_ai.models
import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

import kedi
from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program

README = Path(__file__).resolve().parents[1] / "README.md"
SOURCES = dict(
    re.findall(
        r"(?m)^<!-- kedi-example: ([a-z-]+) -->\n\n```python\n([\s\S]*?)^```\s*$",
        README.read_text(),
    )
)


@pytest.fixture(autouse=True)
def isolated_api_config():
    from kedi.api.state import _DEFAULT_CONFIG, _config_var

    token = _config_var.set(_DEFAULT_CONFIG)
    try:
        yield
    finally:
        _config_var.reset(token)


@pytest.fixture
def run_example(monkeypatch):
    monkeypatch.delenv("KEDI_ADAPTER_MODEL", raising=False)
    monkeypatch.delenv("KEDI_AGENT", raising=False)
    runtimes = []

    def run(name, model):
        runtime = compile_program(parse_program(SOURCES[name]), adapter=PydanticAdapter(model))
        runtimes.append(runtime)
        return runtime.run_main()

    yield run
    for runtime in runtimes:
        runtime.artifacts.close()


def test_readme_marks_all_kedi_examples():
    assert set(SOURCES) == {"actions", "release-notes", "subagents", "jev", "rubric", "cloud-agent"}
    for source in SOURCES.values():
        parse_program(source)


def test_readme_python_api_returns_the_capture(monkeypatch):
    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.delenv("KEDI_ADAPTER_MODEL", raising=False)
    monkeypatch.delenv("KEDI_AGENT", raising=False)
    source = re.search(r"(?m)^```python\n(import kedi\n[\s\S]*?)^```", README.read_text())
    assert source is not None
    notice = "Checkout is timing out; recovery has not been confirmed."
    model = TestModel(call_tools=[], custom_output_args={"notice": notice})
    namespace = {}
    with kedi.context(adapter=PydanticAdapter(model), model=None):
        exec(compile(source[1], str(README), "exec"), namespace)
    assert namespace["notice"] == notice


def test_readme_repl_preserves_typed_captures_between_inputs():
    from kedi.idle import IdleConsole

    transcript = re.search(
        r"<!-- kedi-repl-example -->\n\n```console\n([\s\S]*?)\n```", README.read_text()
    )
    assert transcript is not None
    stdout, stderr = StringIO(), StringIO()
    model = TestModel(call_tools=[], custom_output_args={"capital": "Stockholm"})
    with kedi.interactive(adapter=PydanticAdapter(model), model=None) as session:
        console = IdleConsole(session, stdout=stdout, stderr=stderr)
        for line in transcript[1].splitlines():
            if line.startswith("+++ "):
                assert console.push(line[4:]) is False

    assert stderr.getvalue() == ""
    assert stdout.getvalue() == "Stockholm\nSTOCKHOLM\n"


def test_readme_actions_return_native_typed_objects(run_example):
    action = {"task": "Add regression tests", "owner": "Sam", "deadline": "Friday"}
    result = run_example(
        "actions", TestModel(call_tools=[], custom_output_args={"actions": [action]})
    )
    assert isinstance(result[0], BaseModel)
    assert result[0].model_dump() == action


def test_readme_conditions_filter_internal_changes(run_example):
    calls = []

    def respond(messages, info):
        prompt = str(messages[-1])
        calls.append(prompt)
        tool = info.output_tools[0]
        if "Claim:" in prompt:
            result = {
                next(iter(tool.parameters_json_schema["properties"])): "fixture" not in prompt
            }
        else:
            result = {"note": "CSV exports preserve accents."}
        return ModelResponse(parts=[ToolCallPart(tool.name, result)])

    assert run_example("release-notes", FunctionModel(respond)) == ["CSV exports preserve accents."]
    assert len(calls) == 3


def test_readme_coordinator_delegates_and_combines_both_reviews(run_example):
    calls = []
    recommendation = "Block release until authorization and cleanup are fixed."

    def respond(messages, info):
        if "delegate_task" in {tool.name for tool in info.function_tools}:
            calls.append("coordinator")
            reviews = [
                part
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_name == "delegate_task"
            ]
            if not reviews:
                assert "exports every account's data" in str(messages)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "delegate_task",
                            {
                                "subagent": reviewer,
                                "task": "Review an endpoint that exports every account's data "
                                "and closes its connection only on success.",
                            },
                            tool_call_id=reviewer,
                        )
                        for reviewer in ("security", "reliability")
                    ]
                )
            assert len(reviews) == 2
            assert "Missing authorization" in str(reviews)
            assert "Missing cleanup" in str(reviews)
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"recommendation": recommendation})]
            )

        assert "closes its connection only on success" in str(messages)
        issue = (
            "Missing authorization"
            if "missing authorization" in str(messages)
            else "Missing cleanup"
        )
        calls.append(issue)
        return ModelResponse(parts=[TextPart(issue)])

    assert run_example("subagents", FunctionModel(respond)) == recommendation
    assert calls[0] == calls[-1] == "coordinator"
    assert sorted(calls[1:-1]) == ["Missing authorization", "Missing cleanup"]


@pytest.mark.parametrize(
    "probability,expected", [(0.95, "Escalate to billing."), (0.8, "Queue for billing.")]
)
def test_readme_jev_criteria_and_threshold(run_example, monkeypatch, probability, expected):
    from kedi_typesafe import TypeSafeModel
    from typesafe_sdk import Choice, ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage

    class Client:
        calls = 0

        async def system_one(self, state, questions, *, model=None):
            self.calls += 1
            assert "every hour" in str(state)
            assert len(questions) == 2
            assert "Ongoing harm" in str(questions)
            assert "hypothetical" in str(questions)
            return SystemOneResponse(
                model="test-jev",
                answers={
                    name: ChoiceAnswer(
                        choice="billing",
                        confidence=1,
                        probabilities={"billing": 1, "access": 0, "general": 0},
                    )
                    if isinstance(question, Choice)
                    else NoulAnswer(noul=probability)
                    for name, question in questions.items()
                },
                usage=Usage(input_tokens=50, output_tokens=2),
            )

        async def aclose(self):
            pass

    client = Client()
    model = TypeSafeModel(client=client)
    monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    assert run_example("jev", model) == expected
    assert client.calls == 1


def test_readme_rubric_reconstructs_nested_scores(run_example, monkeypatch):
    from kedi_typesafe import TypeSafeModel
    from typesafe_sdk import Noul, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

    class Client:
        calls = 0

        async def system_one(self, state, questions, *, model=None):
            self.calls += 1
            assert len(questions) == 2
            assert "Specific, actionable next step" in str(questions)
            return SystemOneResponse(
                model="test-jev",
                answers={
                    name: NoulAnswer(noul=0.95)
                    if isinstance(question, Noul)
                    else ScoreAnswer(
                        score=1.75,
                        confidence=0.9,
                        legend={
                            0: "No next step",
                            1: "Vague suggestion",
                            2: "Specific, actionable next step",
                        },
                        probabilities={0: 0, 1: 0.25, 2: 0.75},
                    )
                    for name, question in questions.items()
                },
                usage=Usage(input_tokens=50, output_tokens=2),
            )

        async def aclose(self):
            pass

    client = Client()
    model = TypeSafeModel(client=client)
    monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    result = run_example("rubric", model)
    assert isinstance(result, BaseModel)
    assert result.grounded == 0.95
    assert result.helpfulness == 1.75
    assert client.calls == 1
