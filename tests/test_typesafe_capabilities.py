from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from kedi_typesafe import TypeSafeModel
from kedi_typesafe.integrations.langchain import TypeSafeChatModel
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.output_schema import build_output_model
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.agent_profile import AgentProfile
from kedi.decisions import decision_info_for_env
from kedi.lang import compile_program, parse_program

EXAMPLES = Path(__file__).parent / "fixtures" / "typesafe"


class Client:
    def __init__(self, probability=0.95):
        self.calls = []
        self.probability = probability

    async def system_one(self, state, questions, *, model=None):
        self.calls.append((state, questions))
        answers = {}
        for key, question in questions.items():
            if isinstance(question, Choice):
                options = list(question.criteria)
                value = next(
                    (option for option in options if option.startswith("__typesafe_no_matching")),
                    options[0],
                )
                answers[key] = ChoiceAnswer(
                    choice=value,
                    confidence=1,
                    probabilities={option: float(option == value) for option in options},
                )
            elif isinstance(question, Score):
                answers[key] = ScoreAnswer(
                    score=1.25,
                    confidence=0.5,
                    probabilities={0: 0, 1: 0.75, 2: 0.25},
                    legend=dict(enumerate(question.criteria)),
                )
            else:
                answers[key] = NoulAnswer(
                    noul=self.probability if not key.endswith("label2") else 0.1
                )
        return SystemOneResponse(
            model="test", answers=answers, usage=Usage(input_tokens=10, output_tokens=2)
        )

    async def aclose(self):
        pass


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
@pytest.mark.parametrize(
    ("filename", "expected", "count"),
    [
        ("01_rubric.kedi", 1.25, 1),
        ("02_probability.kedi", "high confidence", 1),
        ("03_choice.kedi", "billing", 1),
        ("04_boolean.kedi", "escalate", 1),
        ("05_labels.kedi", ["billing", "login"], 3),
        (
            "06_nested.kedi",
            {"routing": {"department": "billing", "urgent": True}, "requests_refund": True},
            3,
        ),
        ("07_nullable.kedi", {"department": None, "email": None}, 1),
        ("08_decision_evidence.kedi", 0.95, 1),
    ],
)
def test_capability_programs_execute_on_both_adapters(
    monkeypatch, adapter_name, filename, expected, count
):
    client = Client()
    if adapter_name == "pydantic":
        model = TypeSafeModel(client=client)
        adapter = PydanticAdapter(model)
        monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    else:
        model = TypeSafeChatModel(client=client)
        adapter = LangChainAdapter(model)
        monkeypatch.setattr(LangChainAdapter, "_resolve_model", lambda self, name: model)
    source = (
        (EXAMPLES / filename)
        .read_text()
        .replace("> adapter: pydantic", f"> adapter: {adapter_name}")
    )
    runtime = compile_program(parse_program(source), adapter=adapter)
    assert runtime.run_main() == expected
    assert len(client.calls) == 1
    assert len(client.calls[0][1]) == count
    if filename == "06_nested.kedi":
        info = decision_info_for_env(runtime._globals_env, "review")
        assert info is not None
        assert len(info.answers) == 3
        assert info.probability is None
    if filename == "08_decision_evidence.kedi":
        info = decision_info_for_env(runtime._globals_env, "duplicate_charge")
        assert info is not None
        assert info.provider == "typesafe"
        assert info.model == "test"
        assert info.probability == 0.95
        assert info.threshold == 0.85
        assert info.state_fingerprint is not None
        assert info.criteria_fingerprint is not None
        assert info.config_fingerprint is not None
        assert info.request_fingerprint is not None
        assert info.matches_inputs(ticket="I was charged twice for the same invoice.")
        assert len(client.calls) == 1


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
def test_decision_lifecycle_program_executes_on_both_adapters(monkeypatch, adapter_name):
    client = Client()
    if adapter_name == "pydantic":
        model = TypeSafeModel(client=client)
        adapter = PydanticAdapter(model)
        monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    else:
        model = TypeSafeChatModel(client=client)
        adapter = LangChainAdapter(model)
        monkeypatch.setattr(LangChainAdapter, "_resolve_model", lambda self, name: model)
    source = (
        (EXAMPLES / "09_decision_lifecycle.kedi")
        .read_text()
        .replace("> adapter: pydantic", f"> adapter: {adapter_name}")
    )

    runtime = compile_program(parse_program(source), adapter=adapter)

    assert runtime.run_main() == [True, True, True, True, True]
    assert len(client.calls) == 2
    assert [len(questions) for _, questions in client.calls] == [2, 1]


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
@pytest.mark.parametrize(("probability", "expected"), [(0.95, "billing"), (0.85, "review")])
def test_routing_program_only_returns_the_selected_path(
    monkeypatch, adapter_name, probability, expected
):
    client = Client(probability)
    if adapter_name == "pydantic":
        model = TypeSafeModel(client=client)
        adapter = PydanticAdapter(model)
        monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    else:
        model = TypeSafeChatModel(client=client)
        adapter = LangChainAdapter(model)
        monkeypatch.setattr(LangChainAdapter, "_resolve_model", lambda self, name: model)
    source = (
        (EXAMPLES / "routing.kedi")
        .read_text()
        .replace("> adapter: pydantic", f"> adapter: {adapter_name}")
    )
    runtime = compile_program(parse_program(source), adapter=adapter)
    assert runtime.run_main() == expected
    assert len(client.calls) == 1
    assert len(client.calls[0][1]) == 2


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
@pytest.mark.parametrize(("threshold", "expected"), [(0.94, "yes"), (0.95, "no"), (0.96, "no")])
def test_claim_profile_threshold_overrides_constructor(adapter_name, threshold, expected):
    client = Client()
    adapter = (
        PydanticAdapter(TypeSafeModel(client=client, threshold=0.99))
        if adapter_name == "pydantic"
        else LangChainAdapter(TypeSafeChatModel(client=client, threshold=0.99))
    )
    runtime = compile_program(
        parse_program("> if: Supported claim\n  = yes\n> else:\n  = no\n"),
        adapter=adapter,
        default_agent_profile=AgentProfile(settings={"decision_boolean_threshold": threshold}),
    )
    assert runtime.run_main() == expected
    assert len(client.calls) == 1


def test_jev_claim_with_tools_has_an_explicit_output_purpose():
    client = Client()

    def lookup() -> str:
        """Return current authoritative evidence for the claim."""

        return "Current evidence"

    tool = replace(python_callable_tool_spec("lookup", lookup), risk="read_only")
    profile = AgentProfile(
        tools=("lookup",),
        tool_bindings={"lookup": tool},
        approval=ApprovalPolicy.allow(),
    )
    runtime = compile_program(
        parse_program("> if: A current private claim\n  = yes\n> else:\n  = no\n"),
        adapter=PydanticAdapter(TypeSafeModel(client=client)),
        default_agent_profile=profile,
    )

    assert runtime.run_main() == "yes"
    state, questions = client.calls[0]
    assert "lookup" in questions["__typesafe_route"].criteria
    assert "instructions" in state
    assert "current context" in state["instructions"]
    assert "time-sensitive" in state["instructions"]


def test_output_schema_preserves_non_jev_constraints_and_description():
    from typing import Annotated

    from pydantic import Field, ValidationError

    output = build_output_model(
        {"probability": Annotated[float, "Claim probability", Field(ge=0, le=1)]}
    )
    field = output.model_json_schema()["properties"]["probability"]
    assert field["minimum"] == 0 and field["maximum"] == 1
    assert field["description"] == "Claim probability"
    with pytest.raises(ValidationError):
        output(probability=2)


def test_plain_upstream_model_cannot_silently_use_its_claim_threshold(monkeypatch):
    from pydantic_ai.models.typesafe import TypeSafeModel as UpstreamModel
    from typesafe_sdk import AsyncTypeSafeClient

    async def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported claim policy must fail before HTTP")

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(AsyncTypeSafeClient, "system_one", forbidden)
    runtime = compile_program(
        parse_program("> if: A claim\n  = yes\n> else:\n  = no\n"),
        adapter=PydanticAdapter(UpstreamModel("jev-latest")),
    )
    with pytest.raises(RuntimeError, match="enforce decision_boolean_threshold"):
        runtime.run_main()


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
@pytest.mark.parametrize(
    ("filename", "expected"),
    [("judge.kedi", 0.625), ("weighted_judge.kedi", 0.625), ("binary_judge.kedi", 1.0)],
)
def test_jev_judge_uses_existing_kedi_eval_surface(monkeypatch, adapter_name, filename, expected):
    from kedi.tests_runner import run_evals

    client = Client()
    if adapter_name == "pydantic":
        model = TypeSafeModel(client=client)
        adapter = PydanticAdapter(model)
        monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    else:
        model = TypeSafeChatModel(client=client)
        adapter = LangChainAdapter(model)
        monkeypatch.setattr(LangChainAdapter, "_resolve_model", lambda self, name: model)
    source = (
        (EXAMPLES / filename)
        .read_text()
        .replace("> adapter: pydantic", f"> adapter: {adapter_name}")
    )
    results = run_evals(source, adapter=adapter)
    assert results[0].metrics[0].score == expected
    assert len(client.calls) == 1
