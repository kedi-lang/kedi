from __future__ import annotations

import asyncio

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from kedi_typesafe.integrations.langchain import TypeSafeChatModel
from kedi_typesafe.integrations.pydantic import TypeSafeModel
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage

from kedi import UserPromptSubmitDecision
from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.adapters.pydantic.compat import (
    PydanticAICompatibilityError,
    configured_prompt_model,
)
from kedi.agent_adapter.models import model_template_prompt_mode
from kedi.agent_adapter.request import KediTemplatePromptBuilder
from kedi.agent_profile import AgentProfile
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program


def test_decision_prompt_preserves_context_and_capture_order() -> None:
    template = r"The label is \[ok\]. The review is [accepted: bool], with score [score: float]."
    prompt = KediTemplatePromptBuilder(
        template=template,
        output_fields=("accepted", "score"),
        mode="decision",
    ).build()
    assert prompt.kind == "template"
    assert prompt.content == (
        "Evaluate each marked decision field using its question, criteria, and the context below.\n"
        f"Decision fields: (accepted, score).\n\nContext:\n{template}"
    )
    assert "substitution" not in prompt.content
    assert "grammatically" not in prompt.content


def test_prompt_builder_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Template prompt mode"):
        KediTemplatePromptBuilder(template="", output_fields=(), mode="invalid").build()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("typesafe/jev-latest", "decision"),
        ("laya/local-model", "decision"),
        ("laya-mlx/local-model", "decision"),
        ("openai:gpt-5.6-luna", "completion"),
        (None, "completion"),
        (SimpleNamespace(kedi_prompt_mode="decision"), "decision"),
        (SimpleNamespace(kedi_prompt_mode="completion"), "completion"),
        (SimpleNamespace(system="typesafe"), "decision"),
        (SimpleNamespace(_llm_type="typesafe-jev"), "decision"),
        (SimpleNamespace(system="typesafe", kedi_prompt_mode="completion"), "completion"),
    ],
)
def test_prompt_mode_uses_model_contract(model: object, expected: str) -> None:
    assert model_template_prompt_mode(model) == expected


@pytest.mark.parametrize("mode", [None, "invalid", []])
def test_invalid_model_prompt_mode_fails_explicitly(mode: object) -> None:
    with pytest.raises(ValueError, match="kedi_prompt_mode"):
        model_template_prompt_mode(SimpleNamespace(kedi_prompt_mode=mode))


@pytest.mark.parametrize("adapter_type", [PydanticAdapter, LangChainAdapter])
def test_lexical_model_selection_is_lazy_and_has_no_shared_mutable_mode(adapter_type) -> None:
    adapter = adapter_type()
    profiles = [
        AgentProfile(model="typesafe/jev-latest"),
        AgentProfile(model="openai:gpt-5.6-luna"),
        AgentProfile(model="laya/local-model"),
        AgentProfile(),
    ] * 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert (
            list(pool.map(adapter.template_prompt_mode, profiles))
            == [
                "decision",
                "completion",
                "decision",
                "completion",
            ]
            * 8
        )


def test_pydantic_native_model_override_precedes_profile_model() -> None:
    class DecisionModel(TestModel):
        kedi_prompt_mode = "decision"

    adapter = PydanticAdapter(DecisionModel())
    assert adapter.template_prompt_mode(AgentProfile()) == "decision"
    assert adapter.template_prompt_mode(AgentProfile(model="openai:gpt-5.6-luna")) == "completion"
    with adapter.override(model=TestModel()):
        assert (
            adapter.template_prompt_mode(AgentProfile(model="typesafe/jev-latest")) == "completion"
        )
    with adapter.override(model=DecisionModel()):
        assert adapter.template_prompt_mode(AgentProfile(model="openai:gpt-5.6-luna")) == "decision"
    assert adapter.template_prompt_mode(AgentProfile()) == "decision"


def test_pydantic_static_model_capability_and_dynamic_selector() -> None:
    class DecisionModel(TestModel):
        kedi_prompt_mode = "decision"

    class ModelCapability(AbstractCapability):
        def get_model(self):
            return DecisionModel()

    def dynamic_model(_context: Any) -> Any:
        raise AssertionError("Prompt assembly must not execute a dependency-aware selector")

    class DynamicModelCapability(AbstractCapability):
        def get_model(self):
            return dynamic_model

    adapter = PydanticAdapter(TestModel(), capabilities=[ModelCapability()])
    assert adapter.template_prompt_mode(AgentProfile()) == "decision"
    assert adapter.template_prompt_mode(AgentProfile(model="openai:gpt-5.6-luna")) == "completion"
    dynamic = PydanticAdapter(TestModel(), capabilities=[DynamicModelCapability()])
    assert dynamic.template_prompt_mode(AgentProfile()) == "completion"


def test_pydantic_prompt_selection_reports_incompatible_layout() -> None:
    with pytest.raises(PydanticAICompatibilityError, match="unsupported Agent"):
        configured_prompt_model(object(), None)


class DecisionClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    async def system_one(self, state, questions, *, model=None):
        self.calls.append((state, questions))
        return SystemOneResponse(
            model="decision-test",
            usage=Usage(input_tokens=20, output_tokens=1),
            answers={key: NoulAnswer(noul=0.95) for key in questions},
        )

    async def aclose(self):
        pass


@pytest.mark.parametrize("surface", ["pydantic", "langchain"])
def test_claim_description_reaches_decision_question(surface) -> None:
    client = DecisionClient()
    adapter = (
        PydanticAdapter(TypeSafeModel(client=client))
        if surface == "pydantic"
        else LangChainAdapter(TypeSafeChatModel(client=client))
    )
    engine = SequentialEngine()
    runtime = compile_program(
        parse_program("> if: Ankara is the capital of Turkey\n  = `True`\n= `False`\n"),
        adapter=adapter,
        engine=engine,
    )
    try:
        assert runtime.run_main() is True
        assert len(client.calls) == 1
        _, questions = client.calls[0]
        assert "Whether the claim is established" in questions["response"].instructions
        assert "False if contradicted or unknown" in questions["response"].instructions
    finally:
        asyncio.run(runtime.aclose())
        engine.shutdown()


@pytest.mark.parametrize("surface", ["pydantic", "langchain"])
def test_decision_prompt_reaches_hooks_history_and_model_without_losing_criteria(surface) -> None:
    client = DecisionClient()
    adapter = (
        PydanticAdapter(TypeSafeModel(client=client))
        if surface == "pydantic"
        else LangChainAdapter(TypeSafeChatModel(client=client))
    )
    observed: list[str] = []

    @adapter.on("user_prompt_submit")
    def edit(event):
        observed.append(event.content)
        return UserPromptSubmitDecision.edit(event.content.replace("private account", "account"))

    runtime = compile_program(
        parse_program(
            "> history: enabled\n"
            "> settings:\n"
            "  typesafe_threshold: 0.90\n"
            '>> The private account payment was approved. The review is [approved: Annotated[bool, "Was the payment approved?"]].\n'
            "= <approved>\n"
        ),
        adapter=adapter,
    )
    assert runtime.run_main() == "True"
    assert len(client.calls) == len(observed) == 1
    assert observed[0].startswith("Evaluate each marked decision field")
    assert "private account" in observed[0]
    state, questions = client.calls[0]
    assert "private account" not in str(state)
    assert "Fill each unescaped" not in str(state)
    assert "Was the payment approved?" in questions["approved"].instructions
    history = runtime._implicit_conversation_state.snapshot()
    assert history[0].content == observed[0].replace("private account", "account")
    wire_prompt = state if isinstance(state, str) else state["messages"][0]["content"]
    assert history[0].content == wire_prompt


def test_regular_llm_keeps_completion_prompt_in_kedi_runtime() -> None:
    adapter = PydanticAdapter(TestModel(call_tools=[], custom_output_args={"approved": True}))
    observed: list[str] = []
    adapter.on("user_prompt_submit")(lambda event: observed.append(event.content))
    runtime = compile_program(
        parse_program(">> The review is [approved: bool].\n= <approved>\n"),
        adapter=adapter,
    )
    assert runtime.run_main() == "True"
    assert observed[0].startswith("Fill each unescaped")
