from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from kedi_typesafe.core.evaluation import JSONValue
from kedi_typesafe.integrations.langchain import TypeSafeChatModel
from kedi_typesafe.integrations.pydantic import TypeSafeModel
from pydantic_ai.models import Model
from typesafe_sdk import Noul, NoulAnswer, Question, SystemOneResponse, Usage

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.models import (
    close_typesafe_model_current,
    is_typesafe_model_id,
    typesafe_langchain_model,
    typesafe_pydantic_model,
)
from kedi.agent_profile import AgentProfile
from kedi.artifacts import ArtifactCapabilityError, ArtifactPolicy
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


class _ClaimClient:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls: list[tuple[JSONValue, Mapping[str, Question], str | None]] = []

    async def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        self.calls.append((state, questions, model))
        key = next(iter(questions))
        return SystemOneResponse(
            model="jev-test",
            usage=Usage(input_tokens=12, output_tokens=1),
            answers={key: NoulAnswer(noul=self.probability)},
        )

    async def aclose(self) -> None:
        pass


def test_typesafe_model_identifier_builds_and_caches_pydantic_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    adapter = PydanticAdapter("typesafe/jev-latest")

    assert is_typesafe_model_id("typesafe/jev-latest")
    assert isinstance(adapter._agent_model_arg, TypeSafeModel)
    assert adapter._agent_model_arg.model_name == "jev-latest"
    assert adapter._resolve_model("typesafe/jev-latest") is adapter._agent_model_arg


@pytest.mark.parametrize("model_id", ["typesafe/", "typesafe//jev", "https://typesafe.ai/jev"])
def test_typesafe_model_identifier_rejects_invalid_names(model_id: str) -> None:
    with pytest.raises(ValueError):
        typesafe_pydantic_model(model_id)


def test_typesafe_factory_returns_pydantic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert isinstance(typesafe_pydantic_model("jev-latest"), Model)


@pytest.mark.parametrize(
    ("factory", "module_name", "missing_name", "install_hint"),
    [
        (
            typesafe_pydantic_model,
            "kedi_typesafe.integrations.pydantic",
            "kedi_typesafe",
            "kedi[typesafe]",
        ),
        (
            typesafe_langchain_model,
            "kedi_typesafe.integrations.langchain",
            "langchain_core",
            "kedi[langchain,typesafe]",
        ),
    ],
)
def test_typesafe_factories_report_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[str], object],
    module_name: str,
    missing_name: str,
    install_hint: str,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module_name:
            raise ModuleNotFoundError("missing optional dependency", name=missing_name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ModuleNotFoundError, match="Install Kedi") as exc_info:
        factory("typesafe/jev-latest")
    assert install_hint in str(exc_info.value)


@pytest.mark.parametrize(
    ("factory", "module_name"),
    [
        (typesafe_pydantic_model, "kedi_typesafe.integrations.pydantic"),
        (typesafe_langchain_model, "kedi_typesafe.integrations.langchain"),
    ],
)
def test_typesafe_factories_do_not_hide_transitive_import_errors(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[str], object],
    module_name: str,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module_name:
            raise ModuleNotFoundError("missing transport dependency", name="typesafe_transport")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ModuleNotFoundError, match="transport dependency"):
        factory("typesafe/jev-latest")


def test_typesafe_model_identifier_builds_and_caches_langchain_model() -> None:
    adapter = LangChainAdapter("typesafe/jev-latest")

    assert isinstance(adapter._model, TypeSafeChatModel)
    assert adapter._model.model_name == "jev-latest"
    assert adapter.default_model() == "typesafe/jev-latest"
    assert adapter._resolve_model("typesafe/jev-latest") is adapter._model
    assert isinstance(typesafe_langchain_model("jev-latest"), TypeSafeChatModel)


def test_typesafe_profile_model_disables_implicit_artifacts_for_both_adapters() -> None:
    profile = AgentProfile(model="typesafe/jev-latest")

    assert not PydanticAdapter().supports_artifact_contract(profile)
    assert not LangChainAdapter().supports_artifact_contract(profile)


def test_typesafe_model_cleanup_tolerates_models_without_loop_local_close() -> None:
    class LegacyTypeSafeModel:
        system = "typesafe"

    asyncio.run(close_typesafe_model_current(LegacyTypeSafeModel()))


def test_kedi_template_condition_uses_jev_noul_through_typed_produce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ClaimClient(probability=0.91)
    model = TypeSafeModel(client=client)
    closed: list[TypeSafeModel] = []

    async def close_current(active: TypeSafeModel) -> None:
        closed.append(active)

    monkeypatch.setattr(TypeSafeModel, "aclose_current", close_current)
    adapter = PydanticAdapter(model)
    runtime = compile_program(
        parse_program(
            "[status] = unknown\n"
            "> if: Ankara is the capital of Turkey\n"
            '  `status = "established"`\n'
            "> else:\n"
            '  `status = "not established"`\n'
            "= <status>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "established"
    assert len(client.calls) == 1
    state, questions, selected_model = client.calls[0]
    assert selected_model == "jev-latest"
    assert len(questions) == 1
    assert isinstance(next(iter(questions.values())), Noul)
    assert "Claim: Ankara is the capital of Turkey" in str(state)
    assert "current context" in state["instructions"]
    assert "reliable knowledge" in state["instructions"]
    assert closed == [model]


def test_kedi_template_condition_uses_false_for_low_jev_probability() -> None:
    client = _ClaimClient(probability=0.2)
    adapter = PydanticAdapter(TypeSafeModel(client=client))
    runtime = compile_program(
        parse_program(
            "[status] = unknown\n"
            "> if: Atlantis exists\n"
            '  `status = "established"`\n'
            "> else:\n"
            '  `status = "not established"`\n'
            "= <status>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "not established"
    assert len(client.calls) == 1


def test_kedi_template_condition_uses_jev_through_langchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ClaimClient(probability=0.91)
    model = TypeSafeChatModel(client=client)
    closed: list[TypeSafeChatModel] = []

    async def close_current(active: TypeSafeChatModel) -> None:
        closed.append(active)

    monkeypatch.setattr(TypeSafeChatModel, "aclose_current", close_current)
    adapter = LangChainAdapter(model)
    runtime = compile_program(
        parse_program(
            "[status] = unknown\n"
            "> if: Ankara is the capital of Turkey\n"
            '  `status = "established"`\n'
            "> else:\n"
            '  `status = "not established"`\n'
            "= <status>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "established"
    assert len(client.calls) == 1
    state, questions, selected_model = client.calls[0]
    assert selected_model == "jev-latest"
    assert isinstance(next(iter(questions.values())), Noul)
    assert "Claim: Ankara is the capital of Turkey" in str(state)
    assert closed == [adapter._model]


def test_typesafe_model_rejects_explicit_artifact_contract() -> None:
    client = _ClaimClient(probability=0.9)
    adapter = PydanticAdapter(TypeSafeModel(client=client))
    runtime = compile_program(
        parse_program("> if: established\n  = yes\n"),
        adapter=adapter,
        default_agent_profile=AgentProfile(artifacts=ArtifactPolicy(enabled=True)),
    )

    with pytest.raises(KediExecutionError, match="does not support artifact") as exc_info:
        runtime.run_main()
    assert isinstance(exc_info.value.original, ArtifactCapabilityError)
    assert client.calls == []


def test_langchain_typesafe_model_rejects_explicit_artifact_contract() -> None:
    client = _ClaimClient(probability=0.9)
    adapter = LangChainAdapter(TypeSafeChatModel(client=client))
    runtime = compile_program(
        parse_program("> if: established\n  = yes\n"),
        adapter=adapter,
        default_agent_profile=AgentProfile(artifacts=ArtifactPolicy(enabled=True)),
    )

    with pytest.raises(KediExecutionError, match="does not support artifact") as exc_info:
        runtime.run_main()
    assert isinstance(exc_info.value.original, ArtifactCapabilityError)
    assert client.calls == []


def test_typesafe_model_rejects_unconstrained_kedi_string_before_api_call() -> None:
    client = _ClaimClient(probability=0.9)
    adapter = PydanticAdapter(TypeSafeModel(client=client))
    runtime = compile_program(
        parse_program(">> Capital of Turkey is [city]\n= <city>\n"),
        adapter=adapter,
    )

    with pytest.raises(KediExecutionError, match="unsupported"):
        runtime.run_main()
    assert client.calls == []


def test_langchain_typesafe_model_rejects_unconstrained_kedi_string_before_api_call() -> None:
    client = _ClaimClient(probability=0.9)
    adapter = LangChainAdapter(TypeSafeChatModel(client=client))
    runtime = compile_program(
        parse_program(">> Capital of Turkey is [city]\n= <city>\n"),
        adapter=adapter,
    )

    with pytest.raises(KediExecutionError, match="unsupported"):
        runtime.run_main()
    assert client.calls == []
