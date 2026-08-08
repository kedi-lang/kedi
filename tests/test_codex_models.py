from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

import kedi.agent_adapter.models as model_module
from kedi.agent_adapter import codex_responses_model
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.adapters import pydantic as pydantic_adapter_module
from kedi.agent_profile import AgentProfile


def _fake_helper(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, str, str]],
) -> None:
    helper = ModuleType("codex_auth_helper")

    def pydantic_factory(model: str, *, instructions: str) -> object:
        calls.append(("pydantic", model, instructions))
        return object()

    def langchain_factory(model: str, *, instructions: str) -> object:
        calls.append(("langchain", model, instructions))
        return object()

    helper.create_codex_responses_model = pydantic_factory  # type: ignore[attr-defined]
    helper.create_codex_chat_openai = langchain_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "codex_auth_helper", helper)
    monkeypatch.setattr(model_module, "find_spec", lambda _name: object())


def test_codex_responses_model_builds_supported_framework_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    _fake_helper(monkeypatch, calls)

    assert codex_responses_model("codex/gpt-5.6-luna", "pydantic") is not None
    assert codex_responses_model("gpt-5.6-luna", "langchain") is not None

    assert [(adapter, model) for adapter, model, _instructions in calls] == [
        ("pydantic", "gpt-5.6-luna"),
        ("langchain", "gpt-5.6-luna"),
    ]
    assert all("active instructions" in instructions for _, _, instructions in calls)


def test_codex_responses_model_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_module, "find_spec", lambda _name: None)

    with pytest.raises(ModuleNotFoundError, match=r"kedi\[codex-model\]"):
        codex_responses_model("codex/gpt-5.6-luna", "pydantic")


def test_codex_responses_model_reports_missing_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = ModuleType("codex_auth_helper")

    def missing_login(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("auth.json")

    helper.create_codex_responses_model = missing_login  # type: ignore[attr-defined]
    helper.create_codex_chat_openai = missing_login  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "codex_auth_helper", helper)
    monkeypatch.setattr(model_module, "find_spec", lambda _name: object())

    with pytest.raises(RuntimeError, match="codex login"):
        codex_responses_model("codex/gpt-5.6-luna", "pydantic")


@pytest.mark.parametrize("model_id", ["codex/", "codex/   ", "https://example.test/model"])
def test_codex_responses_model_rejects_invalid_ids(model_id: str) -> None:
    with pytest.raises(ValueError, match="Codex model id"):
        codex_responses_model(model_id, "pydantic")


def test_pydantic_adapter_resolves_and_reuses_codex_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    models = {
        "codex/gpt-5.6-luna": TestModel(custom_output_text="default"),
        "codex/gpt-5.6-terra": TestModel(custom_output_text="override"),
    }

    def build(model_id: str, *, adapter: str) -> Any:
        assert adapter == "pydantic"
        calls.append(model_id)
        return models[model_id]

    monkeypatch.setattr(pydantic_adapter_module, "codex_responses_model", build)

    adapter = PydanticAdapter("codex/gpt-5.6-luna")
    assert adapter.default_model() == "codex/gpt-5.6-luna"
    assert calls == ["codex/gpt-5.6-luna"]

    with adapter.profile_override(AgentProfile(model="codex/gpt-5.6-terra", effort="high")):
        assert adapter.active_profile.model == "codex/gpt-5.6-terra"
    with adapter.profile_override(AgentProfile(model="codex/gpt-5.6-terra")):
        pass

    assert calls == ["codex/gpt-5.6-luna", "codex/gpt-5.6-terra"]


def test_codex_responses_model_rejects_unknown_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    _fake_helper(monkeypatch, calls)

    with pytest.raises(ValueError, match="Unsupported Codex model adapter"):
        codex_responses_model("codex/gpt-5.6-luna", "dspy")  # type: ignore[arg-type]
