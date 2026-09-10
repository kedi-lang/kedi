from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from kedi.agent_adapter.adapters import LangChainAdapter
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.prefix_cache import langchain_provider_cache_policy
from kedi.agent_adapter.native_compaction import langchain_settings_with_native_compaction
from kedi.agent_profile.compaction import CompactionSettings
from kedi.agent_profile import AgentProfile


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=SecretStr("synthetic"),
        use_responses_api=True,
        store=False,
        reasoning={"effort": "low", "summary": "auto"},
        model_kwargs={"instructions": "native", "prompt_cache_key": "model-owned"},
    )


@pytest.mark.parametrize("key", [None, "adapter-owned"])
def test_native_settings_preserve_model_kwargs_and_explicit_cache_identity(key: str | None) -> None:
    model = _model()
    settings: dict[str, Any] = {"reasoning": {"effort": "high"}, "max_tokens": 128}
    if key:
        settings["prompt_cache_key"] = key
    before = deepcopy(settings)
    adapter = LangChainAdapter(model, model_settings=settings)
    with adapter.conversation_scope(AdapterConversationContext(cache_key="generated")):
        effective = adapter._chat_model_for_run()
    assert effective.reasoning == {"effort": "high"}
    assert effective.max_tokens == 128
    assert effective.model_kwargs == {
        "instructions": "native",
        "prompt_cache_key": key or "model-owned",
    }
    assert effective.store is False
    assert effective.use_previous_response_id is False
    assert model.reasoning == {"effort": "low", "summary": "auto"}
    assert model.model_kwargs == {"instructions": "native", "prompt_cache_key": "model-owned"}
    assert settings == before


def test_native_kwargs_layering_and_profile_effort() -> None:
    adapter = LangChainAdapter(
        _model(),
        model_kwargs={"custom": "constructor"},
        model_settings={"model_kwargs": {"instructions": "adapter"}},
    )
    with adapter.profile_override(AgentProfile(effort="high", settings={"num_retries": 9})):
        effective = adapter._chat_model_for_run()
    assert effective.reasoning == {"effort": "high", "summary": "auto"}
    assert effective.model_kwargs == {
        "instructions": "adapter",
        "prompt_cache_key": "model-owned",
        "custom": "constructor",
    }
    assert "num_retries" not in effective.model_kwargs


def test_openai_subclasses_keep_provider_cache_policy() -> None:
    class NativeSubclass(ChatOpenAI):
        pass

    model = NativeSubclass(model="gpt-5.6-luna", api_key=SecretStr("synthetic"), store=False)
    adapter = LangChainAdapter(model)
    with adapter.conversation_scope(AdapterConversationContext(cache_key="generated")):
        effective = adapter._chat_model_for_run()
    assert langchain_provider_cache_policy(model).stable_cache_key
    assert effective.model_kwargs["prompt_cache_key"] == "generated"
    settings = langchain_settings_with_native_compaction(
        {}, model=model, settings=CompactionSettings(mode="native", threshold=100_000)
    )
    assert "context_management" in settings


@pytest.mark.parametrize("settings", [{"base_url": "https://example.test/v1"}, {"max_retries": 7}])
def test_initialized_transport_settings_are_not_silently_ignored(settings: dict[str, Any]) -> None:
    adapter = LangChainAdapter(_model(), model_settings=settings)
    with pytest.raises(ValueError, match="native model constructor"):
        adapter._chat_model_for_run()


def test_langchain_run_sessions_share_sequential_calls_and_isolate_child_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[bool] = []
    exits: list[bool] = []

    class Model:
        supports_responses_session_isolation = True

        @asynccontextmanager
        async def responses_session(self, *, isolate: bool = False):
            events.append(isolate)
            try:
                yield
            finally:
                exits.append(True)

    class Agent:
        async def ainvoke(self, payload: Any, config: Any = None) -> Any:
            await asyncio.sleep(0)
            return {"output": "ok"}

    adapter = LangChainAdapter(Model())
    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: Agent())

    async def run() -> None:
        async with adapter.responses_session():
            assert await adapter.invoke(prompt="first") == "ok"
            async with adapter.responses_session():
                assert await adapter.invoke(prompt="second") == "ok"
            assert events == [True]
            await asyncio.gather(
                adapter.invoke(prompt="child one"), adapter.invoke(prompt="child two")
            )
            assert events == [True, True, True]
        assert len(exits) == 3
        assert await adapter.invoke(prompt="independent") == "ok"
        assert len(exits) == 4

    asyncio.run(run())
