from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import AsyncCallbackManager
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from pydantic_ai import UsageLimits

import kedi.agent_adapter.adapters.langchain.adapter as langchain_adapter_module
import kedi.agent_adapter.adapters.langchain.messages as langchain_messages_module
from kedi.agent_adapter.adapters import LangChainAdapter
from kedi.agent_adapter.adapters.langchain.runtime import (
    LangChainRunMiddleware,
    langchain_run_scope,
)
from kedi.agent_adapter.compaction.archive import cache_epoch_key


@pytest.mark.parametrize("retries", [True, -1, 1.5])
def test_retry_budget_requires_nonnegative_integer(retries):
    with pytest.raises(ValueError, match="tool_retries"):
        LangChainAdapter(tool_retries=retries)


def test_unsupported_native_session_and_model_setting():
    adapter = LangChainAdapter(object())

    async def run():
        async with adapter.responses_session():
            pytest.fail("unsupported transport opened")

    with pytest.raises(TypeError, match="does not support Responses sessions"):
        asyncio.run(run())

    class Model(BaseModel):
        name: str = "test"

    with pytest.raises(ValueError, match="Unsupported LangChain model setting"):
        LangChainAdapter(Model(), model_settings={"unsupported": True})._chat_model_for_run()


@pytest.mark.parametrize(
    "settings,model_key", [({}, None), ({"prompt_cache_key": "explicit"}, None), ({}, "native")]
)
def test_unscoped_middleware_preserves_requests_and_rotates_only_existing_keys(settings, model_key):
    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)
    messages = [
        HumanMessage(
            content="checkpoint", additional_kwargs={"kedi_compaction_checkpoint": "epoch"}
        )
    ]
    request = SimpleNamespace(
        messages=messages,
        model_settings=settings,
        model=SimpleNamespace(model_kwargs={"prompt_cache_key": model_key}),
    )
    request.override = lambda **changes: SimpleNamespace(**{**vars(request), **changes})
    seen = []

    async def handler(value):
        seen.append(value)
        return "result"

    async def run():
        assert await middleware.awrap_model_call(request, handler) == "result"
        assert await middleware.awrap_tool_call(request, handler) == "result"

    asyncio.run(run())
    key = settings.get("prompt_cache_key") or model_key
    if key:
        assert seen[0].model_settings["prompt_cache_key"] == cache_epoch_key(key, "epoch")
    else:
        assert seen[0] is request
    assert seen[1] is request


@pytest.mark.parametrize(
    "config,limits,explicit",
    [
        ({"recursion_limit": 42}, None, None),
        ({}, UsageLimits(request_limit=None), None),
        ({}, None, 9),
    ],
)
def test_native_run_recursion_configuration_is_preserved(config, limits, explicit, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def events(*args):
        yield SimpleNamespace(complete_run=lambda **kwargs: None)

    monkeypatch.setattr(langchain_adapter_module, "agent_event_run", events)
    monkeypatch.setattr(
        langchain_messages_module, "_langchain_event_handler", lambda _: SimpleNamespace()
    )
    callbacks = AsyncCallbackManager([])
    seen = []

    class Agent:
        async def ainvoke(self, payload, config):
            seen.append(config)
            return {"messages": []}

    payload = {"messages": [], "config": {**config, "callbacks": callbacks}, "usage_limits": limits}
    if explicit is not None:
        payload["recursion_limit"] = explicit
    asyncio.run(LangChainAdapter()._invoke_agent(Agent(), payload, emit_final=False))
    assert seen[0].get("recursion_limit") == (
        explicit if explicit is not None else config.get("recursion_limit")
    )
    assert callbacks.handlers == []
    assert len(seen[0]["callbacks"].handlers) == 1


@pytest.mark.parametrize(
    "metadata,reason",
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_messages"}},
            "max_messages",
        ),
        ({"status": "failed"}, "failed"),
        ({"finish_reason": "length"}, "length"),
        ({"stop_reason": "max_tokens"}, "max_tokens"),
        ({"finish_reason": "content_filter"}, "content_filter"),
        (
            {"status": "incomplete", "incomplete_details": {"reason": "Authorization: secret"}},
            "unknown",
        ),
    ],
)
def test_incomplete_model_result_reports_cause_and_accounts_usage(metadata, reason):
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)
    request = SimpleNamespace(
        messages=[HumanMessage("question")], model_settings={}, model=object()
    )

    async def handler(_):
        return SimpleNamespace(
            result=[
                AIMessage(
                    content="partial",
                    response_metadata=metadata,
                    usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
                )
            ]
        )

    async def run():
        with langchain_run_scope(None) as state:
            with pytest.raises(UnexpectedModelBehavior, match=rf"incomplete \(reason: {reason}\)"):
                await middleware.awrap_model_call(request, handler)
            assert state.native_usage.requests == 1
            assert state.usage.input_tokens == 10
            assert state.usage.output_tokens == 3

    asyncio.run(run())


def test_completion_check_uses_latest_message_and_preserves_valid_output():
    from kedi.agent_adapter.adapters.langchain.runtime import _check_model_completion

    _check_model_completion([])
    _check_model_completion([HumanMessage("question")])
    _check_model_completion(
        [
            AIMessage(content="old partial", response_metadata={"status": "incomplete"}),
            AIMessage(content="ready", response_metadata={"finish_reason": "stop"}),
        ]
    )
