import asyncio
from types import SimpleNamespace
from typing import Any

from kedi.agent_adapter.adapters.pydantic.budget import _fallback_judgment


def _judge(model: Any, value: Any) -> bool:
    return asyncio.run(_fallback_judgment(model, value))


def test_judgment_asks_the_policy_object_that_newer_releases_attach():
    async def should_fallback(value: Any) -> bool:
        return value == "retry"

    model = SimpleNamespace(_predicates=SimpleNamespace(should_fallback=should_fallback))
    assert _judge(model, "retry") is True
    assert _judge(model, "stop") is False


def test_judgment_calls_the_model_method_of_older_releases():
    async def _should_fallback(value: Any) -> bool:
        return value == "retry"

    model = SimpleNamespace(_should_fallback=_should_fallback)
    assert _judge(model, "retry") is True
    assert _judge(model, "stop") is False
