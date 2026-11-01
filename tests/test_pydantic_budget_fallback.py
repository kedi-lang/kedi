import asyncio
from types import SimpleNamespace
from typing import Any

from kedi.agent_adapter.adapters.pydantic.budget import _fallback_judgment


def _judge(model: Any, value: Any) -> bool:
    return asyncio.run(_fallback_judgment(model, value))


def test_judgment_asks_the_wrapped_models_fallback_policy():
    async def should_fallback(value: Any) -> bool:
        return value == "retry"

    model = SimpleNamespace(_predicates=SimpleNamespace(should_fallback=should_fallback))
    assert _judge(model, "retry") is True
    assert _judge(model, "stop") is False
