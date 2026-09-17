from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec


class Result(BaseModel):
    value: int


def _model(framework: str, *, structured: bool = False) -> tuple[Any, dict[str, Any]]:
    current: ContextVar[int | None] = ContextVar("synthetic_turn", default=None)
    state: dict[str, Any] = {"enters": [], "exits": [], "requests": [], "connections": 0}

    class Lifecycle:
        supports_responses_session_isolation = True

        @asynccontextmanager
        async def responses_session(self, *, isolate: bool = False) -> AsyncIterator[None]:
            state["connections"] += 1
            yield

        @asynccontextmanager
        async def responses_turn(self) -> AsyncIterator[None]:
            turn_id = len(state["enters"]) + 1
            state["enters"].append(turn_id)
            token = current.set(turn_id)
            try:
                yield
            finally:
                state["exits"].append(turn_id)
                current.reset(token)

    if framework == "pydantic":

        class PydanticModel(Lifecycle, TestModel):
            async def request(self, *args: Any, **kwargs: Any) -> Any:
                assert current.get() is not None
                state["requests"].append(current.get())
                return await super().request(*args, **kwargs)

        return PydanticModel(), state

    class LangChainModel(Lifecycle, FakeMessagesListChatModel):
        def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
            return self

        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            assert current.get() is not None
            state["requests"].append(current.get())
            return super()._generate(*args, **kwargs)

    response = (
        AIMessage(content="", tool_calls=[{"name": "Result", "args": {"value": 1}, "id": "result"}])
        if structured
        else AIMessage(content="ok")
    )
    return LangChainModel(responses=[response]), state


@pytest.mark.parametrize("framework", ["pydantic", "langchain"])
@pytest.mark.parametrize("method", ["invoke", "produce"])
def test_adapter_turn_is_fresh_for_each_run_inside_shared_connection(
    framework: str, method: str
) -> None:
    model, state = _model(framework, structured=method == "produce")
    adapter = PydanticAdapter(model) if framework == "pydantic" else LangChainAdapter(model)

    async def invoke() -> None:
        if method == "produce":
            await adapter.produce("Produce a value.", output_type=Result)
        else:
            await adapter.invoke(prompt="hello")

    async def run() -> None:
        async with adapter.responses_session():
            await invoke()
            await invoke()
            assert state["connections"] == 1
            assert state["enters"] == state["exits"] == [1, 2]
            await asyncio.gather(invoke(), invoke())
        assert state["connections"] == 3
        assert sorted(state["enters"]) == sorted(state["exits"]) == [1, 2, 3, 4]
        assert set(state["requests"]) == {1, 2, 3, 4}

    asyncio.run(run())


@pytest.mark.parametrize("method", ["run", "iter"])
def test_native_pydantic_turn_encloses_tool_loop(method: str) -> None:
    model, state = _model("pydantic")
    adapter = PydanticAdapter(model)
    tools: list[int] = []

    def lookup() -> str:
        tools.append(1)
        return "synthetic"

    adapter.register_tool(replace(python_callable_tool_spec("lookup", lookup), risk="read_only"))

    async def run() -> None:
        async with adapter.responses_session():
            for _ in range(2):
                if method == "run":
                    await adapter.run("lookup")
                else:
                    async with adapter.iter("lookup") as agent_run:
                        async for _ in agent_run:
                            pass

    asyncio.run(run())
    assert tools == [1, 1]
    assert state["connections"] == 1
    assert state["enters"] == state["exits"] == [1, 2]
    assert state["requests"] == [1, 1, 2, 2]


def test_pydantic_cancellation_closes_entire_turn_and_connection_scope() -> None:
    model, state = _model("pydantic")
    started = asyncio.Event()
    waiting = asyncio.Event()
    original = model.request

    async def request(*args: Any, **kwargs: Any) -> Any:
        started.set()
        await waiting.wait()
        return await original(*args, **kwargs)

    model.request = request
    adapter = PydanticAdapter(model)

    async def run() -> None:
        async def consume() -> None:
            async with adapter.iter("wait") as agent_run:
                async for _ in agent_run:
                    pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert state["enters"] == state["exits"] == [1]
        waiting.set()
        await adapter.run("next")
        assert state["enters"] == state["exits"] == [1, 2]

    asyncio.run(run())
