import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.tools.specs import ToolSpec
from tests.test_langchain_branch_parity import ScriptedModel


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("sequential", [False, True])
async def test_native_tool_batch_respects_exclusive_spec(backend, sequential):
    active, peak, calls = 0, 0, []

    async def first() -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append("first")
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            return "first receipt"
        finally:
            active -= 1

    async def second() -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append("second")
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            return "second receipt"
        finally:
            active -= 1

    async def response(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("first", {}), ToolCallPart("second", {})])

    adapter = (
        PydanticAdapter(FunctionModel(response))
        if backend == "pydantic"
        else LangChainAdapter(
            ScriptedModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": name, "args": {}, "id": name} for name in ("first", "second")
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )
    )
    for fn in (first, second):
        adapter.register_tool(
            ToolSpec(
                fn=fn,
                name=fn.__name__,
                risk="read_only",
                sequential=sequential and fn is first,
                artifact_result="never",
            )
        )
    assert await adapter.invoke(prompt="Use both tools.") == "done"
    assert sorted(calls) == ["first", "second"]
    assert active == 0
    assert peak == (1 if sequential else 2)


def test_langchain_tool_roundtrip_preserves_sequential_flag_without_mutating_metadata():
    from kedi.agent_adapter.adapters.langchain.tools import (
        _langchain_tool_from_spec,
        _langchain_tool_spec_from_native,
    )

    def operation() -> str:
        return "done"

    metadata = {"source": "fixture"}
    original = ToolSpec(fn=operation, name="operation", metadata=metadata, sequential=True)
    native = _langchain_tool_from_spec(original)
    assert _langchain_tool_spec_from_native(native).sequential
    assert metadata == {"source": "fixture"}


@pytest.mark.asyncio
async def test_langchain_cancelled_exclusive_waiter_releases_reservation():
    from kedi.agent_adapter.adapters.langchain.runtime import (
        LangChainRunMiddleware,
        langchain_run_scope,
    )

    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)
    entered, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    holder_request = SimpleNamespace(tool_call={"name": "holder", "id": "holder"}, tool=None)
    exclusive_request = SimpleNamespace(
        tool_call={"name": "exclusive", "id": "exclusive"},
        tool=SimpleNamespace(metadata={"kedi_sequential": True}),
    )

    async def hold(_):
        entered.set()
        await release.wait()
        return "held"

    async def forbidden(_):
        pytest.fail("cancelled waiter entered its tool body")

    async def wait_exclusively():
        waiting.set()
        return await middleware.awrap_tool_call(exclusive_request, forbidden)

    async def probe(_):
        return "parallel"

    with langchain_run_scope(None) as state:
        holder = asyncio.create_task(middleware.awrap_tool_call(holder_request, hold))
        waiter = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            waiter = asyncio.create_task(wait_exclusively())
            await asyncio.wait_for(waiting.wait(), 2)
            assert state.pending_tools == 2
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert state.pending_tools == 1
            assert (
                await asyncio.wait_for(middleware.awrap_tool_call(holder_request, probe), 2)
                == "parallel"
            )
        finally:
            release.set()
            if waiter is not None and not waiter.done():
                waiter.cancel()
            await asyncio.gather(
                holder, *([waiter] if waiter is not None else []), return_exceptions=True
            )
        assert state.pending_tools == 0
        assert state.native_usage.tool_calls == 2


@pytest.mark.asyncio
async def test_langchain_exclusive_gate_is_run_local_and_releases_after_failure():
    from kedi.agent_adapter.adapters.langchain.runtime import (
        LangChainRunMiddleware,
        langchain_run_scope,
    )

    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)
    entered, release = asyncio.Event(), asyncio.Event()
    request = SimpleNamespace(
        tool_call={"name": "operation", "id": "operation"},
        tool=SimpleNamespace(metadata={"kedi_sequential": True}),
    )

    async def fail_after_release(_):
        entered.set()
        await release.wait()
        raise RuntimeError("fixture body failure")

    async def succeed(_):
        return "done"

    with langchain_run_scope(None) as outer:
        holder = asyncio.create_task(middleware.awrap_tool_call(request, fail_after_release))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            with langchain_run_scope(None) as inner:
                assert (
                    await asyncio.wait_for(middleware.awrap_tool_call(request, succeed), 2)
                    == "done"
                )
                assert inner.native_usage.tool_calls == 1
                assert outer.native_usage.tool_calls == 0
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="fixture body failure"):
                await holder
        assert outer.pending_tools == 0
        assert await asyncio.wait_for(middleware.awrap_tool_call(request, succeed), 2) == "done"
        assert outer.native_usage.tool_calls == 1
