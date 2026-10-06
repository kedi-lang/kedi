from __future__ import annotations

import asyncio
from contextvars import ContextVar
from time import monotonic

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from kedi.agent_adapter.cancellation import RunCancellation, RunCancelled, use_run_cancellation
from kedi.agent_adapter.execution import (
    RunDeadlineExceeded,
    run_sync,
    use_async_owner_loop,
    use_run_deadline,
)


@pytest.mark.asyncio
async def test_owner_loop_reuses_real_http_client_and_closes_on_owner():
    seen = []

    async def respond(request):
        seen.append(request.transport)
        return web.json_response({"value": len(seen)})

    app = web.Application()
    app.router.add_get("/", respond)
    async with TestServer(app) as server, httpx.AsyncClient() as client:
        owner = asyncio.get_running_loop()
        loops = []

        async def request():
            loops.append(asyncio.get_running_loop())
            return (await client.get(str(server.make_url("/")))).json()["value"]

        with use_async_owner_loop():
            assert [await asyncio.to_thread(lambda: run_sync(request())) for _ in range(3)] == [
                1,
                2,
                3,
            ]
        assert loops == [owner] * 3
        assert len({id(connection) for connection in seen}) == 1
        assert not client.is_closed
    assert client.is_closed
    assert not owner.is_closed()


@pytest.mark.asyncio
async def test_owner_loop_preserves_concurrent_contexts_and_restores_scope():
    label = ContextVar("test_label", default="outside")
    gate = asyncio.Event()
    seen = []
    owner = asyncio.get_running_loop()

    async def request():
        seen.append(label.get())
        if len(seen) == 2:
            gate.set()
        await gate.wait()
        return label.get(), asyncio.get_running_loop()

    def worker(value):
        token = label.set(value)
        try:
            return run_sync(request())
        finally:
            label.reset(token)

    with use_async_owner_loop():
        results = await asyncio.wait_for(
            asyncio.gather(asyncio.to_thread(worker, "first"), asyncio.to_thread(worker, "second")),
            5,
        )
    assert results == [("first", owner), ("second", owner)]
    assert label.get() == "outside"
    assert await asyncio.to_thread(lambda: run_sync(asyncio.sleep(0, result=17))) == 17


@pytest.mark.asyncio
async def test_owner_loop_cancellation_drains_only_cancelled_work():
    cancellation = RunCancellation()
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def request():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    def worker():
        with use_run_cancellation(cancellation), pytest.raises(RunCancelled):
            run_sync(request())

    with use_async_owner_loop():
        task = asyncio.create_task(asyncio.to_thread(worker))
        await asyncio.wait_for(entered.wait(), 5)
        cancellation.cancel()
        await asyncio.wait_for(task, 5)
        assert cleaned.is_set()
        assert await asyncio.to_thread(lambda: run_sync(asyncio.sleep(0, result="next"))) == "next"
    assert not cancellation._tasks


@pytest.mark.asyncio
async def test_owner_loop_deadline_and_exception_propagation():
    async def fail():
        raise ValueError("original error")

    def worker():
        with use_run_deadline(monotonic() + 0.02), pytest.raises(RunDeadlineExceeded):
            run_sync(asyncio.sleep(60))
        with pytest.raises(ValueError, match="original error"):
            run_sync(fail())

    with use_async_owner_loop():
        await asyncio.to_thread(worker)


@pytest.mark.asyncio
async def test_owner_loop_rejects_blocking_itself_without_leaking_coroutine():
    coroutine = asyncio.sleep(0)
    with use_async_owner_loop(), pytest.raises(RuntimeError, match="asyncio.to_thread"):
        run_sync(coroutine)
    assert coroutine.cr_frame is None


@pytest.mark.asyncio
async def test_owner_loop_rejects_future_without_cancelling_it():
    future = asyncio.get_running_loop().create_future()
    with use_async_owner_loop(), pytest.raises(RuntimeError, match="asyncio.to_thread"):
        run_sync(future)
    assert not future.done()
    future.set_result("still owned by caller")
    assert await future == "still owned by caller"


def test_adapter_cancellation_without_cancelled_owner_remains_async_cancellation():
    async def cancelled():
        raise asyncio.CancelledError

    with use_run_deadline(monotonic() + 10), pytest.raises(asyncio.CancelledError):
        run_sync(cancelled())


def test_owner_loop_requires_async_host():
    with pytest.raises(RuntimeError, match="no running event loop"), use_async_owner_loop():
        pass
