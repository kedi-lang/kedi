import asyncio
import inspect
from threading import Event

import pytest

from kedi import ApprovalDecision
from kedi.agent_adapter.approvals.policy import ApprovalPolicy, resolve_tool_approval_async
from kedi.agent_adapter.execution import run_sync
from tests.test_async_approval_owner import request


@pytest.mark.asyncio
async def test_repeated_cancellation_retains_approval_worker_until_cleanup_finishes():
    loop = asyncio.get_running_loop()
    entered, cleanup, release, finished = (asyncio.Event() for _ in range(4))

    async def inference():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            await release.wait()

    def handler(call):
        try:
            return run_sync(inference())
        finally:
            loop.call_soon_threadsafe(finished.set)

    task = asyncio.create_task(
        resolve_tool_approval_async(ApprovalPolicy.dynamic(handler), request())
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.wait_for(cleanup.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done(), "Owner returned while its approval model was still cleaning up"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        await asyncio.wait_for(finished.wait(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("result_kind", ["decision", "coroutine", "cancelled"])
async def test_cancelled_approval_discards_late_worker_result(result_kind):
    entered, release = Event(), Event()
    returned = []

    async def deferred():
        pytest.fail("Cancelled approval must not start a late coroutine")

    def handler(call):
        entered.set()
        assert release.wait(5)
        if result_kind == "cancelled":
            raise asyncio.CancelledError
        result = deferred() if result_kind == "coroutine" else ApprovalDecision.allow()
        returned.append(result)
        return result

    task = asyncio.create_task(
        resolve_tool_approval_async(ApprovalPolicy.dynamic(handler), request())
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    if result_kind == "coroutine":
        assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED
