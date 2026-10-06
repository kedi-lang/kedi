from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from kedi.agent_adapter.cancellation import RunCancellation, RunCancelled, use_run_cancellation
from kedi.agent_adapter.execution import OwnedAsyncLoop, run_sync


@pytest.mark.parametrize("inside_loop", [False, True])
def test_cancel_drains_owned_work_before_returning(inside_loop: bool) -> None:
    owner = RunCancellation()
    entered, cleaned = Event(), Event()

    async def work() -> None:
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    def call() -> None:
        with use_run_cancellation(owner):
            with pytest.raises(RunCancelled, match="interrupted"):
                run_sync(work())
        assert cleaned.is_set()

    async def outer() -> None:
        call()

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(lambda: asyncio.run(outer()) if inside_loop else call())
        assert entered.wait(5)
        owner.cancel()
        owner.cancel()
        running.result(timeout=5)
    assert owner._tasks == set()
    assert run_sync(asyncio.sleep(0, result="next owner")) == "next owner"


def test_cancel_before_admission_never_starts_work() -> None:
    owner = RunCancellation()
    owner.cancel()
    entered = []

    async def work() -> None:
        entered.append(True)

    with use_run_cancellation(owner), pytest.raises(RunCancelled):
        run_sync(work())
    assert entered == []
    assert owner._tasks == set()


def test_cancel_fences_result_when_coroutine_suppresses_cancellation() -> None:
    owner = RunCancellation()
    entered = Event()

    async def work() -> str:
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return "stale result"

    def call() -> None:
        with use_run_cancellation(owner), pytest.raises(RunCancelled):
            run_sync(work())

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(call)
        assert entered.wait(5)
        owner.cancel()
        running.result(timeout=5)


def test_cancel_isolates_sibling_owner_and_preserves_nested_owner() -> None:
    cancelled_owner, sibling_owner, nested_owner = (RunCancellation() for _ in range(3))
    entered = [Event(), Event()]
    released = Event()

    async def work(index: int) -> str:
        entered[index].set()
        while not released.is_set():
            await asyncio.sleep(0.01)
        return "sibling result"

    def call(index: int) -> str:
        with use_run_cancellation(cancelled_owner if index == 0 else sibling_owner):
            if index == 0:
                with use_run_cancellation(nested_owner), pytest.raises(RunCancelled):
                    run_sync(work(index))
                return "cancelled"
            return run_sync(work(index))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(call, 0), pool.submit(call, 1)
        try:
            assert all(event.wait(5) for event in entered)
            cancelled_owner.cancel()
            assert first.result(timeout=5) == "cancelled"
            assert not second.done()
        finally:
            released.set()
        assert second.result(timeout=5) == "sibling result"
    assert not nested_owner.cancelled
    assert not sibling_owner.cancelled


def test_owned_event_loop_propagates_cancellation_scope() -> None:
    owner = RunCancellation()
    loop = OwnedAsyncLoop()
    entered, cleaned = Event(), Event()

    async def work() -> None:
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.set()

    try:
        with use_run_cancellation(owner):
            future = loop.submit(work())
        assert entered.wait(5)
        owner.cancel()
        with pytest.raises(RunCancelled):
            future.result(timeout=5)
        assert cleaned.is_set()
    finally:
        loop.close()


@pytest.mark.parametrize("closed", [False, True])
def test_cancel_distinguishes_closed_loop_race_from_scheduling_failure(monkeypatch, closed):
    loop = asyncio.new_event_loop()
    owner = RunCancellation()
    future = loop.create_future()
    owner.register(future)
    future.set_result(None)
    try:
        if closed:
            loop.close()
            owner.cancel()
        else:

            def fail(*args):
                raise RuntimeError("scheduler failed")

            monkeypatch.setattr(loop, "call_soon_threadsafe", fail)
            with pytest.raises(RuntimeError, match="scheduler failed"):
                owner.cancel()
        assert owner.cancelled
    finally:
        owner.unregister(future)
        loop.close()
