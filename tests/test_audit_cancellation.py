from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from typing import Any

import pytest

from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.tools.dispatch import SequentialDispatchGate


def test_interrupted_sync_turn_does_not_block_later_tickets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ConversationState()
    first = state.reserve_turn()
    interrupted = state.reserve_turn()

    def interrupt(*_args: Any, **_kwargs: Any) -> bool:
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(state._turn_condition, "wait", interrupt)
        with pytest.raises(KeyboardInterrupt):
            with state.adapter_turn("test", ticket=interrupted):
                pytest.fail("Interrupted turn entered")
    state.cancel_turn(first)
    try:
        assert state._next_turn == 2
        with state.adapter_turn("test") as turn:
            assert turn.turn_sequence == 2
    finally:
        state.close()


@pytest.mark.asyncio
async def test_repeated_cancellation_releases_an_acquired_but_undelivered_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ConversationState()
    acquired = threading.Event()
    release = threading.Event()
    begin = state._begin_turn

    def delayed_begin(*args: Any) -> Any:
        context = begin(*args)
        acquired.set()
        assert release.wait(2)
        return context

    monkeypatch.setattr(state, "_begin_turn", delayed_begin)

    async def enter() -> None:
        async with state.adapter_turn_async("test"):
            pytest.fail("Cancelled turn was delivered")

    pending = asyncio.create_task(enter())
    try:
        assert await asyncio.to_thread(acquired.wait, 2)
        pending.cancel()
        await asyncio.sleep(0)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done(), "Cancellation escaped before the acquired turn was released"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
        assert state._active_ticket is None, "Cancelled handoff retained its active ticket"
        monkeypatch.setattr(state, "_begin_turn", begin)
        async with state.adapter_turn_async("test") as next_turn:
            assert next_turn.turn_sequence == 1
        state.close()
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("cancellations", [1, 2])
async def test_dispatch_cleanup_releases_and_wakes_queued_tools_under_cancellation(
    sequential: bool, cancellations: int
) -> None:
    gate = SequentialDispatchGate()
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def hold() -> None:
        async with gate.enter(sequential=sequential):
            entered.set()
            await leave.wait()

    async def follow() -> str:
        async with gate.enter(sequential=True):
            return "entered"

    holder = asyncio.create_task(hold())
    await asyncio.wait_for(entered.wait(), 2)
    follower = asyncio.create_task(follow())
    await asyncio.sleep(0)
    try:
        async with gate._condition:
            leave.set()
            await asyncio.sleep(0)
            for _ in range(cancellations):
                holder.cancel()
                await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await holder
        assert await asyncio.wait_for(follower, 0.5) == "entered"
        assert gate._active_parallel == 0
        assert not gate._exclusive
    finally:
        holder.cancel()
        follower.cancel()
        await asyncio.gather(holder, follower, return_exceptions=True)
