"""Native terminal turns must stop owned async bindings before reporting completion."""

import asyncio
from contextlib import nullcontext

import pytest

from kedi.agent_adapter.subagents.control import LiveInput, live_input_scope
from kedi.agent_adapter.tools.specs import ToolSpec
from tests.test_codex_adapter import _adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("stop", ["cancel", "timeout"])
async def test_terminal_codex_turn_drains_its_tool_before_return(tmp_path, live, stop):
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def checkpoint() -> str:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            cancelled.set()
            raise
        return "obsolete"

    adapter, _ = _adapter(tmp_path, turn_timeout=0.2 if stop == "timeout" else 5)
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    task = None
    try:
        with (
            live_input_scope(control) if live else nullcontext(),
            adapter.tool_scope((ToolSpec(fn=checkpoint, name="checkpoint", risk="read_only"),)),
        ):
            task = asyncio.create_task(adapter.invoke(prompt="tool:checkpoint:{}"))
            await asyncio.wait_for(entered.wait(), 3)
            if stop == "cancel":
                task.cancel()
            with pytest.raises(asyncio.CancelledError if stop == "cancel" else TimeoutError):
                await asyncio.wait_for(task, 7)
        assert cancelled.is_set(), "Native turn returned while its tool was still running"
        assert not adapter._client._thread_tools
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()


@pytest.mark.asyncio
async def test_turn_cleanup_isolates_sibling_bindings_and_rejects_late_calls():
    from kedi.agent_adapter.adapters.codex.live import TurnBindings

    first, second = TurnBindings(), TurnBindings()
    entered, sibling_entered = asyncio.Event(), asyncio.Event()
    release, stopped = asyncio.Event(), asyncio.Event()

    async def operation(ready):
        ready.set()
        try:
            await release.wait()
        finally:
            if ready is entered:
                stopped.set()
        return 7

    async def bound(owner, ready):
        with owner.binding():
            return await owner.resolve(operation(ready))

    task = asyncio.create_task(bound(first, entered))
    sibling = asyncio.create_task(bound(second, sibling_entered))
    try:
        await entered.wait()
        await sibling_entered.wait()
        await asyncio.to_thread(first.close)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
        assert not sibling.done()
        with pytest.raises(asyncio.CancelledError), first.binding():
            pytest.fail("A closed turn must not admit a new tool")
        release.set()
        assert await sibling == 7
    finally:
        release.set()
        await asyncio.to_thread(second.close)
        await asyncio.gather(task, sibling, return_exceptions=True)


def test_unstoppable_sync_binding_reports_unconfirmed_cleanup():
    from kedi.agent_adapter.adapters.codex.live import TurnBindings
    from kedi.agent_adapter.subagents.control import UnconfirmedStop

    owner = TurnBindings()
    with owner.binding():
        with pytest.raises(UnconfirmedStop, match="bindings did not stop"):
            owner.close(timeout=0)
    owner.close(timeout=0)


@pytest.mark.asyncio
async def test_redirect_and_turn_close_cancel_binding_only_once():
    from kedi.agent_adapter.adapters.codex.live import TurnBindings

    owner = TurnBindings()
    entered, cleaning, finish, drained = (asyncio.Event() for _ in range(4))

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            drained.set()

    async def bound():
        with owner.binding():
            await owner.resolve(operation())

    task = asyncio.create_task(bound())
    closer = None
    try:
        await entered.wait()
        task.cancel()
        await cleaning.wait()
        closer = asyncio.create_task(asyncio.to_thread(owner.close))
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert not closer.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closer
        assert drained.is_set()
    finally:
        finish.set()
        await asyncio.gather(
            task, *([closer] if closer is not None else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_completed_tool_does_not_transfer_ownership_of_its_background_task():
    from kedi.agent_adapter.adapters.codex.live import TurnBindings

    owner = TurnBindings()
    release = asyncio.Event()
    background = None

    async def start_background():
        nonlocal background
        background = asyncio.create_task(release.wait())
        return "delegated"

    try:
        with owner.binding():
            assert await owner.resolve(start_background()) == "delegated"
        await asyncio.to_thread(owner.close)
        assert background is not None and not background.done()
        release.set()
        assert await background
    finally:
        release.set()
        if background is not None:
            await asyncio.gather(background, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "timeout"])
async def test_terminal_turn_drains_native_async_approval(tmp_path, stop):
    from kedi import ApprovalDecision
    from kedi.agent_adapter.approvals.policy import ApprovalPolicy
    from kedi.agent_profile import AgentProfile

    entered, drained, release = (asyncio.Event() for _ in range(3))

    async def approve(_request):
        entered.set()
        try:
            await release.wait()
        finally:
            await asyncio.sleep(0.01)
            drained.set()
        return ApprovalDecision.allow()

    adapter, _ = _adapter(tmp_path, turn_timeout=0.2 if stop == "timeout" else 5)
    task = None
    try:
        with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))):
            task = asyncio.create_task(adapter.invoke(prompt="approval:command-write"))
            await asyncio.wait_for(entered.wait(), 3)
            if stop == "cancel":
                task.cancel()
            with pytest.raises(asyncio.CancelledError if stop == "cancel" else TimeoutError):
                await asyncio.wait_for(task, 7)
        assert drained.is_set()
        assert not adapter._client._thread_approvals
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()


@pytest.mark.asyncio
async def test_repeated_owner_cancel_still_waits_for_tool_cleanup(tmp_path):
    entered, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    drained = asyncio.Event()

    async def checkpoint() -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            drained.set()

    adapter, _ = _adapter(tmp_path, turn_timeout=5)
    task = None
    try:
        with adapter.tool_scope((ToolSpec(fn=checkpoint, name="checkpoint", risk="read_only"),)):
            task = asyncio.create_task(adapter.invoke(prompt="tool:checkpoint:{}"))
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            await asyncio.wait_for(cleaning.wait(), 3)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done(), "Repeated cancel bypassed the tool cleanup barrier"
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert drained.is_set()
    finally:
        finish.set()
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()


@pytest.mark.asyncio
async def test_closed_turn_cancels_awaitable_registered_after_close():
    from kedi.agent_adapter.adapters.codex.live import TurnBindings

    owner = TurnBindings()
    owner.close()
    pending = asyncio.get_running_loop().create_future()
    with pytest.raises(asyncio.CancelledError):
        await owner.resolve(pending)
    assert pending.cancelled()
    assert not owner._tasks


def test_legacy_async_tool_binding_without_turn_lifetime():
    from contextvars import ContextVar, copy_context
    import sys
    from kedi.agent_adapter.adapters.codex.app_server import (
        CodexAppServerClient,
        _ToolBinding,
        _invoke_binding,
    )

    scope = ContextVar("legacy_binding_scope", default="outside")

    async def tool():
        await asyncio.sleep(0)
        return scope.get()

    token = scope.set("bound")
    client = CodexAppServerClient((sys.executable,))
    binding = _ToolBinding(fn=tool, context=copy_context(), await_result=client._await_sync_result)
    scope.reset(token)
    try:
        assert _invoke_binding(binding) == "bound"
        assert scope.get() == "outside"
    finally:
        client.close()


@pytest.mark.asyncio
async def test_cancelled_turn_does_not_accept_tool_that_swallows_cancellation():
    from kedi.agent_adapter.adapters.codex.live import TurnBindings

    owner = TurnBindings()
    entered = asyncio.Event()

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "stale answer"

    task = asyncio.create_task(owner.resolve(operation()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not owner._tasks
