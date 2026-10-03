from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Event
from types import TracebackType
from typing import TYPE_CHECKING

import anyio
import pytest

from kedi.agent_adapter.adapters.claude.live import run_query
from kedi.agent_adapter.adapters.codex.app_server import CodexAppServerClient
from kedi.agent_adapter.subagents.control import (
    LiveInput,
    RunInterrupted,
    UnconfirmedStop,
    live_input_scope,
)

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, Message


@pytest.mark.asyncio
@pytest.mark.parametrize("during_startup", [False, True])
async def test_codex_redirect_before_turn_start_never_starts_obsolete_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_startup: bool
) -> None:
    from tests.test_codex_adapter import _adapter, _messages, _method

    started, release = Event(), Event()
    start_thread = CodexAppServerClient._start_thread

    def held_start(client, parameters):
        response = start_thread(client, parameters)
        started.set()
        assert release.wait(2)
        return response

    monkeypatch.setattr(CodexAppServerClient, "_start_thread", held_start)
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    adapter, log = _adapter(tmp_path)
    try:
        with live_input_scope(control, revision=0):
            if not during_startup:
                control.admit("updated scope", sender_id="parent", interrupt=True)
            task = asyncio.create_task(adapter.invoke(prompt="obsolete task"))
            if during_startup:
                assert await asyncio.to_thread(started.wait, 2)
                control.admit("updated scope", sender_id="parent", interrupt=True)
            release.set()
            with pytest.raises(RunInterrupted):
                await asyncio.wait_for(task, 3)
        messages = _messages(log) if log.exists() else []
        assert not _method(messages, "turn/start")
        assert len(_method(messages, "thread/start")) == int(during_startup)
    finally:
        release.set()
        await adapter.aclose()


@pytest.mark.asyncio
async def test_codex_interrupt_ack_without_terminal_event_cannot_restart_the_task(
    tmp_path: Path,
) -> None:
    from tests.test_codex_adapter import _adapter, _messages, _method

    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    adapter, log = _adapter(tmp_path)
    with live_input_scope(control):
        task = asyncio.create_task(adapter.invoke(prompt="unconfirmed-interrupt"))
    try:
        with live_input_scope(control):
            with anyio.fail_after(3):
                while not log.exists() or not _method(_messages(log), "turn/start"):
                    await asyncio.sleep(0.01)
            receipt = control.admit("Updated scope", sender_id="parent", interrupt=True)
            with pytest.raises(TimeoutError, match="interrupt.*terminal completion"):
                await asyncio.wait_for(task, 7)
        messages = _messages(log)
        assert len(_method(messages, "turn/start")) == 1
        assert len(_method(messages, "turn/interrupt")) == 1
        assert receipt.message_id not in control.delivered
        assert control.waiting_input
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()


@pytest.mark.asyncio
async def test_codex_redirect_drains_native_async_tool_before_continuation(tmp_path: Path) -> None:
    from kedi.agent_adapter.subagents.tool_activity import with_live_tool_tracking
    from kedi.agent_adapter.tools.specs import ToolSpec
    from tests.test_codex_adapter import _adapter, _messages, _method

    started, cancelled = asyncio.Event(), asyncio.Event()

    async def checkpoint() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "obsolete"

    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    adapter, log = _adapter(tmp_path)
    task: asyncio.Task[str] | None = None
    try:
        tool = with_live_tool_tracking(ToolSpec(fn=checkpoint, name="checkpoint", risk="read_only"))
        with live_input_scope(control), adapter.tool_scope((tool,)):
            task = asyncio.create_task(adapter.invoke(prompt="tool:checkpoint:{}"))
            await asyncio.wait_for(started.wait(), 3)
            receipt = control.admit("Updated scope", sender_id="parent", interrupt=True)
            with pytest.raises(RunInterrupted):
                await asyncio.wait_for(task, 7)
        assert cancelled.is_set()
        await control.wait_quiescent()
        assert receipt.message_id not in control.delivered
        with live_input_scope(control, revision=control.begin_attempt()):
            assert "updated" in await adapter.invoke(prompt="updated")
        turns = _method(_messages(log), "turn/start")
        assert len(turns) == 2
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()


@pytest.mark.asyncio
async def test_codex_redirect_between_pending_check_and_claim_never_steers_an_empty_input(
    tmp_path: Path,
) -> None:
    from tests.test_codex_adapter import _adapter, _messages, _method

    paused, release = Event(), Event()

    class PausingInbox(LiveInput):
        @property
        def pending(self) -> bool:
            has_input = super().pending
            if has_input and not paused.is_set():
                paused.set()
                assert release.wait(3)
            return has_input

    control = PausingInbox("worker")
    control.configure(input=True, interrupt=True)
    adapter, log = _adapter(tmp_path)
    with live_input_scope(control):
        task = asyncio.create_task(adapter.invoke(prompt="live-input"))
    try:
        with anyio.fail_after(3):
            while not log.exists() or not _method(_messages(log), "turn/start"):
                await asyncio.sleep(0.01)
        first = control.admit("Additional evidence", sender_id="parent")
        assert await asyncio.to_thread(paused.wait, 3)
        redirect = control.admit("Updated scope", sender_id="parent", interrupt=True)
        release.set()
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(task, 3)
        messages = _messages(log)
        assert not _method(messages, "turn/steer")
        assert len(_method(messages, "turn/interrupt")) == 1
        assert len(_method(messages, "turn/start")) == 1
        assert first.message_id not in control.delivered
        assert redirect.message_id not in control.delivered
        assert len(control.take_pending()) == 2
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()


def test_codex_parent_late_input_reuses_then_releases_its_ephemeral_native_thread(
    tmp_path: Path,
) -> None:
    from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
    from kedi.agent_adapter.subagents.control import InputReceipt, current_live_input
    from kedi.lang import compile_program, parse_program
    from tests.test_codex_adapter import _adapter, _messages, _method

    receipts: list[InputReceipt] = []

    def add_late_input(usage: InvocationUsage) -> None:
        del usage
        if not receipts:
            control = current_live_input()
            assert control is not None and control.parent is None
            receipts.append(control.admit("Updated scope", sender_id="worker"))

    source = """
> profile: worker:
    > agent: codex
> profile: manager:
    > agent: codex
    > subagent: worker
> use: manager
[answer] << Review the original scope.
= <answer>
"""
    adapter, log = _adapter(tmp_path)
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        with observe_invocations(add_late_input):
            assert "Updated scope" in runtime.run_main()
        messages = _messages(log)
        turns = _method(messages, "turn/start")
        assert len(turns) == 2
        assert turns[0]["params"]["threadId"] == turns[1]["params"]["threadId"]
        assert len(_method(messages, "thread/delete")) == 1
        assert receipts[0].status == "queued"
    finally:
        adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("terminal", ["completed", "failed", "running"])
async def test_codex_control_racing_with_completion_preserves_undelivered_input(
    tmp_path: Path, interrupt: bool, terminal: str
) -> None:
    from kedi.agent_adapter.adapters.codex.app_server import CodexAppServerProtocolError
    from kedi.agent_adapter.conversation import AdapterConversationContext
    from tests.test_codex_adapter import _adapter, _messages, _method

    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    conversation = AdapterConversationContext(session_id="worker", cache_key="worker")
    adapter, log = _adapter(
        tmp_path, extra_env={"KEDI_CODEX_FAKE_CONTROL_RACE": terminal}, turn_timeout=3
    )
    with live_input_scope(control), adapter.conversation_scope(conversation):
        task = asyncio.create_task(adapter.invoke(prompt="live-input"))
    try:
        with anyio.fail_after(3):
            while not log.exists() or not _method(_messages(log), "turn/start"):
                await asyncio.sleep(0.01)
        receipt = control.admit("Updated scope", sender_id="parent", interrupt=interrupt)
        if terminal != "completed":
            with pytest.raises(CodexAppServerProtocolError, match="No active turn"):
                await asyncio.wait_for(task, 3)
        else:
            if interrupt:
                with pytest.raises(RunInterrupted):
                    await asyncio.wait_for(task, 3)
            else:
                assert await asyncio.wait_for(task, 3) == "Original scope completed"
            assert receipt.message_id not in control.delivered
            assert control.pending
            assert not control.try_finish(control.revision, lambda: None)
            conversation.resume_state.update(conversation.result_state)
            with (
                live_input_scope(control, revision=control.begin_attempt()),
                adapter.conversation_scope(conversation),
            ):
                items = control.take_pending()
                assert len(items) == 1
                assert "Updated scope" in await adapter.invoke(prompt=items[0].render())
            assert control.delivered == (receipt.message_id,)
            assert control.try_finish(control.revision, lambda: None)
            turns = _method(_messages(log), "turn/start")
            assert len(turns) == 2
            assert turns[0]["params"]["threadId"] == turns[1]["params"]["threadId"]
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("during_query", [False, True])
async def test_claude_redirect_requires_terminal_confirmation(
    monkeypatch: pytest.MonkeyPatch, confirmed: bool, during_query: bool
) -> None:
    pytest.importorskip("claude_agent_sdk")
    import claude_agent_sdk
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage

    entered, interrupted, closed = (asyncio.Event() for _ in range(3))
    observed: list[Message] = []
    queries: list[str] = []
    interrupts: list[bool] = []

    class Client(ClaudeSDKClient):
        def __init__(self, *, options: ClaudeAgentOptions) -> None:
            self.options = options

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            closed.set()

        async def query(self, prompt: str, session_id: str = "default") -> None:
            queries.append(prompt)
            if during_query:
                entered.set()
                await interrupted.wait()

        async def interrupt(self) -> None:
            interrupts.append(True)
            interrupted.set()

        async def receive_response(self) -> AsyncIterator[Message]:
            entered.set()
            await interrupted.wait()
            if confirmed:
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=1,
                    session_id="retained-session",
                    result="partial work",
                )

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    task = asyncio.create_task(run_query(ClaudeAgentOptions(), "task", control, observed.append))
    await asyncio.wait_for(entered.wait(), 1)
    control.admit("new scope", sender_id="parent", interrupt=True)
    exception = RunInterrupted if confirmed else UnconfirmedStop
    with pytest.raises(exception):
        await asyncio.wait_for(task, 1)
    assert queries == ["task"]
    assert interrupts == [True]
    assert closed.is_set()
    if confirmed:
        assert len(observed) == 1
        assert isinstance(observed[0], ResultMessage)
        assert observed[0].session_id == "retained-session"
    else:
        assert observed == []
