from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Generator, Sequence
from dataclasses import replace
from threading import Event

import pytest
from pydantic_ai.messages import ToolReturn

from kedi.agent_adapter.adapters.codex.app_server import CodexTurnResult
from kedi.agent_adapter.adapters.codex.live import ensure_turn_current, run_cancellable_turn
from kedi.agent_adapter.subagents.control import (
    MAX_INPUT_BYTES,
    MAX_PENDING_INPUT_BYTES,
    AgentInput,
    InputPhase,
    LiveInput,
    RunInterrupted,
    controlled_attempt,
    current_live_input,
    current_live_revision,
    live_input_scope,
    sender_is_current,
)
from kedi.agent_adapter.subagents.tool_activity import with_live_tool_tracking
from kedi.agent_adapter.tools.specs import ToolSpec


def test_input_rejects_empty_text_before_admission() -> None:
    control = LiveInput("worker")
    control.configure(input=True)
    for text in ("", " ", "\n\t"):
        with pytest.raises(ValueError, match="nonempty"):
            control.admit(text, sender_id="parent")
    assert not control.waiting_input


def test_backend_without_interrupt_rejects_redirect_without_consuming_revision() -> None:
    control = LiveInput("worker")
    assert control.admit("context", sender_id="parent").reason == "unsupported"
    control.configure(input=True, interrupt=False)
    assert control.admit("redirect", sender_id="parent", interrupt=True).reason == "unsupported"
    assert control.revision == 0
    assert control.admit("context", sender_id="parent").status == "queued"


def test_claimed_bytes_remain_charged_until_confirmed_then_capacity_is_reusable() -> None:
    control = LiveInput("worker")
    control.configure(input=True)
    message = "x" * MAX_INPUT_BYTES
    count = MAX_PENDING_INPUT_BYTES // MAX_INPUT_BYTES
    receipts = [control.admit(message, sender_id="parent") for _ in range(count)]
    assert all(receipt.status == "queued" for receipt in receipts)
    claimed = control.take_pending()
    assert control.admit("x", sender_id="parent").reason == "queue_full"
    control.mark_delivered(claimed[:1])
    control.mark_delivered(claimed[:1])
    assert control.admit(message, sender_id="parent").status == "queued"
    assert control.admit("x", sender_id="parent").reason == "queue_full"
    assert control.delivered == (receipts[0].message_id,)


def test_input_events_report_delivery_and_undelivered_close_without_payload_loss() -> None:
    events: list[tuple[str, InputPhase]] = []

    def observe(item: AgentInput, phase: InputPhase) -> None:
        events.append((item.message, phase))

    control = LiveInput("worker", observe=observe)
    control.configure(input=True)
    control.admit("first", sender_id="parent")
    claimed = control.take_pending()
    control.restore_pending(claimed)
    assert control.pending
    control.mark_delivered(control.take_pending())
    control.admit("claimed", sender_id="parent")
    control.take_pending()
    control.admit("queued", sender_id="parent")
    control.close()
    assert events == [
        ("first", "admitted"),
        ("first", "delivered"),
        ("claimed", "admitted"),
        ("queued", "admitted"),
        ("claimed", "undelivered"),
        ("queued", "undelivered"),
    ]
    assert control.finished and not control.waiting_input


@pytest.mark.asyncio
@pytest.mark.parametrize("with_control", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_awaitable_returning_sync_tool_balances_activity_on_success_and_failure(
    with_control: bool, fail: bool
) -> None:
    control = LiveInput("worker")

    async def result() -> str:
        if fail:
            raise ValueError("tool failed")
        return "completed"

    def tool() -> Awaitable[str]:
        return result()

    tracked = with_live_tool_tracking(ToolSpec(fn=tool, name="tool"))
    with live_input_scope(control if with_control else None):
        if fail:
            with pytest.raises(ValueError, match="tool failed"):
                await tracked.fn()
        else:
            assert await tracked.fn() == "completed"
    await asyncio.wait_for(control.wait_quiescent(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_control", [False, True])
async def test_sync_tool_failure_releases_activity(with_control: bool) -> None:
    control = LiveInput("worker")

    def tool() -> str:
        raise ValueError("failed before returning")

    tracked = with_live_tool_tracking(ToolSpec(fn=tool, name="tool"))
    with live_input_scope(control if with_control else None):
        with pytest.raises(ValueError, match="failed before returning"):
            tracked.fn()
    await asyncio.wait_for(control.wait_quiescent(), 1)


@pytest.mark.asyncio
async def test_closed_input_rejects_new_tool_and_wakes_registered_waiter() -> None:
    control = LiveInput("worker")
    waiter = asyncio.create_task(control.wait_changed(control.sequence))
    await asyncio.sleep(0)
    control.close()
    await asyncio.wait_for(waiter, 1)
    with pytest.raises(RuntimeError, match="finished or superseded"):
        control.tool_started()


@pytest.mark.asyncio
async def test_obsolete_already_created_future_is_cancelled_before_attempt_start() -> None:
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    with live_input_scope(control, revision=0):
        control.admit("replacement", sender_id="parent", interrupt=True)
        with pytest.raises(RunInterrupted):
            await controlled_attempt(control, future)
    assert future.cancelled()


@pytest.mark.asyncio
async def test_wait_changed_observes_input_admitted_before_registration() -> None:
    control = LiveInput("worker")
    control.configure(input=True)
    before = control.sequence
    control.admit("context", sender_id="parent")
    await asyncio.wait_for(control.wait_changed(before), 1)


@pytest.mark.asyncio
async def test_non_quiescent_tool_fails_cleanup_without_forced_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kedi.agent_adapter.subagents.control as control_module

    monkeypatch.setattr(control_module, "INTERRUPT_CLEANUP_SECONDS", 0.01)
    control = LiveInput("worker")
    control.tool_started()
    try:
        with pytest.raises(RuntimeError, match="tool cleanup.*not restarted"):
            await control.wait_quiescent()
    finally:
        control.tool_finished()
    await control.wait_quiescent()


@pytest.mark.asyncio
async def test_native_stop_failure_cancels_attempt_without_starting_replacement() -> None:
    entered, finished = asyncio.Event(), asyncio.Event()
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)

    async def operation() -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
        return "obsolete"

    async def stop() -> None:
        raise ConnectionError("stop transport disconnected")

    task = asyncio.create_task(controlled_attempt(control, operation(), stop=stop))
    await entered.wait()
    control.admit("replacement", sender_id="parent", interrupt=True)
    with pytest.raises(ConnectionError, match="stop transport disconnected"):
        await task
    await asyncio.wait_for(finished.wait(), 1)


@pytest.mark.asyncio
async def test_terminal_confirmation_failure_is_not_reclassified_as_success() -> None:
    from kedi.agent_adapter.subagents.control import UnconfirmedStop

    async def operation() -> str:
        raise UnconfirmedStop("terminal confirmation missing")

    with pytest.raises(UnconfirmedStop, match="terminal confirmation missing"):
        await controlled_attempt(LiveInput("worker"), operation())


@pytest.mark.asyncio
async def test_individual_native_tool_does_not_wait_for_unrelated_active_tool() -> None:
    control = LiveInput("worker")
    control.tool_started()

    async def completed_tool() -> str:
        return "message sent"

    try:
        assert (
            await asyncio.wait_for(
                controlled_attempt(control, completed_tool(), quiesce_tools=False), 1
            )
            == "message sent"
        )
    finally:
        control.tool_finished()
    await asyncio.wait_for(control.wait_quiescent(), 1)


@pytest.mark.asyncio
async def test_redirect_during_native_future_resolution_cannot_publish_obsolete_tool_result() -> (
    None
):
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)

    class RedirectOnAwait(asyncio.Future[str]):
        def __await__(self) -> Generator[None, None, str]:
            control.admit("replacement scope", sender_id="parent", interrupt=True)
            return (yield from super().__await__())

    completed = RedirectOnAwait()
    completed.set_result("obsolete result")
    control.tool_started()
    try:
        with live_input_scope(control, revision=0):
            with pytest.raises(RunInterrupted):
                await asyncio.wait_for(
                    controlled_attempt(control, completed, quiesce_tools=False), 1
                )
        assert control.revision == 1 and control.waiting_input
    finally:
        control.tool_finished()
    await control.wait_quiescent()


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", ["none", "root", "stale"])
async def test_parent_tool_denies_unowned_or_superseded_sender(sender: str) -> None:
    from kedi.agent_adapter.subagents.messaging import input_tool_specs
    from kedi.core.runtime import KediRuntime

    parent = LiveInput("parent")
    parent.configure(input=True)
    child = LiveInput("child", parent=parent if sender == "stale" else None)
    child.configure(input=True, interrupt=True)
    tool = input_tool_specs(KediRuntime(), children=False, parent=True)[0]
    with live_input_scope(None if sender == "none" else child, revision=0):
        if sender == "stale":
            child.admit("redirect", sender_id="parent", interrupt=True)
        result = await tool.fn(message="Untrusted update")
    assert result["status"] == "rejected"
    assert result["reason"] == "not_authorized"
    assert not parent.waiting_input


def test_native_turn_with_cancelled_start_never_runs() -> None:
    stop = Event()
    stop.set()
    with pytest.raises(asyncio.CancelledError):
        ensure_turn_current(None, 0, stop)


@pytest.mark.asyncio
@pytest.mark.parametrize("native_cancel", [False, True])
async def test_codex_owner_cancellation_drains_native_thread_before_return(
    native_cancel: bool,
) -> None:
    entered, finished = Event(), Event()

    def turn(stop: Event) -> CodexTurnResult:
        entered.set()
        assert stop.wait(2)
        finished.set()
        if native_cancel:
            raise asyncio.CancelledError
        return CodexTurnResult(text="partial", turn={"status": "interrupted"})

    task = asyncio.create_task(run_cancellable_turn(turn))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_history_processor_rejection_keeps_input_undelivered() -> None:
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
    from kedi.agent_adapter.history_processing import HistoryProcessorContext
    from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
    from kedi.agent_adapter.subagents.runs import SubagentInputEvent
    from kedi.errors import KediExecutionError
    from kedi.lang import compile_program, parse_program
    from tests.test_native_subagent_tasks import PROFILES

    entered, release = asyncio.Event(), asyncio.Event()
    requests: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        if len(requests) == 1:
            entered.set()
            await release.wait()
        return ModelResponse(parts=[TextPart("Completed without the extra context.")])

    def remove_input(ctx: HistoryProcessorContext[ModelMessage]) -> Sequence[ModelMessage]:
        return [
            replace(
                message,
                parts=[
                    part
                    for part in message.parts
                    if not (
                        isinstance(part, UserPromptPart)
                        and isinstance(part.content, str)
                        and "Kedi agent input:" in part.content
                    )
                ],
            )
            if isinstance(message, ModelRequest)
            else message
            for message in ctx.messages
        ]

    with pytest.warns(UserWarning, match="prefix-cache"):
        adapter = PydanticAdapter(FunctionModel(respond), history_processor=remove_input)
    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"),
        adapter=adapter,
        subagent_usage_limits=SubagentUsageLimits(request_limit=2),
    )
    with pytest.raises(KediExecutionError, match="preserve the current request tail"):
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review the change.")
            await asyncio.wait_for(entered.wait(), 1)
            receipt = await job.send("Include the migration evidence.")
            release.set()
            await asyncio.wait_for(job.wait(), 2)
    assert len(requests) == 1
    assert runtime._subagent_coordinator.runs.get(job.run_id).status == "failed"
    phases = [
        event.phase
        for event in runtime._subagent_coordinator.events.snapshot()
        if isinstance(event, SubagentInputEvent) and event.message_id == receipt.message_id
    ]
    assert phases == ["admitted", "undelivered"]


@pytest.mark.asyncio
async def test_pydantic_interrupt_retains_rich_tool_content_and_dynamic_availability() -> None:
    from pydantic_ai import Agent, CancellationToken
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolAvailabilityDeltaPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.run import AgentRun
    from pydantic_ai.tools import Tool
    from pydantic_ai.usage import RunUsage

    from kedi.agent_adapter.adapters.pydantic.live import (
        InputDeliveryCapability,
        run_with_input,
    )

    completed, started = asyncio.Event(), asyncio.Event()
    effects: list[str] = []
    history: list[ModelMessage] = []

    async def fast() -> ToolReturn:
        effects.append("committed")
        completed.set()
        return ToolReturn(
            "Completed host effect",
            content="Evidence retained for the next request.",
            metadata={"receipt": "committed"},
            tools=["follow_up"],
        )

    async def slow() -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    def follow_up() -> str:
        return "Follow-up evidence"

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart("fast", {}, "fast"), ToolCallPart("slow", {}, "slow")]
        )

    def capture(run: AgentRun[None, str]) -> None:
        history.extend(run.all_messages())

    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    capability: InputDeliveryCapability[None] = InputDeliveryCapability(control=control)
    token, usage = CancellationToken(), RunUsage()
    agent: Agent[None, str] = Agent(
        FunctionModel(respond), tools=[fast, slow, Tool(follow_up, defer_loading=True)]
    )
    task = asyncio.create_task(
        run_with_input(
            agent.iter("Run the tools.", capabilities=[capability], cancellation_token=token),
            control=control,
            capability=capability,
            capture=capture,
            usage=usage,
        )
    )
    await asyncio.wait_for(completed.wait(), 1)
    await asyncio.wait_for(started.wait(), 1)
    control.admit("Use the retained evidence.", sender_id="parent", interrupt=True)
    with pytest.raises(RunInterrupted):
        await asyncio.wait_for(task, 2)
    parts = [
        part for message in history if isinstance(message, ModelRequest) for part in message.parts
    ]
    assert effects == ["committed"]
    assert usage.requests == 1
    assert any(
        isinstance(part, ToolReturnPart)
        and part.tool_call_id == "fast"
        and part.content == "Completed host effect"
        and part.metadata == {"receipt": "committed"}
        for part in parts
    )
    assert any(
        isinstance(part, UserPromptPart)
        and part.content == "Evidence retained for the next request."
        for part in parts
    )
    assert any(
        isinstance(part, ToolAvailabilityDeltaPart) and part.tools_added == ["follow_up"]
        for part in parts
    )
    assert any(
        isinstance(part, ToolReturnPart)
        and part.tool_call_id == "slow"
        and part.outcome == "failed"
        for part in parts
    )


@pytest.mark.asyncio
async def test_parent_delivery_failure_does_not_silently_complete_the_attempt() -> None:
    entered, stopped = asyncio.Event(), asyncio.Event()
    control = LiveInput("worker")
    control.configure(input=True)

    async def operation() -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return "unreachable"

    async def deliver(items: tuple[AgentInput, ...]) -> bool:
        assert items[0].message == "New evidence"
        raise ConnectionError("input transport disconnected")

    task = asyncio.create_task(controlled_attempt(control, operation(), deliver=deliver))
    await entered.wait()
    control.admit("New evidence", sender_id="parent")
    with pytest.raises(ConnectionError, match="input transport disconnected"):
        await task
    await asyncio.wait_for(stopped.wait(), 1)
    assert not control.delivered
    assert control.waiting_input


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
async def test_input_after_native_response_prevents_obsolete_final_publication(
    interrupt: bool,
) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
    from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
    from kedi.agent_adapter.subagents.control import InputReceipt
    from kedi.agent_adapter.subagents.runs import SubagentCompleted, SubagentInputEvent
    from kedi.lang import compile_program, parse_program
    from tests.test_native_subagent_tasks import PROFILES

    requests: list[list[ModelMessage]] = []
    receipts: list[InputReceipt] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        return ModelResponse(parts=[TextPart("obsolete" if len(requests) == 1 else "updated")])

    def update_at_response_boundary(usage: InvocationUsage) -> None:
        if receipts:
            return
        control = current_live_input()
        assert control is not None and control.parent is not None
        receipts.append(
            control.admit(
                "Review the updated scope.", sender_id=control.parent.target_id, interrupt=interrupt
            )
        )

    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"), adapter=PydanticAdapter(FunctionModel(respond))
    )
    with observe_invocations(update_at_response_boundary):
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review the original scope.")
            result = await asyncio.wait_for(job.wait(), 2)
            assert await job.wait() is result
    assert result.run_id == job.run_id
    assert result.task_summary == "updated"
    assert len(requests) == 2
    assert "Review the updated scope." in str(requests[1])
    assert "obsolete" in str(requests[1])
    run = runtime._subagent_coordinator.runs.get(job.run_id)
    assert run.status == "completed" and run.requests == 2
    assert receipts[0].status == ("interrupt_requested" if interrupt else "queued")
    events = runtime._subagent_coordinator.events.snapshot()
    assert sum(isinstance(event, SubagentCompleted) for event in events) == 1
    assert [
        event.phase
        for event in events
        if isinstance(event, SubagentInputEvent) and event.message_id == receipts[0].message_id
    ] == ["admitted", "delivered"]


def test_reentering_live_scope_does_not_refresh_an_obsolete_attempt_revision() -> None:
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    other = LiveInput("other")
    with live_input_scope(control, revision=0):
        control.admit("Updated scope", sender_id="parent", interrupt=True)
        with live_input_scope(control):
            assert current_live_revision(control) == 0
            assert not sender_is_current(control)
            with pytest.raises(RuntimeError, match="superseded"):
                control.tool_started(revision=current_live_revision(control))
        with live_input_scope(other):
            assert current_live_revision(other) == 0
        assert current_live_revision(control) == 0
        with live_input_scope(control, revision=control.begin_attempt()):
            assert current_live_revision(control) == 1
            assert sender_is_current(control)


@pytest.mark.asyncio
async def test_superseded_parent_cannot_spawn_child_through_native_owner() -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
    from kedi.errors import KediExecutionError
    from kedi.lang import compile_program, parse_program
    from tests.test_native_subagent_tasks import PROFILES

    requests: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        return ModelResponse(parts=[TextPart("Unexpected inference")])

    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"), adapter=PydanticAdapter(FunctionModel(respond))
    )
    control = LiveInput("parent")
    control.configure(input=True, interrupt=True)
    async with runtime.subagents(parent="manager") as agents:
        with live_input_scope(control, revision=0):
            control.admit("Updated scope", sender_id="owner", interrupt=True)
            with pytest.raises(KediExecutionError, match="superseded agent attempt"):
                await agents.start("worker", task="Obsolete child task")
    assert requests == []
    assert runtime._subagent_coordinator.runs.snapshot() == ()


@pytest.mark.asyncio
async def test_python_owner_inbox_wakes_on_close_without_implicit_inference() -> None:
    from kedi.errors import KediExecutionError
    from kedi.lang import compile_program, parse_program
    from tests.test_native_subagent_tasks import PROFILES

    runtime = compile_program(parse_program(PROFILES + "= ready\n"))
    scope = runtime.subagents(parent="manager")
    with pytest.raises(KediExecutionError, match="scope is not active"):
        await scope.receive_messages()
    async with scope as agents:
        pending = asyncio.create_task(agents.receive_messages())
        await asyncio.sleep(0)
        assert not pending.done()
    assert await asyncio.wait_for(pending, 1) == ()
    with pytest.raises(KediExecutionError, match="scope is not active"):
        await scope.receive_messages()


@pytest.mark.asyncio
async def test_native_enqueue_rejection_fails_once_without_busy_retry_or_false_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.run import AgentRun

    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
    from kedi.agent_adapter.subagents.runs import SubagentInputEvent
    from kedi.errors import KediExecutionError
    from kedi.lang import compile_program, parse_program
    from tests.test_native_subagent_tasks import PROFILES

    entered, stopped = asyncio.Event(), asyncio.Event()
    attempts: list[tuple[str, ...]] = []

    def reject_enqueue(run: AgentRun[None, str], *content: str) -> str | None:
        attempts.append(content)
        if len(attempts) > 1:
            raise AssertionError("Repeated an invalid native enqueue")
        raise ValueError("Native input queue is closed")

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return ModelResponse(parts=[TextPart("unreachable")])

    monkeypatch.setattr(AgentRun, "enqueue", reject_enqueue)
    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"), adapter=PydanticAdapter(FunctionModel(respond))
    )
    with pytest.raises(KediExecutionError, match="Native input queue is closed"):
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review the change.")
            await asyncio.wait_for(entered.wait(), 1)
            receipt = await job.send("New evidence")
            await asyncio.wait_for(job.wait(), 2)
    await asyncio.wait_for(stopped.wait(), 1)
    assert len(attempts) == 1
    assert runtime._subagent_coordinator.runs.get(job.run_id).status == "failed"
    assert [
        event.phase
        for event in runtime._subagent_coordinator.events.snapshot()
        if isinstance(event, SubagentInputEvent) and event.message_id == receipt.message_id
    ] == ["admitted", "undelivered"]


@pytest.mark.parametrize(
    ("body", "error", "requests_expected"),
    [
        (
            "[job: int] = `1`\n> send: job:\n    >> More evidence.\n",
            "expects a subagent task",
            0,
        ),
        (
            "> task [job]: worker:\n    >> Review the change.\n"
            "> await: job\n> send: job:\n    >> Too late.\n",
            "Agent input rejected: task_finished",
            1,
        ),
    ],
    ids=["not-a-task", "completed-task"],
)
def test_send_dsl_rejects_invalid_or_finished_task_without_extra_inference(
    body: str, error: str, requests_expected: int
) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
    from kedi.errors import KediExecutionError
    from kedi.lang import compile_program, parse_program
    from tests.test_native_subagent_tasks import PROFILES

    requests: list[list[ModelMessage]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        return ModelResponse(parts=[TextPart("Completed review.")])

    runtime = compile_program(
        parse_program(PROFILES + body), adapter=PydanticAdapter(FunctionModel(respond))
    )
    with pytest.raises(KediExecutionError, match=error):
        runtime.run_main()
    assert len(requests) == requests_expected
