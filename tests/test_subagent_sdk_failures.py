from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Generator, Sequence
from contextlib import asynccontextmanager, nullcontext
from typing import NoReturn

import pytest
from langchain.agents.middleware.types import AgentState, ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic_ai import Agent, CancellationToken, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError, RunCancelled, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelResponse
from pydantic_ai.run import AgentRun
from pydantic_ai.usage import RunUsage

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.adapters.langchain.live import run_with_input as run_graph
from kedi.agent_adapter.adapters.langchain.runtime import (
    LangChainRunMiddleware,
    langchain_run_scope,
)
from kedi.agent_adapter.adapters.pydantic.live import InputDeliveryCapability, run_with_input
from kedi.agent_adapter.subagents import SubagentConversationContext
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
from kedi.agent_adapter.subagents.control import (
    LiveInput,
    RunInterrupted,
    UnconfirmedStop,
    controlled_attempt,
    live_input_scope,
)
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope
from tests.test_langchain_branch_parity import ScriptedModel
from tests.test_subagent_parent_messaging import Reply, responding_adapter, tool_returns


def test_live_input_capability_has_no_persistable_serialization_name() -> None:
    # An invocation-owned controller cannot be reconstructed from an agent spec.
    assert InputDeliveryCapability.get_serialization_name() is None


def test_child_cache_key_is_owned_by_conversation_not_attempt() -> None:
    conversation = SubagentConversationContext(conversation_id="review-conversation")
    assert conversation.cache_key == "kedi-subagent-review-conversation"
    conversation.result_state["session"] = "native-session"
    conversation.resume_state.update(conversation.result_state)
    assert conversation.cache_key == "kedi-subagent-review-conversation"
    assert conversation.cache_key != SubagentConversationContext(conversation_id="other").cache_key


@pytest.mark.asyncio
@pytest.mark.parametrize("before_start", [False, True])
async def test_live_pydantic_run_preserves_caller_cancellation_token(before_start: bool) -> None:
    started, stopped = asyncio.Event(), asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        raise AssertionError("Cancelled request returned")

    adapter = PydanticAdapter(FunctionModel(respond))
    token = CancellationToken()
    control = LiveInput("worker")
    if before_start:
        token.cancel()
    with live_input_scope(control):
        task = asyncio.create_task(adapter.invoke(prompt="Review.", cancellation_token=token))
    try:
        if not before_start:
            await asyncio.wait_for(started.wait(), 2)
            token.cancel()
        with pytest.raises(RunCancelled):
            await asyncio.wait_for(task, 2)
        assert started.is_set() is not before_start
        assert stopped.is_set() is not before_start
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_pydantic_redirect_does_not_cancel_a_siblings_shared_token() -> None:
    started = [asyncio.Event(), asyncio.Event()]
    stopped = [asyncio.Event(), asyncio.Event()]
    token = CancellationToken()

    def adapter_for(index: int) -> PydanticAdapter[None, str]:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            started[index].set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped[index].set()
            raise AssertionError("Cancelled request returned")

        return PydanticAdapter(FunctionModel(respond))

    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    with live_input_scope(control):
        worker = asyncio.create_task(
            adapter_for(0).invoke(prompt="Review.", cancellation_token=token)
        )
    sibling = asyncio.create_task(adapter_for(1).invoke(prompt="Review.", cancellation_token=token))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
        control.admit("Changed scope", sender_id="parent", interrupt=True)
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(worker, 2)
        assert stopped[0].is_set()
        assert not token.cancelled
        assert not stopped[1].is_set()
        assert not sibling.done()
        token.cancel()
        with pytest.raises(RunCancelled):
            await asyncio.wait_for(sibling, 2)
    finally:
        worker.cancel()
        sibling.cancel()
        await asyncio.gather(worker, sibling, return_exceptions=True)


@pytest.mark.asyncio
async def test_pydantic_redirect_cancels_initialization_before_run_is_available() -> None:
    started, stopped = asyncio.Event(), asyncio.Event()
    agent: Agent[None, str] = Agent(TestModel())

    @asynccontextmanager
    async def delayed_iterator() -> AsyncIterator[AgentRun[None, str]]:
        started.set()
        try:
            await asyncio.Event().wait()
            async with agent.iter("Review.") as run:
                yield run
        finally:
            stopped.set()

    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    snapshots: list[AgentRun[None, str]] = []
    task = asyncio.create_task(
        run_with_input(
            delayed_iterator(),
            control=control,
            capability=InputDeliveryCapability(control=control),
            capture=snapshots.append,
            usage=None,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        control.admit("Updated scope", sender_id="parent", interrupt=True)
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(task, 2)
        assert stopped.is_set()
        assert snapshots == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_invalid_pydantic_capabilities_fail_before_starting_a_native_request() -> None:
    adapter = PydanticAdapter(TestModel())
    control = LiveInput("worker")
    control.configure(input=True)
    usage = RunUsage()
    with live_input_scope(control), pytest.raises(TypeError, match="capabilities.*list or tuple"):
        await adapter.invoke(prompt="Review.", capabilities="invalid", usage=usage)
    assert usage.requests == 0
    assert not control.delivered


@pytest.mark.asyncio
async def test_langchain_pending_input_does_not_skip_an_already_requested_tool() -> None:
    control = LiveInput("worker")
    control.configure(input=True)
    control.admit("Additional evidence.", sender_id="parent")
    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)
    state: AgentState = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "read-1"}])
        ]
    }
    with live_input_scope(control):
        assert await middleware.aafter_model(state, Runtime()) is None
    assert control.pending


@pytest.mark.asyncio
async def test_langchain_graph_handoff_command_is_not_journaled_as_a_tool_result() -> None:
    command: Command[str] = Command(
        graph=Command.PARENT, goto="coordinator", update={"handoff": True}
    )

    async def handler(request: ToolCallRequest) -> Command[str]:
        assert request.tool_call["id"] == "handoff-1"
        return command

    request = ToolCallRequest(
        tool_call={"name": "handoff", "args": {}, "id": "handoff-1"},
        tool=None,
        state={"messages": []},
        runtime=ToolRuntime(
            state={"messages": []},
            context=None,
            config={},
            stream_writer=lambda event: None,
            tool_call_id="handoff-1",
            store=None,
        ),
    )
    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)
    with langchain_run_scope(None) as state:
        assert await middleware.awrap_tool_call(request, handler) is command
        assert state.native_usage.tool_calls == 1
        assert state.completed_tools == {}


@pytest.mark.asyncio
async def test_redirect_before_monitor_registration_cancels_the_obsolete_attempt() -> None:
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    stopped = asyncio.Event()

    async def operation() -> str:
        control.admit("Updated scope", sender_id="parent", interrupt=True)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return "unreachable"

    with pytest.raises(RunInterrupted):
        await controlled_attempt(control, operation())
    assert stopped.is_set()
    assert control.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["failure", "redirect"])
async def test_descendant_cleanup_timeout_keeps_original_failure_without_replacement(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    import kedi.agent_adapter.subagents.communication as communication
    import kedi.agent_adapter.subagents.control as control_module

    monkeypatch.setattr(communication, "INTERRUPT_CLEANUP_SECONDS", 0.02)
    monkeypatch.setattr(control_module, "INTERRUPT_CLEANUP_SECONDS", 0.2)
    leaf_started, worker_started, release, stopped = (asyncio.Event() for _ in range(4))
    requests: list[str] = []

    async def respond(messages: Sequence[BaseMessage | ModelMessage]) -> Reply:
        text = str(messages)
        requests.append(text)
        if "Parent review" in text:
            returned = tool_returns(messages)
            if "delegate_task" not in returned:
                return "delegate_task", {
                    "subagent": "leaf",
                    "task": "Leaf review",
                    "background": True,
                }
            await leaf_started.wait()
            worker_started.set()
            if reason == "failure":
                raise ValueError("Parent model failed")
            await asyncio.Event().wait()
            return None, "obsolete"
        leaf_started.set()
        try:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        finally:
            stopped.set()
        return None, "late leaf result"

    source = """
> profile: leaf:
    > adapter: pydantic
> profile: worker:
    > adapter: pydantic
    > subagent: leaf
> profile: manager:
    > adapter: pydantic
    > subagent: worker
> use: manager
= ready
"""
    runtime = compile_program(
        parse_program(source), adapter=responding_adapter("pydantic", respond)
    )
    expected = "Parent model failed" if reason == "failure" else "descendants did not stop"
    error_type = KediExecutionError if reason == "failure" else RuntimeError
    try:
        with pytest.raises(error_type, match=expected) as failure:
            async with runtime.subagents(parent="manager") as agents:
                job = await agents.start("worker", task="Parent review")
                await asyncio.wait_for(worker_started.wait(), 2)
                if reason == "redirect":
                    assert (
                        await job.send("Updated scope", interrupt=True)
                    ).status == "interrupt_requested"
                try:
                    await asyncio.wait_for(job.wait(), 2)
                finally:
                    release.set()
                    await asyncio.wait_for(stopped.wait(), 2)
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert "descendants did not stop" in str(failure.value.__cause__)
        assert len(requests) == 3
        assert runtime._subagent_coordinator.runs.get(job.run_id).status == "failed"
    finally:
        release.set()


@pytest.mark.asyncio
async def test_descendant_cleanup_accepts_a_completed_child_whose_control_was_pruned() -> None:
    requests: list[str] = []

    async def respond(messages: Sequence[BaseMessage | ModelMessage]) -> Reply:
        text = str(messages)
        requests.append(text)
        if "Parent review" not in text:
            return None, "Leaf complete"
        returned = tool_returns(messages)
        if "delegate_task" not in returned:
            return "delegate_task", {"subagent": "leaf", "task": "Leaf review", "background": True}
        if "wait_subagent" not in returned:
            return "wait_subagent", {"run_id": returned["delegate_task"]["run_id"]}
        return None, "Parent complete"

    source = """
> profile: leaf:
    > adapter: pydantic
> profile: worker:
    > adapter: pydantic
    > subagent: leaf
> profile: manager:
    > adapter: pydantic
    > subagent: worker
> use: manager
= ready
"""
    runtime = compile_program(
        parse_program(source), adapter=responding_adapter("pydantic", respond)
    )
    async with runtime.subagents(parent="manager") as agents:
        job = await agents.start("worker", task="Parent review")
        assert (await job.wait()).task_summary == "Parent complete"
        coordinator = runtime._subagent_coordinator
        leaf = next(run for run in coordinator.runs.snapshot() if run.profile_name == "leaf")
        assert leaf.parent_run_id == job.run_id and leaf.status == "completed"
        coordinator._controls.pop(leaf.run_id)
        await coordinator.stop_descendants(job.run_id)
        assert len(requests) == 4


@pytest.mark.asyncio
async def test_closed_embedded_loop_does_not_prevent_notifying_a_live_waiter() -> None:
    control = LiveInput("worker")
    control.configure(input=True)
    loop = asyncio.new_event_loop()
    abandoned = loop.create_future()
    # Model an embedding loop torn down before its waiter can unregister.
    with control._lock:
        control._waiters[abandoned] = loop
    loop.close()
    waiter = asyncio.create_task(control.wait_changed(control.sequence))
    await asyncio.sleep(0)
    try:
        control.admit("New evidence", sender_id="parent")
        await asyncio.wait_for(waiter, 1)
        assert control.pending
    finally:
        with control._lock:
            control._waiters.pop(abandoned, None)
        abandoned.cancel()
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("budget_kind", ["native", "kedi"])
async def test_native_request_budget_rejection_does_not_confirm_input_delivery(
    backend: str,
    budget_kind: str,
) -> None:
    model = ScriptedModel(responses=[AIMessage(content="Unexpected request")])
    adapter = PydanticAdapter(TestModel()) if backend == "pydantic" else LangChainAdapter(model)
    control = LiveInput("worker")
    control.configure(input=True)
    receipt = control.admit("Additional evidence.", sender_id="parent")
    claimed = control.take_pending()
    usage = RunUsage()
    observed: list[InvocationUsage] = []
    budget = (
        run_budget_scope(RunBudget(request_limit=0)) if budget_kind == "kedi" else nullcontext()
    )
    error_type = RunBudgetExceeded if budget_kind == "kedi" else UsageLimitExceeded
    with (
        live_input_scope(control),
        budget,
        observe_invocations(observed.append),
        pytest.raises(error_type),
    ):
        await adapter.invoke(
            prompt=claimed[0].render(),
            usage=usage,
            usage_limits=UsageLimits(request_limit=0 if budget_kind == "native" else None),
        )
    assert usage.requests == 0 and model.seen == []
    assert sum(item.requests for item in observed) == 0
    assert receipt.message_id not in control.delivered
    assert control.waiting_input


@pytest.mark.asyncio
@pytest.mark.parametrize("request_limit", [1, 2])
@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("previous_requests", [0, 7])
async def test_pydantic_budget_admission_counts_fallback_requests_not_rejected_attempts(
    request_limit: int,
    observe: bool,
    previous_requests: int,
) -> None:
    requests: list[list[ModelMessage]] = []

    async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        raise ModelHTTPError(status_code=500, model_name="primary")

    adapter = PydanticAdapter(
        FallbackModel(FunctionModel(fail), TestModel(custom_output_text="fallback complete"))
    )
    control = LiveInput("worker")
    control.configure(input=True)
    receipt = control.admit("Additional evidence.", sender_id="parent")
    prompt = control.take_pending()[0].render()
    usage = RunUsage(requests=previous_requests)
    observed: list[InvocationUsage] = []
    observation = observe_invocations(observed.append) if observe else nullcontext()
    with (
        live_input_scope(control),
        run_budget_scope(RunBudget(request_limit=request_limit)) as ledger,
        observation,
    ):
        assert ledger is not None
        if request_limit == 1:
            with pytest.raises(RunBudgetExceeded):
                await adapter.invoke(prompt=prompt, usage=usage)
        else:
            assert await adapter.invoke(prompt=prompt, usage=usage) == "fallback complete"
        assert ledger.spent("request") == request_limit
    assert len(requests) == 1
    assert usage.requests == previous_requests + request_limit
    assert sum(item.requests for item in observed) == (
        previous_requests + request_limit if observe else 0
    )
    assert receipt.message_id in control.delivered


@pytest.mark.asyncio
async def test_superseded_custom_awaitable_is_not_entered_or_silently_started() -> None:
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    entered = asyncio.Event()

    class DeferredValue(Awaitable[str]):
        def __await__(self) -> Generator[asyncio.Future[str], None, str]:
            entered.set()
            return (yield from future.__await__())

    with live_input_scope(control, revision=0):
        control.admit("Updated scope", sender_id="parent", interrupt=True)
        with pytest.raises(RunInterrupted):
            await controlled_attempt(control, DeferredValue())
    assert not entered.is_set()
    assert not future.done()
    future.cancel()


@pytest.mark.asyncio
async def test_pydantic_iterator_without_terminal_result_fails_instead_of_returning_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def end_without_result(run: AgentRun[None, str]) -> NoReturn:
        raise StopAsyncIteration

    monkeypatch.setattr(AgentRun, "__anext__", end_without_result)
    control = LiveInput("worker")
    capability: InputDeliveryCapability[None] = InputDeliveryCapability(control=control)
    snapshots: list[AgentRun[None, str]] = []
    agent: Agent[None, str] = Agent(TestModel())
    token, usage = CancellationToken(), RunUsage()
    with pytest.raises(RuntimeError, match="without a terminal result"):
        await run_with_input(
            agent.iter("Review.", capabilities=[capability], cancellation_token=token),
            control=control,
            capability=capability,
            capture=snapshots.append,
            usage=usage,
        )
    assert len(snapshots) == 1 and snapshots[0].result is None
    assert usage.requests == 0
    assert control.delivered == ()


@pytest.mark.asyncio
async def test_langgraph_snapshot_does_not_invent_a_return_id_for_an_unidentified_tool_call() -> (
    None
):
    def incomplete_response(state: MessagesState) -> MessagesState:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read", "args": {}, "id": None},
                        {"name": "search", "args": {}, "id": "search-1"},
                    ],
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("model", incomplete_response)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    agent = graph.compile(checkpointer=InMemorySaver())
    captured: list[AnyMessage] = []
    result = await run_graph(
        agent,
        {"messages": [HumanMessage(content="Review.")]},
        {},
        LiveInput("worker"),
        captured.extend,
    )
    assert isinstance(result["messages"][-1], AIMessage)
    returned = [message for message in captured if isinstance(message, ToolMessage)]
    assert len(returned) == 1
    assert returned[0].tool_call_id == "search-1"
    assert returned[0].status == "error"
    assert "do not assume it succeeded" in returned[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_already_terminal_native_future_does_not_wait_for_an_input_notification(
    cancelled: bool,
) -> None:
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    if cancelled:
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await controlled_attempt(LiveInput("worker"), future)
    else:
        future.set_result("completed")
        assert await controlled_attempt(LiveInput("worker"), future) == "completed"


@pytest.mark.asyncio
async def test_input_during_native_result_handoff_cannot_publish_an_obsolete_value() -> None:
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)

    class HandoffFuture(asyncio.Future[str]):
        def __await__(self) -> Generator[asyncio.Future[str], None, str]:
            result = yield from super().__await__()
            control.admit("Updated scope", sender_id="parent", interrupt=True)
            return result

    future = HandoffFuture()
    future.set_result("obsolete")
    with pytest.raises(RunInterrupted):
        await controlled_attempt(control, future)
    assert control.pending
    assert control.revision == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", [False, True])
async def test_unconfirmed_stop_times_out_without_replacement_and_drains_late_completion(
    monkeypatch: pytest.MonkeyPatch, late_failure: bool
) -> None:
    import kedi.agent_adapter.subagents.control as control_module

    monkeypatch.setattr(control_module, "INTERRUPT_CLEANUP_SECONDS", 0.01)
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)

    async def operation() -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            if late_failure:
                raise ConnectionError("late native teardown failure") from None
            return "late completion"
        finally:
            stopped.set()
        return "unreachable"

    async def acknowledge_only() -> None:
        return None

    native = asyncio.create_task(operation())
    attempt = asyncio.create_task(controlled_attempt(control, native, stop=acknowledge_only))
    try:
        await entered.wait()
        control.admit("Updated scope", sender_id="parent", interrupt=True)
        with pytest.raises(RuntimeError, match="interrupt cleanup.*not restarted"):
            await attempt
        assert not stopped.is_set()
        assert control.pending
    finally:
        release.set()
        await asyncio.wait_for(stopped.wait(), 1)
        await asyncio.sleep(0)
    assert native.done()
    assert not native.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_error",
    [
        asyncio.CancelledError(),
        UnconfirmedStop("Native stop was not confirmed"),
        ConnectionError("Native stop transport disconnected"),
    ],
)
async def test_failed_native_stop_drains_its_operation_without_restarting(
    stop_error: BaseException,
) -> None:
    entered, stopped = asyncio.Event(), asyncio.Event()
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)

    async def operation() -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            stopped.set()
        return "unreachable"

    async def cancelled_stop() -> NoReturn:
        raise stop_error

    native = asyncio.create_task(operation())
    attempt = asyncio.create_task(controlled_attempt(control, native, stop=cancelled_stop))
    try:
        await entered.wait()
        control.admit("Updated scope", sender_id="parent", interrupt=True)
        error_type = (
            UnconfirmedStop if isinstance(stop_error, asyncio.CancelledError) else type(stop_error)
        )
        with pytest.raises(error_type):
            await attempt
        assert native.cancelled()
        assert stopped.is_set()
    finally:
        native.cancel()
        await asyncio.gather(native, return_exceptions=True)
