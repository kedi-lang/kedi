from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Event

import pytest

from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.subagents import DelegationResult, SubagentConversationContext
from kedi.agent_adapter.subagents.control import (
    MAX_INPUT_BYTES,
    MAX_PENDING_INPUTS,
    LiveInput,
    RunInterrupted,
    controlled_attempt,
    live_input_scope,
)
from kedi.core.runtime import KediRuntime
from kedi.lang import KediParseError, parse_program


def inbox(name: str = "worker") -> LiveInput:
    result = LiveInput(name)
    result.configure(input=True, interrupt=True)
    return result


def test_admission_fences_finalization_and_preserves_fifo() -> None:
    control = inbox()
    first = control.admit("Extra context", sender_id="parent")
    assert first.status == "queued"
    assert not control.try_finish(0, lambda: None)
    second = control.admit("Updated scope", sender_id="parent", interrupt=True)
    assert second.revision == 1
    assert second.status == "interrupt_requested"
    assert control.admit("Another redirect", sender_id="parent", interrupt=True).reason == "busy"
    assert not control.try_finish(0, lambda: None)
    assert control.take_pending(revision=0) == ()
    assert control.begin_attempt() == 1
    items = control.take_pending(revision=1)
    assert [m.message for m in items] == ["Extra context", "Updated scope"]
    control.mark_delivered(items)
    assert control.delivered == (first.message_id, second.message_id)
    assert control.try_finish(1, lambda: None)
    assert control.admit("Too late", sender_id="parent").reason == "task_finished"


def test_bounded_input_rejects_without_eviction() -> None:
    control = inbox()
    assert (
        control.admit("x" * (MAX_INPUT_BYTES + 1), sender_id="parent").reason == "input_too_large"
    )
    receipts = [control.admit(str(i), sender_id="parent") for i in range(MAX_PENDING_INPUTS)]
    overflow = control.admit("overflow", sender_id="parent")
    assert overflow.reason == "queue_full"
    assert overflow.message_id is None
    assert [i.message_id for i in control.take_pending()] == [r.message_id for r in receipts]


def test_claimed_messages_remain_bounded_and_restore_in_fifo_order() -> None:
    control = inbox()
    receipts = [control.admit(str(i), sender_id="parent") for i in range(MAX_PENDING_INPUTS)]
    items = control.take_pending()
    assert control.admit("overflow", sender_id="parent").reason == "queue_full"
    control.confirm_contents((items[0].render(),))
    last = control.admit("last", sender_id="parent")
    assert last.status == "queued"
    control.begin_attempt()
    assert [i.message_id for i in control.take_pending()] == [
        *(r.message_id for r in receipts[1:]),
        last.message_id,
    ]


def test_input_capacity_measures_utf8_not_characters() -> None:
    control = inbox()
    message = "\N{SNOWMAN}" * (MAX_INPUT_BYTES // 3 + 1)
    assert len(message) < MAX_INPUT_BYTES
    assert control.admit(message, sender_id="parent").reason == "input_too_large"


def test_routing_rejects_foreign_parent_and_superseded_sender() -> None:
    from kedi.agent_adapter.subagents.communication import send_input
    from kedi.agent_adapter.subagents.messaging import input_tool_specs

    parent, other = inbox("parent"), inbox("other")
    child = LiveInput("child", parent=parent)
    child.configure(input=True, interrupt=True)
    runtime = KediRuntime()
    coordinator = runtime._subagent_coordinator
    coordinator._controls["child"] = child
    assert (
        send_input(coordinator, "child", "context", interrupt=False, sender=other).reason
        == "not_authorized"
    )
    with live_input_scope(parent):
        parent.admit("new scope", sender_id="owner", interrupt=True)
        assert (
            send_input(coordinator, "child", "context", interrupt=False, sender=parent).reason
            == "not_authorized"
        )
    assert len(input_tool_specs(runtime, children=False, parent=False)) == 0


@pytest.mark.asyncio
async def test_model_wait_ignores_tool_completion_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter.subagents.messaging import wait_for_child

    control = inbox("parent")
    done = asyncio.Event()

    result = DelegationResult(run_id="child", subagent="worker", task_summary="Completed.")

    async def wait(run_id: str, *, timeout: float | None = None) -> DelegationResult:
        await done.wait()
        return result

    rt = KediRuntime()
    monkeypatch.setattr(rt, "_wait_subagent", wait)
    with live_input_scope(control):
        waiting = asyncio.create_task(wait_for_child(rt, "child", None))
        await asyncio.sleep(0)
        control.tool_started()
        control.tool_finished()
        await asyncio.sleep(0.01)
        assert not waiting.done()
        done.set()
        assert await waiting is result


def test_concurrent_admission_and_completion_are_atomic() -> None:
    for _ in range(30):
        control = inbox()
        with ThreadPoolExecutor(max_workers=2) as pool:
            send = pool.submit(control.admit, "Context", sender_id="parent")
            finish = pool.submit(control.try_finish, 0, lambda: None)
            receipt, completed = send.result(), finish.result()
        assert (receipt.status == "queued" and not completed) or (
            receipt.reason == "task_finished" and completed
        )


@pytest.mark.asyncio
async def test_controlled_attempt_interrupt_waits_for_cleanup() -> None:
    control = inbox()
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    attempt = asyncio.create_task(controlled_attempt(control, operation()))
    await started.wait()
    control.admit("New scope", sender_id="parent", interrupt=True)
    with pytest.raises(RunInterrupted):
        await attempt
    assert cleaned.is_set()
    assert control.pending


@pytest.mark.asyncio
async def test_pydantic_input_redirects_final_response_without_interrupt() -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    started, release = asyncio.Event(), asyncio.Event()
    requests: list[str] = []

    async def model(messages, info):
        del info
        requests.append(str(messages))
        if len(requests) == 1:
            started.set()
            await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    adapter = PydanticAdapter(model=FunctionModel(model))
    control = inbox()
    conversation = SubagentConversationContext(conversation_id="conversation")
    with live_input_scope(control), adapter.conversation_scope(conversation):
        task = asyncio.create_task(adapter.invoke(prompt="Original task"))
        await started.wait()
        receipt = control.admit("Check the migration too", sender_id="parent")
        await asyncio.sleep(0)
        release.set()
        assert await task == "answer"
    assert len(requests) == 2
    assert "Check the migration too" in requests[-1]
    assert receipt.message_id in control.delivered
    assert conversation.result_state["pydantic_messages"]


@pytest.mark.asyncio
async def test_pydantic_interrupt_retains_partial_history() -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    started = asyncio.Event()

    async def model(messages, info):
        del messages, info
        started.set()
        await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart("obsolete")])

    adapter = PydanticAdapter(model=FunctionModel(model))
    control = inbox()
    conversation = SubagentConversationContext(conversation_id="conversation")
    with live_input_scope(control), adapter.conversation_scope(conversation):
        task = asyncio.create_task(adapter.invoke(prompt="Original task"))
        await started.wait()
        control.admit("Replacement task", sender_id="parent", interrupt=True)
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(task, 2)
    assert conversation.result_state["pydantic_messages"]
    assert control.take_pending()[0].message == "Replacement task"


@pytest.mark.parametrize("operation", ["send", "interrupt"])
def test_agent_input_syntax_rejects_output_and_multiple_templates(operation: str) -> None:
    parse_program(f"> {operation}: job:\n    >> Include <notes>.\n")
    with pytest.raises(KediParseError):
        parse_program(f"> {operation}: job:\n    >> The result is [result].\n")
    with pytest.raises(KediParseError):
        parse_program(f"> {operation}: job:\n    >> First.\n    >> Second.\n")


@pytest.mark.asyncio
async def test_langchain_queued_input_reaches_next_request() -> None:
    from typing import Any

    from langchain_core.messages import AIMessage

    from kedi.agent_adapter.adapters.langchain import LangChainAdapter
    from tests.test_langchain_branch_parity import ScriptedModel

    class WaitingModel(ScriptedModel):
        started: Any
        release: Any

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            if not self.seen:
                self.started.set()
                await self.release.wait()
            return self._generate(messages, stop, run_manager, **kwargs)

    started, release = asyncio.Event(), asyncio.Event()
    model = WaitingModel(
        responses=[AIMessage(content="first"), AIMessage(content="updated")],
        started=started,
        release=release,
    )
    adapter = LangChainAdapter(model)
    control = inbox()
    conversation = SubagentConversationContext(conversation_id="conversation")
    with live_input_scope(control), adapter.conversation_scope(conversation):
        task = asyncio.create_task(adapter.invoke(prompt="Original task"))
        await started.wait()
        receipt = control.admit("Check the migration too", sender_id="parent")
        release.set()
        assert await asyncio.wait_for(task, 3) == "updated"
    assert len(model.seen) == 2
    assert "Check the migration too" in str(model.seen[-1])
    assert receipt.message_id in control.delivered


@pytest.mark.asyncio
async def test_langchain_interrupt_preserves_checkpoint() -> None:
    from typing import Any

    from kedi.agent_adapter.adapters.langchain import LangChainAdapter
    from tests.test_langchain_branch_parity import ScriptedModel

    class WaitingModel(ScriptedModel):
        started: Any

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            self.started.set()
            await asyncio.Event().wait()

    started = asyncio.Event()
    adapter = LangChainAdapter(WaitingModel(responses=[], started=started))
    control = inbox()
    conversation = SubagentConversationContext(conversation_id="conversation")
    with live_input_scope(control), adapter.conversation_scope(conversation):
        task = asyncio.create_task(adapter.invoke(prompt="Original task"))
        await started.wait()
        control.admit("Replacement task", sender_id="parent", interrupt=True)
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(task, 3)
    assert "Original task" in str(conversation.result_state["langchain_messages"])
    assert control.take_pending()[0].message == "Replacement task"


@pytest.mark.asyncio
async def test_native_task_redirect_keeps_identity_budget_and_repeated_wait() -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.lang import compile_program
    from tests.test_native_subagent_tasks import PROFILES

    started = asyncio.Event()
    requests: list[str] = []

    async def model(messages, info):
        del info
        requests.append(str(messages))
        if len(requests) == 1:
            started.set()
            await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart("revised result")])

    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"), adapter=PydanticAdapter(FunctionModel(model))
    )
    async with runtime.subagents(parent="manager") as scope:
        job = await scope.start("worker", task="Original task")
        await started.wait()
        receipt = await job.send("Review only authentication", interrupt=True)
        assert receipt.status == "interrupt_requested"
        result = await asyncio.wait_for(job.wait(), 3)
        assert result.task_summary == "revised result"
        assert result.run_id == job.task_id == receipt.target_id
        assert await job.wait() is result
        assert (await job.send("Late message")).reason == "task_finished"
        run = runtime._subagent_coordinator.runs.get(job.run_id)
        assert run.requests == 2
        assert (
            scope._owner.budget_for(
                runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
            ).started
            == 1
        )
    assert "Original task" in requests[1]
    assert "Review only authentication" in requests[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["Stop", "PostToolUse", "PostToolUseFailure"])
async def test_claude_hooks_preserve_existing_hooks_and_block_finalization(
    monkeypatch: pytest.MonkeyPatch,
    event: str,
) -> None:
    pytest.importorskip("claude_agent_sdk")
    import claude_agent_sdk
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage
    from claude_agent_sdk.types import HookInput, HookJSONOutput

    from kedi.agent_adapter.adapters.claude.live import run_query

    entered, release = asyncio.Event(), asyncio.Event()
    outputs: list[HookJSONOutput] = []

    class Client(ClaudeSDKClient):
        def __init__(self, *, options: ClaudeAgentOptions) -> None:
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def query(self, prompt):
            assert prompt == "task"

        async def interrupt(self):
            pytest.fail("Ordinary input must not interrupt Claude")

        async def receive_response(self):
            entered.set()
            await release.wait()
            hook_input: HookInput
            if event == "Stop":
                hook_input = {
                    "hook_event_name": "Stop",
                    "session_id": "session",
                    "transcript_path": "/tmp/transcript.jsonl",
                    "cwd": "/tmp",
                    "stop_hook_active": False,
                }
            elif event == "PostToolUse":
                hook_input = {
                    "hook_event_name": "PostToolUse",
                    "session_id": "session",
                    "transcript_path": "/tmp/transcript.jsonl",
                    "cwd": "/tmp",
                    "tool_name": "read",
                    "tool_input": {"path": "notes.txt"},
                    "tool_response": "Evidence",
                    "tool_use_id": "read-1",
                }
            else:
                hook_input = {
                    "hook_event_name": "PostToolUseFailure",
                    "session_id": "session",
                    "transcript_path": "/tmp/transcript.jsonl",
                    "cwd": "/tmp",
                    "tool_name": "read",
                    "tool_input": {"path": "notes.txt"},
                    "error": "Not found",
                    "tool_use_id": "read-1",
                }
            for matcher in self.options.hooks[event]:
                for callback in matcher.hooks:
                    outputs.append(await callback(hook_input, None, {}))
            outputs.append(await self.options.hooks[event][-1].hooks[0](hook_input, None, {}))
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="session",
                result="updated",
            )

    async def existing(*args):
        return {"systemMessage": "existing"}

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    control = inbox()
    options = ClaudeAgentOptions(hooks={event: [HookMatcher(hooks=[existing])]})
    task = asyncio.create_task(run_query(options, "task", control, lambda message: None))
    await entered.wait()
    receipt = control.admit("Additional evidence", sender_id="parent")
    release.set()
    assert (await task).result == "updated"
    assert outputs[0] == {"systemMessage": "existing"}
    if event == "Stop":
        assert outputs[1]["decision"] == "block"
        assert "Additional evidence" in outputs[1]["reason"]
    else:
        assert outputs[1]["hookSpecificOutput"]["hookEventName"] == event
        assert "Additional evidence" in outputs[1]["hookSpecificOutput"]["additionalContext"]
    assert outputs[2] == {}
    assert receipt.message_id in control.delivered
    assert len(options.hooks[event]) == 1


@pytest.mark.asyncio
async def test_parent_message_wakes_model_wait_but_not_native_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter.subagents.messaging import wait_for_child

    control = inbox("parent")
    done = asyncio.Event()

    async def wait(run_id: str, *, timeout: float | None = None) -> DelegationResult:
        await done.wait()
        return DelegationResult(run_id=run_id, subagent="worker", task_summary="Completed.")

    rt = KediRuntime()
    monkeypatch.setattr(rt, "_wait_subagent", wait)
    rt._subagent_coordinator.runs.create(
        run_id="child", parent_run_id="parent", root_run_id="parent", profile_name="worker", depth=1
    )
    rt._subagent_coordinator.runs.transition("child", "running")
    with live_input_scope(control):
        waiting = asyncio.create_task(wait_for_child(rt, "child", None))
        await asyncio.sleep(0)
        control.admit("I need more context", sender_id="child")
        result = await asyncio.wait_for(waiting, 1)
    assert result["status"] == "running"
    assert result["input_available"] is True
    assert not done.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
async def test_codex_steer_and_interrupt_reuse_native_thread(tmp_path, interrupt) -> None:
    from threading import Event

    from tests.test_codex_adapter import _adapter, _messages, _method

    adapter, log = _adapter(tmp_path, turn_timeout=3)
    confirmed = Event()
    control = LiveInput(
        "worker", observe=lambda item, phase: confirmed.set() if phase == "delivered" else None
    )
    control.configure(input=True, interrupt=True)
    initial = control.admit("Initial evidence", sender_id="parent")
    claimed = control.take_pending()
    conversation = SubagentConversationContext(conversation_id="conversation")
    try:
        with live_input_scope(control), adapter.conversation_scope(conversation):
            task = asyncio.create_task(
                adapter.invoke(prompt="live-input\n\n" + claimed[0].render())
            )
            for _ in range(100):
                if log.exists() and _method(_messages(log), "turn/start"):
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Native turn did not start")
            assert await asyncio.to_thread(confirmed.wait, 1)
            assert initial.message_id in control.delivered
            receipt = control.admit(
                "Review authentication", sender_id="parent", interrupt=interrupt
            )
            if interrupt:
                with pytest.raises(RunInterrupted):
                    await asyncio.wait_for(task, 3)
                conversation.resume_state.update(conversation.result_state)
                control.begin_attempt()
                inputs = control.take_pending()
                with live_input_scope(control, revision=control.revision):
                    await adapter.invoke(prompt=inputs[0].render())
            else:
                text = await asyncio.wait_for(task, 3)
                assert "Review authentication" in text
                assert receipt.message_id in control.delivered
        messages = _messages(log)
        assert len(_method(messages, "thread/start")) == 1
        if interrupt:
            assert len(_method(messages, "turn/interrupt")) == 1
            assert len(_method(messages, "thread/resume")) == 1
        else:
            assert len(_method(messages, "turn/steer")) == 1
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_tool_cleanup_blocks_restart_until_thread_returns() -> None:
    from threading import Event

    from kedi.agent_adapter.subagents.tool_activity import with_live_tool_tracking
    from kedi.agent_adapter.tools.specs import ToolSpec

    started, release = Event(), Event()
    control = inbox()

    def tool():
        started.set()
        release.wait(timeout=2)
        return "done"

    tracked = with_live_tool_tracking(ToolSpec(fn=tool, name="tool", risk="read_only"))
    with live_input_scope(control):
        work = asyncio.create_task(asyncio.to_thread(tracked.fn))
        await asyncio.to_thread(started.wait, 1)
        quiet = asyncio.create_task(control.wait_quiescent())
        await asyncio.sleep(0.01)
        assert not quiet.done()
        release.set()
        assert await work == "done"
        await quiet


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("structured", [False, True])
async def test_interrupt_retains_completed_parallel_tool_results(
    adapter_name: str, synchronous: bool, structured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.messages import AIMessage
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter.adapters.langchain import LangChainAdapter
    from kedi.agent_adapter.subagents.tool_activity import with_live_tool_tracking
    from kedi.agent_adapter.tools.specs import ToolSpec
    from tests.test_langchain_branch_parity import ScriptedModel

    completed, started = Event(), asyncio.Event()
    effects: list[str] = []

    def fast_sync() -> str | dict[str, str]:
        effects.append("committed")
        completed.set()
        return {"evidence": "confirmed host effect"} if structured else "confirmed host effect"

    async def fast() -> str | dict[str, str]:
        return fast_sync()

    async def slow() -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tools = tuple(
        with_live_tool_tracking(ToolSpec(fn=fn, name=name, risk="read_only"))
        for name, fn in (("fast", fast_sync if synchronous else fast), ("slow", slow))
    )
    calls = [{"id": name, "name": name, "args": {}} for name in ("fast", "slow")]
    if adapter_name == "pydantic":

        async def model(messages, info):
            return ModelResponse(parts=[ToolCallPart(c["name"], c["args"], c["id"]) for c in calls])

        adapter = PydanticAdapter(FunctionModel(model))
        history_key = "pydantic_messages"
    else:
        from langchain_core.tools import StructuredTool

        arun = StructuredTool._arun

        @wraps(arun)
        async def delay_delivery(self, *args, **kwargs):
            result = await arun(self, *args, **kwargs)
            if self.name == "fast":
                await asyncio.Event().wait()
            return result

        monkeypatch.setattr(StructuredTool, "_arun", delay_delivery)
        adapter = LangChainAdapter(
            ScriptedModel(responses=[AIMessage(content="", tool_calls=calls)])
        )
        history_key = "langchain_messages"
    control = inbox()
    conversation = SubagentConversationContext(conversation_id="parallel-tools")
    with (
        live_input_scope(control),
        adapter.conversation_scope(conversation),
        adapter.tool_scope(tools),
    ):
        task = asyncio.create_task(adapter.invoke(prompt="Run the two tools"))
        assert await asyncio.to_thread(completed.wait, 2)
        await asyncio.wait_for(started.wait(), 2)
        control.admit("Change scope", sender_id="parent", interrupt=True)
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(task, 3)
    history = str(conversation.result_state[history_key])
    assert "confirmed host effect" in history
    assert effects == ["committed"]


@pytest.mark.asyncio
async def test_langchain_thread_result_is_recorded_before_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import AIMessage

    from kedi.agent_adapter.adapters.langchain import LangChainAdapter, runtime as native_runtime
    from kedi.agent_adapter.tools.specs import ToolSpec
    from tests.test_langchain_branch_parity import ScriptedModel

    entered, release = Event(), Event()
    record = native_runtime.record_live_tool_result

    def delayed_record(result):
        entered.set()
        assert release.wait(3)
        record(result)

    monkeypatch.setattr(native_runtime, "record_live_tool_result", delayed_record)
    adapter = LangChainAdapter(
        ScriptedModel(
            responses=[
                AIMessage(content="", tool_calls=[{"id": "save", "name": "save", "args": {}}])
            ]
        )
    )
    control = inbox()
    conversation = SubagentConversationContext(conversation_id="thread-result")
    tool = ToolSpec(fn=lambda: "committed result", name="save", risk="read_only")
    quiet = None
    with (
        live_input_scope(control),
        adapter.conversation_scope(conversation),
        adapter.tool_scope((tool,)),
    ):
        task = asyncio.create_task(adapter.invoke(prompt="Save the result."))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            control.admit("Updated scope", sender_id="parent", interrupt=True)
            quiet = asyncio.create_task(control.wait_quiescent())
            done, _ = await asyncio.wait((quiet,), timeout=0.01)
            assert not done
            assert not task.done()
            release.set()
            with pytest.raises(RunInterrupted):
                await asyncio.wait_for(task, 3)
            await quiet
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if quiet is not None:
                await quiet
    assert "committed result" in str(conversation.result_state["langchain_messages"])


@pytest.mark.parametrize("inside_procedure", [False, True])
def test_send_and_interrupt_dsl_execute_and_keep_typed_result(inside_procedure: bool) -> None:
    from textwrap import indent
    from threading import Event

    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.lang import compile_program
    from tests.test_native_subagent_tasks import PROFILES

    started = Event()
    requests: list[str] = []

    async def model(messages, info):
        requests.append(str(messages))
        if len(requests) == 1:
            started.set()
            await asyncio.Event().wait()
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"task_summary": "revised", "final_result": {"value": 7}},
                )
            ]
        )

    body = (
        "> task [job]: worker:\n    >> The result is [value: int].\n"
        + "`started.wait(2)`\n"
        + "> send: job:\n    >> Include the migration notes.\n"
        + "> interrupt: job:\n    >> Only review authentication.\n"
        + "> await [answer]: job\n= `answer.output.value`\n"
    )
    source = PROFILES + (
        "@review() -> int:\n" + indent(body, "    ") + "= `review()`\n"
        if inside_procedure
        else body
    )
    runtime = compile_program(parse_program(source), adapter=PydanticAdapter(FunctionModel(model)))
    runtime._initial_globals_env["started"] = started
    assert runtime.run_main() == 7
    assert len(requests) == 2
    assert "Include the migration notes" in requests[1]
    assert "Only review authentication" in requests[1]


def test_send_handle_definition_references_and_virtual_python() -> None:
    import ast

    from kedi.lsp.python_virtual.document import compute_python_virtual_document
    from tests.lsp_helpers import definition_at, references_at
    from tests.test_native_subagent_tasks import PROFILES

    source = (
        PROFILES
        + "> task [job]: worker:\n    >> Review.\n"
        + "> send: job:\n    >> Include <notes>.\n"
        + "> interrupt: job:\n    >> Review <`notes.strip()`>.\n"
        + "> await [answer]: job\n"
    )
    path, uri = "/tmp/messaging.kedi", "file:///tmp/messaging.kedi"
    target = definition_at(source, uri, path, "job", occurrence=2)
    assert target is not None
    refs = references_at(source, uri, path, "job", True, occurrence=2)
    assert len(refs) == 4
    virtual = compute_python_virtual_document(source, path)
    assert virtual is not None
    ast.parse(virtual["text"])
    assert "notes.strip()" in virtual["text"]


@pytest.mark.asyncio
async def test_redirect_cannot_reset_exhausted_request_budget() -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
    from kedi.errors import KediExecutionError
    from kedi.lang import compile_program
    from tests.test_native_subagent_tasks import PROFILES

    started = asyncio.Event()
    requests = []

    async def model(messages, info):
        requests.append(messages)
        started.set()
        await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart("obsolete")])

    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"), adapter=PydanticAdapter(FunctionModel(model))
    )
    runtime._subagent_coordinator.usage_limits = SubagentUsageLimits(request_limit=1)
    with pytest.raises(KediExecutionError, match="request_limit exhausted"):
        async with runtime.subagents(parent="manager") as scope:
            job = await scope.start("worker", task="original")
            await started.wait()
            await job.send("revised", interrupt=True)
            await job.wait()
    assert runtime._subagent_coordinator.runs.get(job.run_id).requests == 1
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_native_owner_receives_child_messages_without_parent_inference() -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.lang import compile_program
    from tests.test_native_subagent_tasks import PROFILES

    release = asyncio.Event()
    requests = []

    async def model(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            assert "send_parent_message" in {t.name for t in info.function_tools}
            return ModelResponse(
                parts=[ToolCallPart("send_parent_message", {"message": "Need migration evidence"})]
            )
        await release.wait()
        return ModelResponse(parts=[TextPart("reviewed")])

    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"), adapter=PydanticAdapter(FunctionModel(model))
    )
    async with runtime.subagents(parent="manager") as scope:
        job = await scope.start("worker", task="review")
        messages = await asyncio.wait_for(scope.receive_messages(), 2)
        assert len(messages) == 1
        assert messages[0].message == "Need migration evidence"
        assert messages[0].sender_id == job.run_id
        waiting = asyncio.create_task(job.wait())
        await asyncio.sleep(0)
        assert not waiting.done()
        release.set()
        assert (await waiting).task_summary == "reviewed"
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_interrupt_cleanup_failure_is_bounded_and_never_restarts(monkeypatch) -> None:
    import kedi.agent_adapter.subagents.control as module

    monkeypatch.setattr(module, "INTERRUPT_CLEANUP_SECONDS", 0.02)
    control = inbox()
    started, cleaned = asyncio.Event(), asyncio.Event()
    stops = []

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def stop():
        stops.append(True)
        await asyncio.Event().wait()

    task = asyncio.create_task(controlled_attempt(control, operation(), stop=stop))
    await started.wait()
    control.admit("replacement", sender_id="parent", interrupt=True)
    with pytest.raises(RuntimeError, match="task was not restarted"):
        await asyncio.wait_for(task, 0.5)
    await asyncio.wait_for(cleaned.wait(), 0.5)
    assert len(stops) == 1


@pytest.mark.asyncio
async def test_revision_changes_before_driver_starts_fence_old_result() -> None:
    control = inbox()
    with live_input_scope(control, revision=0):
        control.admit("new scope", sender_id="parent", interrupt=True)

        async def operation():
            return "obsolete"

        with pytest.raises(RunInterrupted):
            await controlled_attempt(control, operation())


@pytest.mark.asyncio
async def test_non_interrupting_input_does_not_cancel_async_tool() -> None:
    control = inbox()
    started, release = asyncio.Event(), asyncio.Event()
    deliveries = []

    async def operation():
        started.set()
        await release.wait()
        return "tool result"

    async def deliver(items):
        deliveries.extend(items)

    attempt = asyncio.create_task(controlled_attempt(control, operation(), deliver=deliver))
    await started.wait()
    control.admit("additional evidence", sender_id="parent")
    await asyncio.sleep(0.01)
    assert not attempt.done()
    assert len(deliveries) == 1
    release.set()
    assert await attempt == "tool result"


@pytest.mark.asyncio
async def test_logical_cancel_during_redirect_still_drains_physical_attempt() -> None:
    control = inbox()
    started, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()

    task = asyncio.create_task(controlled_attempt(control, operation()))
    await started.wait()
    control.admit("new scope", sender_id="parent", interrupt=True)
    await cleaning.wait()
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_unconfirmed_native_stop_fails_instead_of_redirecting() -> None:
    from kedi.agent_adapter.subagents.control import UnconfirmedStop

    control = inbox()
    started, disconnected = asyncio.Event(), asyncio.Event()

    async def operation():
        started.set()
        await disconnected.wait()
        raise UnconfirmedStop("terminal response missing")

    async def stop():
        disconnected.set()

    task = asyncio.create_task(controlled_attempt(control, operation(), stop=stop))
    await started.wait()
    control.admit("replacement", sender_id="parent", interrupt=True)
    with pytest.raises(UnconfirmedStop, match="terminal response missing"):
        await task


def test_superseded_tool_dispatch_does_not_start_host_effect() -> None:
    from kedi.agent_adapter.subagents.tool_activity import with_live_tool_tracking
    from kedi.agent_adapter.tools.specs import ToolSpec

    effects = []
    tracked = with_live_tool_tracking(ToolSpec(fn=lambda: effects.append("effect"), name="mutate"))
    control = inbox()
    with live_input_scope(control, revision=0):
        control.admit("new scope", sender_id="parent", interrupt=True)
        with pytest.raises(RuntimeError, match="superseded"):
            tracked.fn()
    assert effects == []


def test_close_records_undelivered_claims_without_releasing_them_twice() -> None:
    events = []
    control = LiveInput(
        "worker", observe=lambda item, phase: events.append((item.message_id, phase))
    )
    control.configure(input=True)
    first = control.admit("first", sender_id="parent")
    claimed = control.take_pending()
    second = control.admit("second", sender_id="parent")
    control.close()
    control.mark_delivered(claimed)
    control.close()
    assert not control.waiting_input
    assert events == [
        (first.message_id, "admitted"),
        (second.message_id, "admitted"),
        (first.message_id, "undelivered"),
        (second.message_id, "undelivered"),
    ]


@pytest.mark.asyncio
async def test_redirect_cancels_obsolete_descendants_but_keeps_sibling_running() -> None:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from kedi.lang import compile_program

    worker_started, leaf_started, sibling_started, leaf_cancelled, release_sibling = (
        asyncio.Event() for _ in range(5)
    )

    async def model(messages, info):
        text = str(messages)
        if "Original parent task" in text:
            if "Updated scope" in text:
                return ModelResponse(parts=[TextPart("revised")])
            if not any(
                isinstance(part, ToolReturnPart) and part.tool_name == "delegate_task"
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            ):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "delegate_task",
                            {"subagent": "leaf", "task": "Leaf task", "background": True},
                        )
                    ]
                )
            worker_started.set()
            await asyncio.Event().wait()
        elif "Sibling task" in text:
            sibling_started.set()
            await release_sibling.wait()
            return ModelResponse(parts=[TextPart("sibling")])
        else:
            leaf_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                leaf_cancelled.set()
        return ModelResponse(parts=[TextPart("obsolete")])

    source = """> profile: leaf:
    > adapter: pydantic
> profile: worker:
    > adapter: pydantic
    > subagent: leaf
> profile: observer:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
    > subagent: observer
    > max_agents: 4
> use: manager
= ready
"""
    runtime = compile_program(parse_program(source), adapter=PydanticAdapter(FunctionModel(model)))
    async with runtime.subagents(parent="manager") as scope:
        worker = await scope.start("worker", task="Original parent task")
        sibling = await scope.start("observer", task="Sibling task")
        await asyncio.wait_for(
            asyncio.gather(worker_started.wait(), leaf_started.wait(), sibling_started.wait()), 3
        )
        await worker.send("Updated scope", interrupt=True)
        assert (await asyncio.wait_for(worker.wait(), 3)).task_summary == "revised"
        assert leaf_cancelled.is_set()
        assert runtime._subagent_coordinator.runs.get(sibling.run_id).status == "running"
        release_sibling.set()
        assert (await sibling.wait()).task_summary == "sibling"
    leaf = next(
        r for r in runtime._subagent_coordinator.runs.snapshot() if r.profile_name == "leaf"
    )
    assert leaf.status == "cancelled"
