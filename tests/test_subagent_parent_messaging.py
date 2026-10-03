from __future__ import annotations

import asyncio
import gc
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TypeAlias

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import JsonValue, TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
from kedi.agent_adapter.subagents.control import InputReceipt, current_live_input
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.test_langchain_branch_parity import ScriptedModel


Reply: TypeAlias = tuple[None, str] | tuple[str, dict[str, JsonValue]]
Respond: TypeAlias = Callable[[Sequence[BaseMessage | ModelMessage]], Awaitable[Reply]]
TOOL_RESULT = TypeAdapter(dict[str, JsonValue])


@pytest.mark.asyncio
@pytest.mark.parametrize("abandon", ["cancel", "timeout"])
async def test_abandoned_parent_wait_retrieves_later_failure_without_hiding_task_error(
    abandon: str,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        entered.set()
        await release.wait()
        raise RuntimeError("review backend failed")

    adapter = PydanticAdapter(FunctionModel(fail))
    runtime = compile_program(
        parse_program("""
> profile: worker:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
> use: manager
= ready
"""),
        adapter=adapter,
    )
    coordinator = runtime._subagent_coordinator
    assert coordinator is not None
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[str] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(str(context.get("message"))))
    try:
        with pytest.raises(KediExecutionError, match="review backend failed"):
            async with runtime.subagents(parent="manager") as scope:
                job = await scope.start("worker", task="Review the original scope.")
                await asyncio.wait_for(entered.wait(), 2)
                if abandon == "cancel":
                    waiter = asyncio.create_task(coordinator.wait(job.task_id))
                    await asyncio.sleep(0)
                    waiter.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiter
                else:
                    with pytest.raises(TimeoutError):
                        await coordinator.wait(job.task_id, timeout=0.01)
                assert coordinator.status(job.task_id) == "running"
                release.set()
                with pytest.raises(KediExecutionError, match="review backend failed"):
                    await job.wait()
        gc.collect()
        await asyncio.sleep(0)
        assert unhandled == []
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)


def responding_adapter(name: str, respond: Respond) -> PydanticAdapter | LangChainAdapter:
    sequence = 0

    async def response(messages: Sequence[BaseMessage | ModelMessage]) -> Reply:
        nonlocal sequence
        sequence += 1
        return await respond(messages)

    if name == "pydantic":

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            tool, value = await response(messages)
            if tool is None:
                assert isinstance(value, str)
                return ModelResponse(parts=[TextPart(value)])
            assert isinstance(value, dict)
            return ModelResponse(parts=[ToolCallPart(tool, value, tool_call_id=f"call-{sequence}")])

        return PydanticAdapter(FunctionModel(model))

    class CallbackModel(ScriptedModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: JsonValue,
        ) -> ChatResult:
            self.seen.append(list(messages))
            tool, value = await response(messages)
            if tool is None:
                assert isinstance(value, str)
                message = AIMessage(content=value)
            else:
                assert isinstance(value, dict)
                message = AIMessage(
                    content="",
                    tool_calls=[{"name": tool, "args": value, "id": f"call-{sequence}"}],
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

    return LangChainAdapter(CallbackModel(responses=[]))


def tool_returns(messages: Sequence[BaseMessage | ModelMessage]) -> dict[str, dict[str, JsonValue]]:
    returns: dict[str, dict[str, JsonValue]] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.name is not None and isinstance(message.content, str)
            returns[message.name] = TOOL_RESULT.validate_json(message.content)
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    returns[part.tool_name] = TOOL_RESULT.validate_python(part.content, strict=True)
    return returns


@pytest.mark.parametrize(
    ("parent_backend", "child_backend"),
    [("pydantic", "langchain"), ("langchain", "pydantic")],
)
def test_child_message_wakes_mixed_backend_parent_and_parent_redirects_same_task(
    parent_backend: str, child_backend: str
) -> None:
    waiting = asyncio.Event()
    parent_steps = []
    child_inputs = []
    task_ids = []

    async def parent_response(messages: Sequence[BaseMessage | ModelMessage]) -> Reply:
        returned = tool_returns(messages)
        parent_steps.append(returned)
        step = len(parent_steps)
        if step == 1:
            return "delegate_task", {
                "subagent": "worker",
                "task": "Inspect the change.",
                "background": True,
            }
        task_id = returned["delegate_task"]["run_id"]
        task_ids.append(task_id)
        if step == 2:
            waiting.set()
            return "wait_subagent", {"run_id": task_id}
        if step == 3:
            assert returned["wait_subagent"]["status"] == "running"
            assert returned["wait_subagent"]["input_available"] is True
            assert "Which part should I review?" in str(messages)
            return "send_subagent_input", {
                "task_id": task_id,
                "message": "Review access tokens only.",
                "interrupt": True,
            }
        if step == 4:
            receipt = returned["send_subagent_input"]
            assert receipt["status"] == "interrupt_requested"
            assert receipt["target_id"] == task_id
            return "wait_subagent", {"run_id": task_id}
        assert step == 5
        assert returned["wait_subagent"]["run_id"] == task_id
        assert returned["wait_subagent"]["task_summary"] == "Access-token review complete."
        return None, "Review complete."

    async def child_response(messages: Sequence[BaseMessage | ModelMessage]) -> Reply:
        child_inputs.append(str(messages))
        if "Review access tokens only." in str(messages):
            return None, "Access-token review complete."
        if len(child_inputs) == 1:
            await asyncio.wait_for(waiting.wait(), 3)
            return "send_parent_message", {"message": "Which part should I review?"}
        await asyncio.Event().wait()
        return None, "obsolete"

    parent = responding_adapter(parent_backend, parent_response)
    child = responding_adapter(child_backend, child_response)
    source = f"""
> profile: worker:
    > adapter: {child_backend}
    > system: Review the supplied change.

> profile: manager:
    > adapter: {parent_backend}
    > subagent: worker

> use: manager
[answer] << Delegate a review and answer any question from the worker.
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=parent)
    coordinator = runtime._subagent_coordinator
    coordinator._deps = replace(coordinator._deps, resolve_adapter=lambda profile: child)
    assert runtime.run_main() == "Review complete."
    assert len(parent_steps) == 5
    assert len(set(task_ids)) == 1
    run = coordinator.runs.get(task_ids[0])
    assert run.status == "completed"
    assert run.requests >= 2
    assert "Inspect the change." in child_inputs[-1]
    assert "Review access tokens only." in child_inputs[-1]


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("persistent", [False, True])
def test_parent_input_after_response_resumes_native_history_before_publishing(
    backend: str, persistent: bool
) -> None:
    requests: list[Sequence[BaseMessage | ModelMessage]] = []
    receipts: list[InputReceipt] = []

    async def respond(messages: Sequence[BaseMessage | ModelMessage]) -> Reply:
        requests.append(messages)
        return None, "obsolete" if len(requests) == 1 else "updated"

    def add_late_input(usage: InvocationUsage) -> None:
        del usage
        if not receipts:
            control = current_live_input()
            assert control is not None and control.parent is None
            receipts.append(control.admit("Updated review scope.", sender_id="worker"))

    source = f"""
> profile: worker:
    > adapter: {backend}
> profile: manager:
    > adapter: {backend}
    > subagent: worker
> use: manager
[answer] << Review the original scope.
= <answer>
"""
    conversation = ConversationState() if persistent else None
    adapter = responding_adapter(backend, respond)
    runtime = compile_program(
        parse_program(source), adapter=adapter, conversation_state=conversation
    )
    try:
        with observe_invocations(add_late_input):
            assert runtime.run_main() == "updated"
        assert len(requests) == 2
        assert "obsolete" in str(requests[1])
        assert "Updated review scope." in str(requests[1])
        assert receipts[0].status == "queued"
        if conversation is not None:
            turns = conversation.snapshot()
            assert len(turns) == 2
            assert turns[-1].role == "assistant" and turns[-1].content == "updated"
    finally:
        if conversation is not None:
            conversation.close()


@pytest.mark.asyncio
async def test_kedi_input_cannot_reuse_a_task_from_a_closed_foreign_invocation() -> None:
    from tests.test_native_subagent_tasks import PROFILES

    requests: list[Sequence[BaseMessage | ModelMessage]] = []

    async def respond(messages: Sequence[BaseMessage | ModelMessage]) -> Reply:
        requests.append(messages)
        return None, "completed"

    adapter = responding_adapter("pydantic", respond)
    runtime = compile_program(parse_program(PROFILES + "= ready\n"), adapter=adapter)
    async with runtime.subagents(parent="manager") as agents:
        job = await agents.start("worker", task="Review.")
        await job.wait()
    foreign = compile_program(
        parse_program(PROFILES + "> send: retained:\n    >> Additional evidence.\n"),
        adapter=adapter,
        runtime_globals={"retained": job},
    )
    with pytest.raises(KediExecutionError, match="task's active invocation"):
        foreign.run_main()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("handle_type", ["native", "coordinator"])
async def test_finished_task_input_rejects_after_control_pruning(handle_type: str) -> None:
    from kedi.agent_adapter.subagents.coordinator import SubagentHandle
    from tests.test_native_subagent_tasks import PROFILES

    requests = []

    async def respond(messages, info):
        requests.append(messages)
        return ModelResponse(parts=[TextPart("completed")])

    runtime = compile_program(
        parse_program(PROFILES + "= ready\n"), adapter=PydanticAdapter(FunctionModel(respond))
    )
    async with runtime.subagents(parent="manager") as agents:
        job = await agents.start("worker", task="Review the change.")
        await job.wait()
        coordinator = runtime._subagent_coordinator
        coordinator._controls.pop(job.run_id)
        handle = job if handle_type == "native" else SubagentHandle(job.run_id, coordinator)
        receipt = await handle.send("Late evidence")
        assert receipt.status == "rejected"
        assert receipt.reason == "task_finished"
        assert receipt.target_id == job.run_id
    assert len(requests) == 1
