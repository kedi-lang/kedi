"""Findings from the paid OpenCode runs: #205 malformed subagent tool arguments, #206 the request
ceiling's context window, #207 `> task` fan-out beyond the subagent concurrency limit."""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.request_budget import (
    ModelRequestBudget,
    ModelRequestBudgetError,
)
from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError

DELEGATING_HOST = (
    "> profile: worker:\n  > instructions: Work.\n"
    "> profile: host:\n  > subagent: worker\n  > workflow: delegate\n"
    "> use: host\n[reply] << Ask the worker.\n= <reply>\n"
)

MALFORMED_CALLS = [
    ("continue_subagent", {"run_id": "r1", "task": "x", "final_schema": '{"type": "object"}'}),
    ("delegate_task", {"subagent": "worker", "task": "x", "final_schema": '{"type": "object"}'}),
    ("continue_subagent", {"run_id": "r1", "task": "x", "unexpected": 1}),
]


def _run(runtime):
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize(("tool", "arguments"), MALFORMED_CALLS)
def test_pydantic_reports_malformed_subagent_arguments_to_the_model(tool, arguments):
    feedback: list[str] = []

    def respond(messages, info):
        if tool not in [definition.name for definition in info.function_tools]:
            return ModelResponse(parts=[TextPart("child done")])
        parts = [part for message in messages for part in getattr(message, "parts", [])]
        replies = [part for part in parts if isinstance(part, (ToolReturnPart, RetryPromptPart))]
        if not replies:
            return ModelResponse(parts=[ToolCallPart(tool, arguments)])
        feedback.append(str(replies[-1].content))
        return ModelResponse(parts=[TextPart("done")])

    runtime = compile_program(
        parse_program(DELEGATING_HOST), adapter=PydanticAdapter(FunctionModel(respond))
    )
    assert _run(runtime) == "done"
    assert len(feedback) == 1
    assert f"Invalid arguments for {tool}" in feedback[0]
    assert "Allowed arguments:" in feedback[0]


@pytest.mark.parametrize(("tool", "arguments"), MALFORMED_CALLS)
def test_langchain_reports_malformed_subagent_arguments_to_the_model(tool, arguments):
    feedback: list[str] = []

    class ChatModel(FakeMessagesListChatModel):
        parent: bool = False

        def bind_tools(self, tools, **kwargs):
            return self.model_copy(update={"parent": any(t.name == tool for t in tools)})

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            replies = [message for message in messages if isinstance(message, ToolMessage)]
            if not self.parent:
                message = AIMessage(content="child done")
            elif not replies:
                call = {"name": tool, "args": arguments, "id": "call-1"}
                message = AIMessage(content="", tool_calls=[call])
            else:
                feedback.append(str(replies[-1].content))
                message = AIMessage(content="done")
            return ChatResult(generations=[ChatGeneration(message=message)])

    runtime = compile_program(
        parse_program(DELEGATING_HOST), adapter=LangChainAdapter(ChatModel(responses=[]))
    )
    assert _run(runtime) == "done"
    assert len(feedback) == 1
    assert f"Invalid arguments for {tool}" in feedback[0]


def test_task_fan_out_beyond_the_concurrency_limit_waits_for_slots():
    active = 0
    peak = 0

    async def respond(messages, info):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return ModelResponse(parts=[TextPart("ok")])

    count = 7
    source = (
        "> profile: worker:\n  > instructions: Work.\n"
        f"> profile: host:\n  > subagent: worker\n  > max_agents: {count}\n"
        "> use: host\n"
        + "".join(f"> task [j{i}]: worker:\n    >> Say ok {i}.\n" for i in range(count))
        + "".join(f"> await [r{i}]: j{i}\n" for i in range(count))
        + "= `[r.task_summary for r in ("
        + ", ".join(f"r{i}" for i in range(count))
        + ")]`\n"
    )
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(respond)),
        subagent_max_concurrency=3,
    )
    assert _run(runtime) == ["ok"] * count
    assert peak == 3
    assert runtime._subagent_coordinator.active_runs == 0
    assert not runtime._subagent_coordinator._slot_waiters


def test_task_fan_out_still_honors_max_agents():
    def respond(messages, info):
        return ModelResponse(parts=[TextPart("ok")])

    source = (
        "> profile: worker:\n  > instructions: Work.\n"
        "> profile: host:\n  > subagent: worker\n  > max_agents: 2\n"
        "> use: host\n"
        + "".join(f"> task [j{i}]: worker:\n    >> Say ok {i}.\n" for i in range(3))
        + "".join(f"> await [r{i}]: j{i}\n" for i in range(3))
        + "= `r0.task_summary`\n"
    )
    runtime = compile_program(
        parse_program(source), adapter=PydanticAdapter(FunctionModel(respond))
    )
    with pytest.raises(Exception, match="invocation limit 2"):
        _run(runtime)


def _counting_program(settings: str = "") -> str:
    return settings + "[doc] = `'word ' * 40000`\n[n: int] << Count the words in <<doc>>.\n= `n`\n"


def test_request_ceiling_names_the_context_limit_setting():
    def respond(messages, info):
        raise AssertionError("the request must be rejected before it is sent")

    runtime = compile_program(
        parse_program(_counting_program()), adapter=PydanticAdapter(FunctionModel(respond))
    )
    with pytest.raises(Exception) as error:
        _run(runtime)
    message = str(error.value)
    assert "safe context ceiling of 153616 tokens for a 200000-token context window" in message
    assert "set context_limit in > settings:" in message


def test_context_limit_setting_raises_the_ceiling_and_stays_local():
    settings_seen = []

    def respond(messages, info):
        settings_seen.append(info.model_settings)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"response": 40000})])

    runtime = compile_program(
        parse_program(_counting_program("> settings:\n  context_limit: 1000000\n")),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    assert _run(runtime) == 40000
    assert settings_seen and all(
        "context_limit" not in (settings or {}) for settings in settings_seen
    )


@pytest.mark.parametrize("value", ["1000", "lots", "0"])
def test_context_limit_rejects_values_below_a_usable_window(value):
    with pytest.raises(KediParseError, match="context_limit"):
        parse_program(_counting_program(f"> settings:\n  context_limit: {value}\n"))


def test_reserve_uses_the_requested_context_window():
    budget = ModelRequestBudget()
    prompt = "x" * 300_000
    with pytest.raises(ModelRequestBudgetError, match="200000-token context window"):
        budget.reserve(prompt)
    budget.reserve(prompt, context_limit_tokens=1_000_000)
    snapshot = budget.snapshot()
    assert snapshot.requests == 1
    assert snapshot.cumulative_input_tokens > 300_000


def test_slot_waiters_on_closed_loops_or_resolved_futures_are_skipped():
    from kedi.agent_adapter.subagents.coordinator import _resolve_slot_waiter

    runtime = compile_program(
        parse_program("= `1`\n"),
        adapter=PydanticAdapter(FunctionModel(lambda messages, info: None)),
    )
    coordinator = runtime._subagent_coordinator
    closed = asyncio.new_event_loop()
    pending = closed.create_future()
    closed.close()
    coordinator._slot_waiters.append((closed, pending))
    coordinator._wake_slot_waiters()
    assert not coordinator._slot_waiters
    assert not pending.done()

    loop = asyncio.new_event_loop()
    try:
        released = loop.create_future()
        released.set_result(None)
        _resolve_slot_waiter(released)
        assert released.result() is None
    finally:
        loop.close()
        asyncio.run(runtime.aclose())
