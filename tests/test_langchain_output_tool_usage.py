"""Structured-output protocol calls are not tool calls (#230).

LangChain returns typed answers as output-function calls in the same `tool_calls` list as
business tools. Only admitted tools count, so the reported usage, the coordinator's child
records and LangChain's native budget agree, as they do on Pydantic AI.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters import LangChainAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.lang import compile_program, parse_program
from tests.test_langchain_branch_parity import ScriptedModel

CHILD = """~Status(status: str)

@lookup(order: str) -> str:
  ###
  Look up an order.
  ###
  > tool:
    risk: read_only
  = `"delivered"`

> profile: worker:
    > output: Status
    > use: lookup
> profile: coordinator:
    > subagent: worker
> use: coordinator
> task [job]: worker:
    >> Report the status of order O-1.
> await [result]: job
= `result.output.status`
"""

ENVELOPE = {"task_summary": "Checked.", "final_result": {"status": "delivered"}}


def _envelope_call(index: int) -> dict[str, Any]:
    return {
        "name": "KediSubagentResult",
        "args": ENVELOPE,
        "id": f"out-{index}",
        "type": "tool_call",
    }


def _lookup_call() -> dict[str, Any]:
    return {"name": "lookup", "args": {"order": "O-1"}, "id": "t-1", "type": "tool_call"}


def _run(source: str, adapter: Any, *, tool_calls_limit: int = 10) -> tuple[Any, ...]:
    async def run() -> tuple[Any, ...]:
        usage: list[InvocationUsage] = []
        with use_async_owner_loop(), observe_invocations(usage.append):
            runtime = compile_program(
                parse_program(source),
                adapter=adapter,
                subagent_usage_limits=SubagentUsageLimits(
                    request_limit=3, tool_calls_limit=tool_calls_limit
                ),
            )
            try:
                value = await asyncio.to_thread(runtime.run_main)
                children = runtime._subagent_coordinator.runs.snapshot()
            finally:
                await runtime.aclose()
        return value, usage, children

    return asyncio.run(run())


def _reported(usage: list[InvocationUsage]) -> int:
    return sum(item.tool_calls for item in usage)


def test_a_typed_answer_without_tools_reports_no_tool_call() -> None:
    model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[_envelope_call(0)])])
    value, usage, [child] = _run(CHILD, LangChainAdapter(model), tool_calls_limit=1)

    assert value == "delivered"
    assert _reported(usage) == 0
    assert (child.requests, child.tool_calls) == (1, 0)


def test_equivalent_duplicate_answers_do_not_use_the_tool_budget() -> None:
    calls = [_envelope_call(0), _envelope_call(1)]
    model = ScriptedModel(responses=[AIMessage(content="", tool_calls=calls)])
    value, usage, [child] = _run(CHILD, LangChainAdapter(model), tool_calls_limit=1)

    assert value == "delivered"
    assert _reported(usage) == 0
    assert (child.requests, child.tool_calls) == (1, 0)


def test_one_real_tool_then_a_typed_answer_reports_exactly_one_tool_call() -> None:
    model = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[_lookup_call()]),
            AIMessage(content="", tool_calls=[_envelope_call(0)]),
        ]
    )
    value, usage, [child] = _run(CHILD, LangChainAdapter(model), tool_calls_limit=1)

    assert value == "delivered"
    assert _reported(usage) == 1
    assert (child.requests, child.tool_calls) == (2, 1)
    tool_results = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_results] == ["delivered"]


def test_pydantic_reports_the_same_child_tool_calls() -> None:
    def answer(with_tool: bool) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            returned = any(
                isinstance(part, ToolReturnPart)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            if with_tool and not returned:
                return ModelResponse(parts=[ToolCallPart("lookup", {"order": "O-1"}, "t-1")])
            [output] = info.output_tools
            return ModelResponse(parts=[ToolCallPart(output.name, ENVELOPE, "out-0")])

        return FunctionModel(respond)

    for with_tool, expected in ((False, (1, 0)), (True, (2, 1))):
        value, usage, [child] = _run(CHILD, PydanticAdapter(answer(with_tool)), tool_calls_limit=1)
        assert value == "delivered"
        assert _reported(usage) == expected[1]
        assert (child.requests, child.tool_calls) == expected


class OutputToolModel(BaseChatModel):
    """Answer by calling the bound tool whose schema is the capture, whatever its name."""

    bound: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "output-tool"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OutputToolModel:
        del kwargs
        functions = [convert_to_openai_tool(tool)["function"] for tool in tools]
        self.bound = {f["name"]: f.get("parameters", {}) for f in functions}
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **_: Any
    ) -> ChatResult:
        [output] = [
            name for name, schema in self.bound.items() if "count" in schema.get("properties", {})
        ]
        call = {"name": output, "args": {"count": 4}, "id": "out-0", "type": "tool_call"}
        message = AIMessage(content="", tool_calls=[call])
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_an_ordinary_typed_capture_reports_no_tool_call_whatever_its_output_name() -> None:
    model = OutputToolModel()
    value, usage, children = _run(
        ">> A dog has [count: int] legs.\n= `count`\n", LangChainAdapter(model)
    )

    assert value == 4
    assert children == ()
    assert "KediSubagentResult" not in model.bound
    assert _reported(usage) == 0
