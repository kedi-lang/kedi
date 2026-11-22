"""#222: when a `run_workflow` result carries a typed `final_result`.

Two ways a child can return a typed `final_result`: the call passed an
explicit `final_schema`, or the child's profile configures an output type
(`> output:`). Only a child with neither has `final_result` set to `None`,
with the answer in `task_summary`. The workflow tool description must say
so; the type-check stub keeps `final_result: Any` because Monty rejects
subscripting `Any | None`. The return semantics themselves do not change.
"""

import asyncio

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.subagents.workflow import _type_check_stubs, _workflow_description
from kedi.lang import compile_program, parse_program
from tests.test_langchain_branch_parity import ScriptedModel

CODE = (
    'r = await worker(task="Report the status of order O-1.")\n'
    '{"O-1": r["final_result"], "summary": r["task_summary"]}'
)

PROGRAM = (
    "> profile: worker:\n  > instructions: Answer the task.\n"
    "> profile: host:\n  > subagent: worker\n  > workflow: dynamic\n  > max_agents: 2\n"
    "> use: host\n[reply] << Get the status of O-1 from the worker.\n= <reply>\n"
)

PROGRAM_TYPED = (
    "> profile: worker:\n  > instructions: Answer the task.\n  > output: dict[str, str]\n"
    "> profile: host:\n  > subagent: worker\n  > workflow: dynamic\n  > max_agents: 2\n"
    "> use: host\n[reply] << Get the status of O-1 from the worker.\n= <reply>\n"
)

TYPED_RESULT = {"task_summary": "Order O-1 is delivered.", "final_result": {"O-1": "delivered"}}

EXPECTED_PYDANTIC = "{'O-1': None, 'summary': 'Order O-1 is delivered.'}"
EXPECTED_LANGCHAIN = '\'{"O-1": null, "summary": "Order O-1 is delivered."}\''
EXPECTED_TYPED_PYDANTIC = "{'O-1': {'O-1': 'delivered'}, 'summary': 'Order O-1 is delivered.'}"
EXPECTED_TYPED_LANGCHAIN = '\'{"O-1": {"O-1": "delivered"}, "summary": "Order O-1 is delivered."}\''


def test_workflow_description_states_when_final_result_is_typed() -> None:
    description = " ".join(_workflow_description({"worker": None}, remaining=1).split())
    assert (
        "`final_result` is the child's typed result when the call passed `final_schema` "
        "or the child's profile configures an output type" in description
    )
    assert "otherwise it is `None` and the child's answer is in `task_summary`" in description


def test_workflow_stub_keeps_final_result_subscriptable() -> None:
    """The stub keeps `Any`: Monty rejects subscripting `Any | None`.

    Workflow code reads `result['final_result']['value']`; typing the stub
    field `Any | None` makes that a type error even when a final_schema was
    passed, so the prose description carries the None case instead.
    """
    stubs = _type_check_stubs({"worker": None})
    assert "final_result: Any" in stubs
    assert "final_result: Any | None" not in stubs


def test_workflow_final_result_is_none_without_schema_or_profile_output_type_pydantic() -> None:
    def respond(messages, info):
        if "run_workflow" not in [t.name for t in info.function_tools]:
            return ModelResponse(parts=[TextPart("Order O-1 is delivered.")])
        returns = [
            p.content
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("run_workflow", {"code": CODE})])
        return ModelResponse(parts=[TextPart(repr(returns[-1]))])

    runtime = compile_program(
        parse_program(PROGRAM), adapter=PydanticAdapter(FunctionModel(respond))
    )
    try:
        assert runtime.run_main() == EXPECTED_PYDANTIC
    finally:
        asyncio.run(runtime.aclose())


def test_workflow_final_result_is_none_without_schema_or_profile_output_type_langchain() -> None:
    class Model(ScriptedModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            child = "Report the status" in str(messages[:2])
            if child:
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content="Order O-1 is delivered."))
                    ]
                )
            if tools:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=repr(tools[-1].content)))]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "run_workflow",
                                    "args": {"code": CODE},
                                    "id": "c1",
                                }
                            ],
                        )
                    )
                ]
            )

    runtime = compile_program(parse_program(PROGRAM), adapter=LangChainAdapter(Model(responses=[])))
    try:
        assert runtime.run_main() == EXPECTED_LANGCHAIN
    finally:
        asyncio.run(runtime.aclose())


def test_workflow_final_result_is_typed_from_profile_output_type_pydantic() -> None:
    def respond(messages, info):
        if info.output_tools:
            (output_tool,) = info.output_tools
            return ModelResponse(parts=[ToolCallPart(output_tool.name, TYPED_RESULT)])
        if "run_workflow" not in [t.name for t in info.function_tools]:
            return ModelResponse(parts=[TextPart("Order O-1 is delivered.")])
        returns = [
            p.content
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("run_workflow", {"code": CODE})])
        return ModelResponse(parts=[TextPart(repr(returns[-1]))])

    runtime = compile_program(
        parse_program(PROGRAM_TYPED), adapter=PydanticAdapter(FunctionModel(respond))
    )
    try:
        assert runtime.run_main() == EXPECTED_TYPED_PYDANTIC
    finally:
        asyncio.run(runtime.aclose())


def test_workflow_final_result_is_typed_from_profile_output_type_langchain() -> None:
    class Model(ScriptedModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            child = "Report the status" in str(messages[:2])
            if child:
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "KediSubagentResult",
                                        "args": TYPED_RESULT,
                                        "id": "c1",
                                    }
                                ],
                            )
                        )
                    ]
                )
            if tools:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=repr(tools[-1].content)))]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "run_workflow",
                                    "args": {"code": CODE},
                                    "id": "c2",
                                }
                            ],
                        )
                    )
                ]
            )

    runtime = compile_program(
        parse_program(PROGRAM_TYPED), adapter=LangChainAdapter(Model(responses=[]))
    )
    try:
        assert runtime.run_main() == EXPECTED_TYPED_LANGCHAIN
    finally:
        asyncio.run(runtime.aclose())
