"""Malformed outer workflow arguments must not execute code or abort model repair."""

import asyncio

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters.langchain.tools import _langchain_tool_from_spec
from kedi.agent_adapter.adapters.pydantic.tools import pydantic_tool_from_spec
from kedi.agent_profile.compile import build_capture_env
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.lang import compile_program, parse_program
from tests.retry_feedback import retry_feedback


@pytest.fixture
def workflow_tool():
    adapter = PydanticAdapter(TestModel())
    runtime = compile_program(
        parse_program(
            "> profile: worker:\n"
            "  > instructions: Complete your assigned task.\n"
            "> profile: manager:\n"
            "  > subagent: worker\n"
            "  > workflow: dynamic\n"
            "  > max_agents: 1\n"
        ),
        adapter=adapter,
    )
    profile = eval_profile_definition(
        runtime, runtime._profile_registry["manager"], build_capture_env(runtime)
    )
    tool = next(
        t for t in runtime._resolve_agent_tool_specs(profile, adapter) if t.name == "run_workflow"
    )
    try:
        yield runtime, tool
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize(
    "arguments", [{}, {"code": "42", "final_schema": {}}, {"code": 42}, {"code": " "}]
)
def test_pydantic_workflow_wrong_arguments_are_repairable_before_execution(
    workflow_tool, arguments
):
    runtime, spec = workflow_tool
    tool = pydantic_tool_from_spec(spec)

    async def run():
        with pytest.raises(ModelRetry):
            await tool.function(**arguments)
        assert await tool.function(code="40 + 2") == 42

    asyncio.run(run())
    assert runtime._subagent_coordinator.runs.snapshot() == ()


@pytest.mark.parametrize(
    "arguments", [{}, {"code": "42", "final_schema": {}}, {"code": 42}, {"code": " "}]
)
def test_langchain_workflow_wrong_arguments_are_repairable_before_execution(
    workflow_tool, arguments
):
    runtime, spec = workflow_tool
    tool = _langchain_tool_from_spec(spec)

    async def run():
        result = await tool.ainvoke(arguments)
        assert isinstance(result, str) and ("code" in result or "workflow" in result)
        assert await tool.ainvoke({"code": "40 + 2"}) == 42

    asyncio.run(run())
    assert runtime._subagent_coordinator.runs.snapshot() == ()


def test_native_pydantic_model_can_repair_outer_workflow_arguments(workflow_tool):
    runtime, spec = workflow_tool
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("run_workflow", {"code": "40 + 2", "final_schema": {}})]
            )
        if calls == 2:
            assert retry_feedback(messages)
            return ModelResponse(parts=[ToolCallPart("run_workflow", {"code": "40 + 2"})])
        return ModelResponse(parts=[TextPart("42")])

    agent = Agent(FunctionModel(respond), tools=[pydantic_tool_from_spec(spec)], retries=1)
    assert agent.run_sync("Execute the workflow").output == "42"
    assert calls == 3
    assert runtime._subagent_coordinator.runs.snapshot() == ()
