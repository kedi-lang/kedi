"""`> workflow: blocking` shows the parent only a synchronous delegate_task."""

import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import PydanticAdapter
from kedi.directive_values import DIRECTIVE_RULES
from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError
from tests.test_subagents import RecordingPydanticAdapter

CHILD = "> profile: worker:\n  > adapter: pydantic\n"


def _parent(workflow: str | None) -> str:
    mode = f"  > workflow: {workflow}\n" if workflow else ""
    return (
        CHILD
        + "> profile: manager:\n  > adapter: pydantic\n  > subagent: worker\n"
        + mode
        + "> use: manager\n"
    )


def _tools(workflow: str | None) -> dict:
    adapter = PydanticAdapter(TestModel())
    runtime = compile_program(parse_program(_parent(workflow)), adapter=adapter)
    profile = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
    return {spec.name: spec for spec in runtime._resolve_agent_tool_specs(profile, adapter)}


def test_default_keeps_the_delegation_and_lifecycle_tools():
    names = set(_tools(None))
    assert {"delegate_task", "continue_subagent", "wait_subagent"} <= names
    assert "background" in _tools(None)["delegate_task"].json_schema["properties"]


def test_explicit_delegate_is_the_same_as_the_default():
    assert set(_tools("delegate")) == set(_tools(None))


def test_blocking_exposes_only_a_synchronous_delegate_task():
    tools = _tools("blocking")
    lifecycle = {"continue_subagent", "wait_subagent", "cancel_subagent", "subagent_status"}
    assert "delegate_task" in tools
    assert not lifecycle & set(tools)
    assert "send_subagent_input" not in tools
    assert "background" not in tools["delegate_task"].json_schema["properties"]
    assert "wait_subagent" not in tools["delegate_task"].description


def test_blocking_delegation_still_validates_its_arguments():
    validator = _tools("blocking")["delegate_task"].argument_validator
    with pytest.raises(ValueError, match="Invalid arguments for delegate_task"):
        validator({"subagent": "worker", "task": "t", "background": True})


def test_unknown_modes_are_rejected_and_name_all_three_values():
    with pytest.raises(KediParseError, match="'delegate', 'dynamic' or 'blocking'"):
        parse_program(_parent("sequential"))


def test_the_editor_value_rule_offers_all_three_modes():
    assert DIRECTIVE_RULES["workflow"].options == ("delegate", "dynamic", "blocking")


@pytest.mark.parametrize(("workflow", "can_report"), [("delegate", True), ("blocking", False)])
def test_a_blocking_parent_gives_its_children_no_channel_back(workflow, can_report):
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(_parent(workflow)), adapter=adapter)

    async def scenario():
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="work")
            await job.wait()

    asyncio.run(scenario())
    child_tools = adapter.invocations[-1][2]
    assert ("send_parent_message" in child_tools) is can_report
