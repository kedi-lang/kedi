"""Every failure of a subagent tool reaches the calling model; only budgets and cancellation end the run."""

import asyncio
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded

from kedi.agent_adapter.adapters.pydantic.tools import _validated_tool_function
from kedi.agent_adapter.subagents.tool_schemas import build_subagent_tool_specs
from kedi.agent_adapter.tools.errors import correctable_tool_error
from kedi.run_budget import RunBudgetExceeded


def _specs(**overrides: Any) -> dict[str, Any]:
    async def unused(**_: Any) -> dict[str, str]:
        return {}

    functions = {
        name: unused
        for name in (
            "delegate_task",
            "continue_subagent",
            "wait_subagent",
            "cancel_subagent",
            "subagent_status",
        )
    }
    functions.update(overrides)
    specs = build_subagent_tool_specs(
        allowed=("worker",), catalog={"worker": "Does work."}, supports_background=True, **functions
    )
    return {spec.name: spec for spec in specs}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"subagent": "worker", "parameters": {"x": 1}}, ["task", "parameters"]),
        ({"subagent": "worker"}, ["task"]),
        ({"subagent": "other", "task": "t"}, ["subagent"]),
        ({"subagent": "worker", "task": ""}, ["task"]),
    ],
)
def test_malformed_arguments_are_described_to_the_model(arguments, expected):
    spec = _specs()["delegate_task"]
    with pytest.raises(ValueError, match="Invalid arguments for delegate_task") as raised:
        spec.argument_validator(arguments)
    assert all(word in str(raised.value) for word in expected)
    assert "Allowed arguments:" in str(raised.value)


def test_valid_arguments_pass_through_unchanged():
    arguments = {"subagent": "worker", "task": "do it", "background": True}
    assert _specs()["delegate_task"].argument_validator(arguments) == arguments


@pytest.mark.parametrize(
    "name", ["continue_subagent", "wait_subagent", "cancel_subagent", "subagent_status"]
)
def test_every_subagent_tool_validates_its_schema(name):
    with pytest.raises(ValueError, match=f"Invalid arguments for {name}"):
        _specs()[name].argument_validator({"unexpected": 1})


@pytest.mark.parametrize(
    "error",
    [
        TypeError("bad"),
        ValueError("bad"),
        KeyError("bad"),
        UsageLimitExceeded("request_limit of 8"),
    ],
)
def test_any_ordinary_failure_is_correctable(error):
    message = correctable_tool_error(_specs()["delegate_task"], error)
    assert message is not None and type(error).__name__ in message


def test_run_budgets_still_end_the_run():
    spec = _specs()["delegate_task"]
    assert correctable_tool_error(spec, RunBudgetExceeded("requests", 8, 8, 1)) is None


@pytest.mark.parametrize("name", ["delegate_task", "wait_subagent"])
def test_pydantic_async_tool_returns_failures_to_the_model(name):
    async def failing(**_: Any):
        raise UsageLimitExceeded("The next request would exceed the request_limit of 8")

    tool = _validated_tool_function(_specs(**{name: failing})[name])
    arguments = {
        "delegate_task": {"subagent": "worker", "task": "work"},
        "wait_subagent": {"run_id": "abc"},
    }[name]
    with pytest.raises(ModelRetry, match="UsageLimitExceeded"):
        asyncio.run(tool(**arguments))
    with pytest.raises(ModelRetry, match="Invalid arguments"):
        asyncio.run(tool(parameters={}))


def test_pydantic_cancellation_is_not_swallowed():
    async def cancelled(**_: Any):
        raise asyncio.CancelledError

    tool = _validated_tool_function(_specs(delegate_task=cancelled)["delegate_task"])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(tool(subagent="worker", task="work"))


def test_langchain_tools_return_failures_to_the_model():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import ToolException

    from kedi.agent_adapter.adapters.langchain.tools import (
        _langchain_async_tool_function,
        _langchain_sync_tool_function,
    )

    async def failing(**_: Any):
        raise TypeError("delegate_task() got an unexpected keyword argument 'parameters'")

    spec = _specs(delegate_task=failing)["delegate_task"]
    with pytest.raises(ToolException, match="TypeError"):
        asyncio.run(_langchain_async_tool_function(spec)(subagent="worker", task="work"))
    with pytest.raises(ToolException, match="Invalid arguments"):
        asyncio.run(_langchain_async_tool_function(spec)(parameters={}))
    sync_spec = _specs(delegate_task=lambda **_: (_ for _ in ()).throw(ValueError("boom")))[
        "delegate_task"
    ]
    with pytest.raises(ToolException, match="ValueError"):
        _langchain_sync_tool_function(sync_spec)(subagent="worker", task="work")
