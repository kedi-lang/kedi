"""Negative capability checks intentionally perform no model or transport I/O."""

import asyncio
from types import SimpleNamespace

import pytest

from kedi.agent_adapter.adapters.a2a import A2AAdapter
from kedi.agent_adapter.adapters.acp import ACPAdapter
from kedi.agent_adapter.dspy_metadata import DSPY_ADAPTER
from kedi.agent_adapter.subagents.communication import control_for_run
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


@pytest.mark.parametrize(
    "capabilities",
    [
        A2AAdapter.capabilities,
        ACPAdapter.capabilities,
        DSPY_ADAPTER.capabilities,
    ],
    ids=["a2a", "acp", "dspy"],
)
@pytest.mark.parametrize("interrupt", [False, True])
def test_actual_unsupported_lane_rejects_input_without_queue_or_revision(capabilities, interrupt):
    events = []
    coordinator = SimpleNamespace(events=SimpleNamespace(append=events.append))
    control = control_for_run(coordinator, "child", capabilities)
    receipt = control.admit("Updated task context", sender_id="parent", interrupt=interrupt)
    assert receipt.status == "rejected"
    assert receipt.reason == "unsupported"
    assert control.revision == 0
    assert not control.waiting_input
    assert not events


def test_actual_acp_parent_rejects_background_before_process_start(monkeypatch):
    adapter = ACPAdapter(command=["this-command-must-not-be-executed"])

    async def forbidden(**kwargs):
        pytest.fail("Unsupported ACP parent must not perform child I/O")

    monkeypatch.setattr(adapter, "invoke", forbidden)
    runtime = compile_program(
        parse_program(
            "> profile: leaf:\n  > instructions: Read the task.\n"
            "> profile: manager:\n  > subagent: leaf\n"
        ),
        adapter=adapter,
    )

    async def scenario():
        async with runtime.subagents(parent="manager") as scope:
            with pytest.raises(KediExecutionError, match="background subagents"):
                await scope.start("leaf", task="Must not start")
        assert not runtime._subagent_coordinator.runs.snapshot()
        await runtime.aclose()

    asyncio.run(scenario())
