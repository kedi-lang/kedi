"""Common model-facing messaging tools and interruptible parent waits."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, TypedDict

from pydantic import JsonValue

from kedi.agent_adapter.subagents.control import (
    InputReceipt,
    InputReceiptResult,
    current_live_input,
    sender_is_current,
)
from kedi.agent_adapter.subagents.runs import SubagentRunStatus
from kedi.agent_adapter.tools.specs import ToolSpec

if TYPE_CHECKING:
    from kedi.agent_adapter.subagents import DelegationResult
    from kedi.core.runtime import KediRuntime


class ChildInputAvailable(TypedDict):
    run_id: str
    subagent: str
    status: SubagentRunStatus
    input_available: Literal[True]


RECEIPT_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "message_id": {"type": ["string", "null"]},
        "target_id": {"type": "string"},
        "revision": {"type": "integer"},
        "status": {"type": "string", "enum": ["queued", "interrupt_requested", "rejected"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["message_id", "target_id", "revision", "status", "reason"],
    "additionalProperties": False,
}


def input_tool_specs(rt: KediRuntime, *, children: bool, parent: bool) -> tuple[ToolSpec, ...]:
    async def send_subagent_input(
        *, task_id: str, message: str, interrupt: bool = False
    ) -> InputReceiptResult:
        return rt._subagent_coordinator.send_input(
            task_id, message, interrupt=interrupt
        ).as_tool_result()

    async def send_parent_message(*, message: str) -> InputReceiptResult:
        control = current_live_input()
        if control is None or control.parent is None or not sender_is_current(control):
            return InputReceipt(None, "parent", 0, "rejected", "not_authorized").as_tool_result()
        return control.parent.admit(message, sender_id=control.target_id).as_tool_result()

    message_schema = {"type": "string", "minLength": 1}
    tools: list[ToolSpec] = []
    if children:
        tools.append(
            ToolSpec(
                fn=send_subagent_input,
                name="send_subagent_input",
                description=(
                    "Send context to your direct child's running task. With interrupt=false, "
                    "deliver at its next model-input boundary without cancelling active work. "
                    "With interrupt=true, stop the current attempt and continue the same task "
                    "and conversation with updated scope. A queued receipt is not confirmation "
                    "that the child read or understood the message. Finished tasks reject input."
                ),
                json_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": (
                                "Use the exact run_id returned by delegate_task, or sender_id "
                                "of the child's input message. Never use that message's "
                                "message_id or target_id, which identifies its recipient."
                            ),
                        },
                        "message": message_schema,
                        "interrupt": {"type": "boolean", "default": False},
                    },
                    "required": ["task_id", "message"],
                    "additionalProperties": False,
                },
                return_schema=RECEIPT_SCHEMA,
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            )
        )
    if parent:
        tools.append(
            ToolSpec(
                fn=send_parent_message,
                name="send_parent_message",
                description=(
                    "Send a progress update or question to your direct parent. The parent receives "
                    "it at its next safe input boundary; this does not pause your task or guarantee "
                    "a reply. A queued receipt is not confirmation of delivery or understanding."
                ),
                json_schema={
                    "type": "object",
                    "properties": {"message": message_schema},
                    "required": ["message"],
                    "additionalProperties": False,
                },
                return_schema=RECEIPT_SCHEMA,
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            )
        )
    return tuple(tools)


async def wait_for_child(
    rt: KediRuntime, run_id: str, timeout: float | None
) -> DelegationResult | ChildInputAvailable:
    control = current_live_input()
    if control is None:
        return await rt._wait_subagent(run_id, timeout=timeout)
    input_sequence = control.input_sequence

    async def wait_input() -> None:
        while True:
            sequence = control.sequence
            if (
                control.input_sequence != input_sequence
                or control.waiting_input
                or control.finished
            ):
                return
            await control.wait_changed(sequence)

    waiter = asyncio.create_task(rt._wait_subagent(run_id, timeout=timeout))
    notification = asyncio.create_task(wait_input())
    try:
        if not control.waiting_input:
            done, _ = await asyncio.wait(
                (waiter, notification), return_when=asyncio.FIRST_COMPLETED
            )
            if waiter in done:
                return waiter.result()
        run = rt._subagent_coordinator.runs.get(run_id)
        return {
            "run_id": run_id,
            "subagent": run.profile_name,
            "status": run.status,
            "input_available": True,
        }
    finally:
        waiter.cancel()
        notification.cancel()
        await asyncio.gather(waiter, notification, return_exceptions=True)
