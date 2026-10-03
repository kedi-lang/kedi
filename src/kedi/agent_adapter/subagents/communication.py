"""Coordinator ownership checks and descendant teardown for live input."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from kedi.agent_adapter.meta import AdapterCapabilities

from .control import (
    INTERRUPT_CLEANUP_SECONDS,
    InputReceipt,
    LiveInput,
    current_live_input,
    sender_is_current,
)
from .runs import SubagentInputEvent

if TYPE_CHECKING:
    from .coordinator import SubagentCoordinator


def control_for_run(
    coordinator: SubagentCoordinator,
    run_id: str,
    capabilities: AdapterCapabilities | None,
) -> LiveInput:
    control = LiveInput(
        run_id,
        parent=current_live_input(),
        observe=lambda item, phase: coordinator.events.append(
            SubagentInputEvent(
                coordinator.runs.get(run_id),
                item.message_id,
                item.sender_id,
                item.revision,
                phase,
                item.interrupt,
            )
        ),
    )
    control.configure(
        input=bool(capabilities and capabilities.supports("live_subagent_input")),
        interrupt=bool(capabilities and capabilities.supports("subagent_interrupt")),
    )
    return control


def send_input(
    coordinator: SubagentCoordinator,
    run_id: str,
    message: str,
    *,
    interrupt: bool,
    sender: LiveInput | None,
) -> InputReceipt:
    with coordinator._task_lock:
        control = coordinator._controls.get(run_id)
    if control is None:
        return InputReceipt(None, run_id, 0, "rejected", "task_finished")
    sender = sender or current_live_input()
    if sender is None or not sender_is_current(sender) or control.parent is not sender:
        return control.reject("not_authorized")
    return control.admit(message, sender_id=sender.target_id, interrupt=interrupt)


async def stop_descendants(coordinator: SubagentCoordinator, run_id: str) -> None:
    with coordinator._task_lock:
        children = tuple(
            child_id
            for child_id in coordinator._tasks
            if coordinator.runs.get(child_id).parent_run_id == run_id
        )
    for child_id in children:
        coordinator.cancel(child_id)
    cleanup = asyncio.gather(
        *(coordinator.wait(child_id) for child_id in children), return_exceptions=True
    )
    try:
        await asyncio.wait_for(cleanup, INTERRUPT_CLEANUP_SECONDS)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Subagent descendants did not stop; the parent task was not restarted"
        ) from exc
    for child_id in children:
        control = coordinator._controls.get(child_id)
        if control is not None:
            await control.wait_quiescent()
