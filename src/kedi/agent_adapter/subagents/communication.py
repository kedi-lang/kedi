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
    *,
    report_to_parent: bool = True,
) -> LiveInput:
    control = LiveInput(
        run_id,
        parent=current_live_input() if report_to_parent else None,
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
    by_parent: dict[str | None, list[str]] = {}
    for run in coordinator.runs.snapshot():
        by_parent.setdefault(run.parent_run_id, []).append(run.run_id)
    children: list[str] = []
    pending = list(by_parent.get(run_id, ()))
    while pending:
        child_id = pending.pop()
        children.append(child_id)
        pending.extend(by_parent.get(child_id, ()))
    with coordinator._task_lock:
        controls = tuple(
            control
            for child_id in children
            if (control := coordinator._controls.get(child_id)) is not None
        )
    first_error: Exception | None = None
    for child_id in children:
        try:
            coordinator.cancel(child_id)
        except Exception as exc:
            first_error = first_error or exc

    async def cleanup() -> None:
        await asyncio.gather(
            *(coordinator.wait(child_id) for child_id in children), return_exceptions=True
        )
        results = await asyncio.gather(
            *(control.wait_quiescent() for control in controls), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    try:
        await asyncio.wait_for(cleanup(), INTERRUPT_CLEANUP_SECONDS)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Subagent descendants did not stop; the parent task was not restarted"
        ) from exc
    if first_error is not None:
        raise first_error
