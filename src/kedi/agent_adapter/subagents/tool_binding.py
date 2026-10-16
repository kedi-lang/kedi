"""Bind delegation tools to their coordinator and scoped execution policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile.activation import require_subagent_activation

from . import DelegationResult
from .attempts import (
    active_delegation_attempt_for_run,
    attempt_status_from_run,
    track_delegation_attempt,
)
from .budget import DelegationBudget, DelegationConcurrency
from .tool_schemas import build_subagent_tool_specs

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile
    from .coordinator import SubagentCoordinator


def bind_subagent_tools(
    coordinator: SubagentCoordinator,
    profile: AgentProfile,
    *,
    activation_profile: AgentProfile,
    budget: DelegationBudget,
    concurrency: DelegationConcurrency,
    catalog: Mapping[str, str | None],
    supports_background: bool,
) -> tuple[ToolSpec, ...]:
    allowed = tuple(profile.subagents)
    subagent_definitions = dict(profile.subagent_bindings)

    async def delegate_task(
        *,
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        require_subagent_activation(activation_profile, subagent)
        with track_delegation_attempt(
            requested_profile=subagent,
            task=task,
        ) as attempt:
            try:
                if background:
                    if not supports_background:
                        raise ValueError("The active adapter does not support background subagents")
                    handle = coordinator.start(
                        allowed_subagents=allowed,
                        subagent=subagent,
                        task=task,
                        final_schema=final_schema,
                        parent_profile=profile,
                        parent_budget=budget,
                        parent_concurrency=concurrency,
                        subagent_definitions=subagent_definitions,
                    )
                    return handle.as_tool_result()
                result = await coordinator.delegate(
                    allowed_subagents=allowed,
                    subagent=subagent,
                    task=task,
                    final_schema=final_schema,
                    parent_profile=profile,
                    parent_budget=budget,
                    parent_concurrency=concurrency,
                    subagent_definitions=subagent_definitions,
                )
            except BaseException as exc:
                if attempt is not None:
                    if attempt.run_id is None:
                        attempt.fail(status="rejected", error=exc)
                    else:
                        run_status = coordinator.runs.get(attempt.run_id).status
                        if run_status != "running":
                            attempt.fail(
                                status=attempt_status_from_run(run_status),
                                error=exc,
                            )
                raise
            if attempt is not None:
                attempt.complete()
            return result.as_tool_result()

    async def continue_subagent(
        *,
        run_id: str,
        task: str,
        final_schema: dict[str, Any] | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        run = coordinator.runs.get(run_id)
        require_subagent_activation(activation_profile, run.profile_name)
        if background:
            if not supports_background:
                raise ValueError("The active adapter does not support background subagents")
            handle = coordinator.start_continuation(
                run_id=run_id,
                allowed_subagents=allowed,
                task=task,
                final_schema=final_schema,
                parent_profile=profile,
                parent_budget=budget,
                parent_concurrency=concurrency,
                subagent_definitions=subagent_definitions,
            )
            return handle.as_tool_result()
        result = await coordinator.continue_run(
            run_id=run_id,
            allowed_subagents=allowed,
            task=task,
            final_schema=final_schema,
            parent_profile=profile,
            parent_budget=budget,
            parent_concurrency=concurrency,
            subagent_definitions=subagent_definitions,
        )
        return result.as_tool_result()

    async def wait_subagent(
        *,
        run_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        attempt = active_delegation_attempt_for_run(run_id)
        try:
            from .messaging import wait_for_child_call

            result = await wait_for_child_call(coordinator, coordinator.wait, run_id, timeout)
            if not isinstance(result, DelegationResult):
                return dict(result)
        except TimeoutError:
            status = coordinator.status(run_id)
            if status == "running":
                run = coordinator.runs.get(run_id)
                return {
                    "run_id": run.run_id,
                    "subagent": run.profile_name,
                    "status": status,
                }
            raise
        except BaseException as exc:
            if attempt is not None:
                status = coordinator.status(run_id)
                # A waiter timeout or cancellation is local to this caller. The
                # child remains observable until it reaches a terminal status.
                if status != "running":
                    attempt.fail(status=attempt_status_from_run(status), error=exc)
            raise
        if attempt is not None:
            attempt.complete()
        return result.as_tool_result()

    def cancel_subagent(*, run_id: str) -> dict[str, Any]:
        requested = coordinator.cancel(run_id)
        return {
            "run_id": run_id,
            "cancel_requested": requested,
            "status": coordinator.status(run_id),
        }

    def subagent_status(*, run_id: str) -> dict[str, str]:
        coordinator.status(run_id)
        run = coordinator.runs.get(run_id)
        return {
            "run_id": run.run_id,
            "subagent": run.profile_name,
            "status": run.status,
        }

    return build_subagent_tool_specs(
        allowed=allowed,
        catalog=catalog,
        supports_background=supports_background,
        delegate_task=delegate_task,
        continue_subagent=continue_subagent,
        wait_subagent=wait_subagent,
        cancel_subagent=cancel_subagent,
        subagent_status=subagent_status,
    )
