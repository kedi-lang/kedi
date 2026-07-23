from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from jsonschema.validators import validator_for
from typing_extensions import runtime_checkable

from ..tool_override import ToolSpec
from .attempts import (
    active_delegation_attempt_for_run,
    attempt_status_from_run,
    track_delegation_attempt,
)
from .budget import DelegationBudget
from .limits import SubagentUsageLimits

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile
    from kedi.agent_adapter.approval import ApprovalPolicy
    from kedi.core import KediRuntime


MAX_FINAL_SCHEMA_BYTES = 32 * 1024
MAX_FINAL_SCHEMA_DEPTH = 16


@dataclass(frozen=True, kw_only=True)
class DelegationResult:
    run_id: str
    subagent: str
    task_summary: str
    final_result: Any | None = None

    def as_tool_result(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentExecutionRequest:
    profile: AgentProfile
    tools: tuple[ToolSpec, ...]
    instructions: str | None
    approval_ceiling: tuple[ApprovalPolicy | None, ...]
    usage_limits: SubagentUsageLimits


@runtime_checkable
class SupportsSubagentExecution(Protocol):
    shortname: str

    def telemetry_model_name(self) -> str | None:
        """Return the model name used for telemetry when no profile override is active."""
        ...

    def subagent_scope(
        self,
        request: SubagentExecutionRequest,
    ) -> AbstractContextManager[None]:
        """Isolate one child profile, its tools, and inherited safety policy."""
        ...

    def subagent_run_kwargs(
        self,
        limits: SubagentUsageLimits,
    ) -> Mapping[str, Any]:
        """Translate generic child limits into adapter-native run arguments."""
        ...

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str: ...

    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Produce one JSON object matching an arbitrary validated JSON Schema."""
        ...


def subagent_tool_specs(
    rt: KediRuntime,
    profile: AgentProfile,
    *,
    supports_background: bool,
) -> tuple[ToolSpec, ...]:
    allowed = tuple(profile.subagents)
    descriptions = _subagent_descriptions(rt, allowed)
    budget = DelegationBudget.for_profile(
        profile_name=profile.name,
        max_agents=profile.max_agents,
    )
    concurrency = rt._new_subagent_concurrency()

    async def delegate_task(
        *,
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        with track_delegation_attempt(
            requested_profile=subagent,
            task=task,
        ) as attempt:
            try:
                if background:
                    if not supports_background:
                        raise ValueError("The active adapter does not support background subagents")
                    handle = rt._start_subagent(
                        allowed_subagents=allowed,
                        subagent=subagent,
                        task=task,
                        final_schema=final_schema,
                        parent_profile=profile,
                        parent_budget=budget,
                        parent_concurrency=concurrency,
                    )
                    return handle.as_tool_result()
                result = await rt._delegate_task(
                    allowed_subagents=allowed,
                    subagent=subagent,
                    task=task,
                    final_schema=final_schema,
                    parent_profile=profile,
                    parent_budget=budget,
                    parent_concurrency=concurrency,
                )
            except BaseException as exc:
                if attempt is not None:
                    status = "rejected"
                    if attempt.run_id is not None:
                        run = rt._subagent_coordinator.runs.get(attempt.run_id)
                        status = attempt_status_from_run(run.status)
                    attempt.fail(status=status, error=exc)
                raise
            if attempt is not None:
                attempt.complete()
            return result.as_tool_result()

    properties: dict[str, Any] = {
        "subagent": {
            "type": "string",
            "enum": list(allowed),
            "description": "Configured child profile that should perform the task.",
        },
        "task": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Self-contained task with all context and acceptance criteria needed by the child."
            ),
        },
        "final_schema": {
            "anyOf": [{"type": "object"}, {"type": "null"}],
            "default": None,
            "description": (
                "Optional JSON Schema for the child result. Use standard lowercase "
                "type values such as object, array, string, integer, number, boolean, "
                "and null."
            ),
        },
    }
    if supports_background:
        properties["background"] = {
            "type": "boolean",
            "default": False,
            "description": (
                "Start without blocking. Call wait_subagent with the returned run_id "
                "before using the result or returning a final answer."
            ),
        }

    tools = [
        ToolSpec(
            fn=delegate_task,
            name="delegate_task",
            description=(
                "Delegate one self-contained task to one direct subagent listed below. "
                "Choose the subagent by its description and include all context it needs; "
                "the child does not inherit the parent conversation. Use final_schema only "
                "when the parent needs a validated structured result."
                + (
                    " Set background=true only for work that can run concurrently, then call "
                    "wait_subagent before using its result or returning a final answer."
                    if supports_background
                    else ""
                )
                + descriptions
            ),
            json_schema={
                "type": "object",
                "properties": properties,
                "required": ["subagent", "task"],
                "additionalProperties": False,
            },
            return_schema=(
                {
                    "oneOf": [
                        _delegation_result_schema(),
                        _subagent_handle_schema(),
                    ]
                }
                if supports_background
                else _delegation_result_schema()
            ),
            metadata={"source": "kedi", "kind": "subagent"},
            # Child tools enforce the child profile's own approval policy. Spawning
            # an isolated child is not itself a filesystem or environment mutation.
            risk="read_only",
        )
    ]
    if not supports_background:
        return tuple(tools)

    async def wait_subagent(
        *,
        run_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        attempt = active_delegation_attempt_for_run(run_id)
        try:
            result = await rt._wait_subagent(run_id, timeout=timeout)
        except BaseException as exc:
            if attempt is not None:
                status = attempt_status_from_run(rt._subagent_status(run_id))
                attempt.fail(status=status, error=exc)
            raise
        if attempt is not None:
            attempt.complete()
        return result.as_tool_result()

    def cancel_subagent(*, run_id: str) -> dict[str, Any]:
        requested = rt._cancel_subagent(run_id)
        return {
            "run_id": run_id,
            "cancel_requested": requested,
            "status": rt._subagent_status(run_id),
        }

    def subagent_status(*, run_id: str) -> dict[str, str]:
        rt._subagent_status(run_id)
        run = rt._subagent_coordinator.runs.get(run_id)
        return {
            "run_id": run.run_id,
            "subagent": run.profile_name,
            "status": run.status,
        }

    tools.extend(
        (
            ToolSpec(
                fn=wait_subagent,
                name="wait_subagent",
                description=(
                    "Wait for a background subagent and return its task_summary and "
                    "final_result. Completed runs may be waited more than once."
                ),
                json_schema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "timeout": {
                            "anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}],
                            "default": None,
                        },
                    },
                    "required": ["run_id"],
                    "additionalProperties": False,
                },
                return_schema=_delegation_result_schema(),
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            ),
            ToolSpec(
                fn=cancel_subagent,
                name="cancel_subagent",
                description=(
                    "Request cancellation of a background subagent. Cancellation is "
                    "idempotent; call wait_subagent to observe terminal completion."
                ),
                json_schema=_run_id_schema(),
                return_schema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "cancel_requested": {"type": "boolean"},
                        "status": {"type": "string"},
                    },
                    "required": ["run_id", "cancel_requested", "status"],
                    "additionalProperties": False,
                },
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            ),
            ToolSpec(
                fn=subagent_status,
                name="subagent_status",
                description=(
                    "Inspect a background subagent's lifecycle status without waiting for it."
                ),
                json_schema=_run_id_schema(),
                return_schema=_subagent_handle_schema(),
                metadata={"source": "kedi", "kind": "subagent_lifecycle"},
                risk="read_only",
            ),
        )
    )
    return tuple(tools)


def _delegation_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "subagent": {"type": "string"},
            "task_summary": {"type": "string"},
            "final_result": {},
        },
        "required": ["run_id", "subagent", "task_summary", "final_result"],
        "additionalProperties": False,
    }


def _subagent_handle_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "subagent": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["run_id", "subagent", "status"],
        "additionalProperties": False,
    }


def _run_id_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
        "additionalProperties": False,
    }


def validate_final_schema(schema: dict[str, Any]) -> None:
    try:
        encoded = json.dumps(schema, ensure_ascii=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("final_schema must be JSON serializable") from exc
    if len(encoded) > MAX_FINAL_SCHEMA_BYTES:
        raise ValueError(f"final_schema exceeds the {MAX_FINAL_SCHEMA_BYTES}-byte size limit")
    _validate_schema_node(schema, depth=0)
    validator_for(schema).check_schema(schema)


def wrapped_final_schema(final_schema: dict[str, Any]) -> dict[str, Any]:
    validate_final_schema(final_schema)
    return {
        "type": "object",
        "title": "KediSubagentResult",
        "properties": {
            "task_summary": {
                "type": "string",
                "maxLength": 1000,
                "description": ("Briefly state what you did, key decisions, and any limitations."),
            },
            "final_result": final_schema,
        },
        "required": ["task_summary", "final_result"],
        "additionalProperties": False,
    }


def validate_structured_result(
    result: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError(
            f"Subagent structured output must be an object, got {type(result).__name__}"
        )
    validator_for(schema)(schema).validate(result)
    return result


def new_subagent_run_id() -> str:
    return uuid4().hex


def _subagent_descriptions(rt: KediRuntime, allowed: tuple[str, ...]) -> str:
    entries: list[str] = []
    for name in allowed:
        definition = rt._profile_registry.get(name)
        description = getattr(definition, "description", None)
        entries.append(f"- {name}: {description}" if description else f"- {name}")
    return "\n\nAvailable subagents:\n" + "\n".join(entries)


def _validate_schema_node(value: Any, *, depth: int) -> None:
    if depth > MAX_FINAL_SCHEMA_DEPTH:
        raise ValueError(
            f"final_schema exceeds the maximum nesting depth of {MAX_FINAL_SCHEMA_DEPTH}"
        )
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$id"}:
                raise ValueError(f"final_schema does not support {key}")
            _validate_schema_node(child, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _validate_schema_node(child, depth=depth + 1)
