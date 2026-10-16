from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from jsonschema.validators import validator_for
from pydantic import TypeAdapter
from typing_extensions import runtime_checkable

from ...errors import KediExecutionError
from ..tools.specs import ToolSpec
from .budget import DelegationBudget
from .limits import SubagentUsageLimits

if TYPE_CHECKING:
    from kedi.agent_adapter.approvals.policy import ApprovalPolicy
    from kedi.agent_profile import AgentProfile
    from kedi.core import KediRuntime


MAX_FINAL_SCHEMA_BYTES = 32 * 1024
MAX_FINAL_SCHEMA_DEPTH = 16


def effective_final_schema(
    profile: AgentProfile,
    explicit_schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return an explicit delegation schema or the child profile's output schema."""
    if explicit_schema is not None:
        return explicit_schema
    if profile.output_type is None:
        return None
    try:
        schema = TypeAdapter(profile.output_type).json_schema()
    except Exception as exc:
        raise KediExecutionError(
            f"Subagent profile {profile.name!r} output type cannot be represented as JSON Schema: "
            f"{exc}"
        ) from exc
    validate_final_schema(schema)
    return schema


@dataclass(frozen=True, kw_only=True)
class DelegationResult:
    run_id: str
    subagent: str
    task_summary: str
    final_result: Any | None = None

    def as_tool_result(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentTurn:
    run_id: str
    task: str
    task_summary: str
    final_result: Any | None = None


@dataclass(slots=True, kw_only=True)
class SubagentConversationContext:
    conversation_id: str
    turns: tuple[SubagentTurn, ...] = ()
    resume_state: dict[str, Any] = field(default_factory=dict)
    result_state: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        return "kedi-subagent-" + self.conversation_id


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentExecutionRequest:
    profile: AgentProfile
    tools: tuple[ToolSpec, ...]
    instructions: str | None
    approval_ceiling: tuple[ApprovalPolicy | None, ...]
    usage_limits: SubagentUsageLimits
    conversation: SubagentConversationContext = field(
        default_factory=lambda: SubagentConversationContext(
            conversation_id=uuid4().hex,
            turns=(),
        )
    )


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
    activation_profile: AgentProfile | None = None,
) -> tuple[ToolSpec, ...]:
    """Resolve runtime-owned dependencies before binding the model-facing tools."""
    from .tool_binding import bind_subagent_tools

    native_owner = rt._native_subagent_owner.get()
    budget = (
        native_owner.budget_for(profile)
        if native_owner is not None
        else DelegationBudget.for_profile(profile_name=profile.name, max_agents=profile.max_agents)
    )
    catalog = {
        name: getattr(
            profile.subagent_bindings.get(name) or rt._profile_registry.get(name),
            "description",
            None,
        )
        for name in profile.subagents
    }
    return bind_subagent_tools(
        rt._subagent_coordinator,
        profile,
        activation_profile=profile if activation_profile is None else activation_profile,
        budget=budget,
        concurrency=native_owner.concurrency
        if native_owner is not None
        else rt._new_subagent_concurrency(),
        catalog=catalog,
        supports_background=supports_background,
    )


def validate_final_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        raise ValueError(
            "final_schema must be a JSON object, not a JSON-encoded string or another value. "
            "Omit final_schema to use the child profile's configured output type."
        )
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
