from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import KediExecutionError
from .profile import AgentProfile
from .registry import ProfileDefinition, ProfileModelMember

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import ModelDirectiveStmt, UseProfileStmt


class AgentProfileOverrideUnsupported(KediExecutionError):
    """Raised when the active adapter cannot apply a model profile override."""


def eval_model_value(
    rt: KediRuntime, *, plain_value: str | None, backtick_code: str | None, env: dict[str, Any]
) -> str:
    if backtick_code is not None:
        value = rt.executor.evaluate_type_expression(rt, backtick_code, env)
        if not isinstance(value, str):
            raise TypeError(f"model directive must evaluate to str, got {type(value).__name__}")
        return value
    if plain_value is not None:
        return plain_value.strip()
    raise ValueError("model directive has no value")


def eval_model_directive(
    rt: KediRuntime,
    stmt: ModelDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    model = eval_model_value(
        rt,
        plain_value=stmt.plain_value,
        backtick_code=stmt.backtick_code,
        env=env,
    )
    return AgentProfile(model=model)


def _member_to_patch(
    rt: KediRuntime,
    member: ProfileModelMember,
    env: dict[str, Any],
) -> AgentProfile:
    if member.kind == "model":
        model = eval_model_value(
            rt,
            plain_value=member.plain_value,
            backtick_code=member.backtick_code,
            env=env,
        )
        return AgentProfile(model=model)
    raise KediExecutionError(f"Unknown profile member kind: {member.kind!r}")


def eval_profile_definition(
    rt: KediRuntime,
    definition: ProfileDefinition,
    env: dict[str, Any],
) -> AgentProfile:
    profile = AgentProfile.empty()
    for member in definition.members:
        profile = profile.merge(_member_to_patch(rt, member, env))
    return profile


def eval_use_profile(
    rt: KediRuntime,
    stmt: UseProfileStmt,
    registry: dict[str, ProfileDefinition],
    env: dict[str, Any],
) -> AgentProfile:
    definition = registry.get(stmt.name)
    if definition is None:
        raise KediExecutionError(f"Unknown profile: {stmt.name!r}")
    return eval_profile_definition(rt, definition, env)
