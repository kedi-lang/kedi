from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..effort import EffortValue, validate_effort
from ..errors import KediExecutionError
from .profile import AgentProfile
from .registry import (
    ProfileDefinition,
    ProfileEffortMember,
    ProfileModelMember,
    ProfileSystemMember,
    ProfileToolMember,
)

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import (
        EffortDirectiveStmt,
        ModelDirectiveStmt,
        SystemDirectiveStmt,
        TemplateLine,
        UseDirectiveStmt,
    )


class AgentProfileOverrideUnsupported(KediExecutionError):
    """Raised when the active adapter cannot apply an agent profile override."""


class AgentToolOverrideUnsupported(KediExecutionError):
    """Raised when the active adapter cannot register tools for an agent run."""


def eval_model_value(
    rt: KediRuntime,
    *,
    directive_name: str = "model",
    plain_value: str | None,
    backtick_code: str | None,
    env: dict[str, Any],
) -> str:
    if backtick_code is not None:
        value = rt.executor.evaluate_type_expression(rt, backtick_code, env)
        if not isinstance(value, str):
            raise TypeError(
                f"{directive_name} directive must evaluate to str, got {type(value).__name__}"
            )
        return value
    if plain_value is not None:
        return plain_value.strip()
    raise ValueError(f"{directive_name} directive has no value")


def eval_system_value(
    rt: KediRuntime,
    *,
    plain_value: str | None,
    backtick_code: str | None,
    template_lines: tuple[TemplateLine, ...] | list[TemplateLine] = (),
    env: dict[str, Any],
) -> str:
    if template_lines:
        from kedi.lang.compiler.evaluation import merge_template_block_lines, render_expression

        merged = merge_template_block_lines(list(template_lines))
        return render_expression(rt, merged.segments, env, context="system").strip()
    return eval_model_value(
        rt,
        directive_name="system",
        plain_value=plain_value,
        backtick_code=backtick_code,
        env=env,
    )


def eval_model_directive(
    rt: KediRuntime,
    stmt: ModelDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    model = eval_model_value(
        rt,
        directive_name="model",
        plain_value=stmt.plain_value,
        backtick_code=stmt.backtick_code,
        env=env,
    )
    return AgentProfile(model=model)


def eval_effort_value(
    rt: KediRuntime,
    *,
    plain_value: str | None,
    backtick_code: str | None,
    env: dict[str, Any],
) -> EffortValue:
    value = eval_model_value(
        rt,
        directive_name="effort",
        plain_value=plain_value,
        backtick_code=backtick_code,
        env=env,
    )
    return validate_effort(value)


def eval_effort_directive(
    rt: KediRuntime,
    stmt: EffortDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    effort = eval_effort_value(
        rt,
        plain_value=stmt.plain_value,
        backtick_code=stmt.backtick_code,
        env=env,
    )
    return AgentProfile(effort=effort)


def eval_system_directive(
    rt: KediRuntime,
    stmt: SystemDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    system = eval_system_value(
        rt,
        plain_value=stmt.plain_value,
        backtick_code=stmt.backtick_code,
        template_lines=stmt.template_lines,
        env=env,
    )
    return AgentProfile(system=system)


def _member_to_patch(
    rt: KediRuntime,
    member: ProfileModelMember | ProfileEffortMember | ProfileSystemMember | ProfileToolMember,
    env: dict[str, Any],
) -> AgentProfile:
    if member.kind == "model":
        model = eval_model_value(
            rt,
            directive_name="model",
            plain_value=member.plain_value,
            backtick_code=member.backtick_code,
            env=env,
        )
        return AgentProfile(model=model)
    if member.kind == "effort":
        effort = eval_effort_value(
            rt,
            plain_value=member.plain_value,
            backtick_code=member.backtick_code,
            env=env,
        )
        return AgentProfile(effort=effort)
    if member.kind == "system":
        system = eval_system_value(
            rt,
            plain_value=member.plain_value,
            backtick_code=member.backtick_code,
            template_lines=member.template_lines,
            env=env,
        )
        return AgentProfile(system=system)
    if member.kind == "tools":
        return AgentProfile(tools=member.tool_names)
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
    stmt: UseDirectiveStmt,
    registry: dict[str, ProfileDefinition],
    env: dict[str, Any],
) -> AgentProfile:
    definition = registry.get(stmt.name)
    if definition is None:
        raise KediExecutionError(f"Unknown profile: {stmt.name!r}")
    return eval_profile_definition(rt, definition, env)
