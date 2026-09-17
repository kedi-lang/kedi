from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import KediExecutionError
from .definitions import ProfileDefinition, ProfileUseMember
from .directives import _member_to_patch
from .profile import AgentProfile

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import UseDirectiveStmt


def eval_profile_definition(
    rt: KediRuntime,
    definition: ProfileDefinition,
    env: dict[str, Any],
    *,
    registry: dict[str, ProfileDefinition] | None = None,
    available_procedures: set[str] | None = None,
    _active_definitions: tuple[int, ...] = (),
) -> AgentProfile:
    identity = id(definition)
    if identity in _active_definitions:
        raise KediExecutionError(f"Cyclic profile use reference involving {definition.name!r}")
    active_definitions = (*_active_definitions, identity)
    effective_registry = (
        registry
        if registry is not None
        else rt._profile_registry
        if isinstance(rt._profile_registry, dict)
        else {}
    )
    effective_procedures = (
        available_procedures if available_procedures is not None else set(rt._procedures)
    )
    profile = AgentProfile.empty()
    for member in definition.members:
        if member.kind == "use":
            profile = profile.merge(
                _eval_profile_use_member(
                    rt,
                    definition,
                    member,
                    env,
                    registry=effective_registry,
                    available_procedures=effective_procedures,
                    active_definitions=active_definitions,
                )
            )
        else:
            profile = profile.merge(_member_to_patch(rt, member, env))
    return profile.merge(
        AgentProfile(
            name=definition.name,
            tool_bindings=definition.tool_bindings,
            subagent_bindings=definition.subagent_bindings,
        )
    )


def _eval_profile_use_member(
    rt: KediRuntime,
    owner: ProfileDefinition,
    member: ProfileUseMember,
    env: dict[str, Any],
    *,
    registry: dict[str, ProfileDefinition],
    available_procedures: set[str],
    active_definitions: tuple[int, ...],
) -> AgentProfile:
    from .use_resolution import python_callable_for_tool

    name = member.name
    if (
        name in owner.tool_bindings
        or name in available_procedures
        or callable(env.get(name))
        or python_callable_for_tool(rt, name) is not None
    ):
        return AgentProfile(tools=(name,))
    nested = owner.profile_bindings.get(name) or registry.get(name)
    if nested is not None:
        return eval_profile_definition(
            rt,
            nested,
            env,
            registry=registry,
            available_procedures=available_procedures,
            _active_definitions=active_definitions,
        )
    raise KediExecutionError(f"Unknown use target {name!r} in profile {owner.name!r}")


def eval_use_profile(
    rt: KediRuntime,
    stmt: UseDirectiveStmt,
    registry: dict[str, ProfileDefinition],
    env: dict[str, Any],
) -> AgentProfile:
    definition = registry.get(stmt.name)
    if definition is None:
        raise KediExecutionError(f"Unknown profile: {stmt.name!r}")
    return eval_profile_definition(rt, definition, env, registry=registry)
