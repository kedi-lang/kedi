from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from ..errors import KediExecutionError, KediTraceFrame
from .definitions import (
    ProfileDefinition,
    ProfileMemberValue,
    ProfileUseMember,
    iter_profile_members,
)
from .directives import _member_to_patch
from .profile import AgentProfile

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import UseDirectiveStmt


_State = TypeVar("_State")


def apply_profile_members(
    members: Sequence[ProfileMemberValue],
    initial: _State,
    apply_member: Callable[[_State, Any], _State],
    apply_use: Callable[[_State, ProfileUseMember], _State],
) -> _State:
    """Shared profile precedence for runtime evaluation and symbolic inspection."""
    state = initial
    for member in members:
        state = apply_use(state, member) if member.kind == "use" else apply_member(state, member)
    return state


def eval_profile_definition(
    rt: KediRuntime,
    definition: ProfileDefinition,
    env: dict[str, Any],
    *,
    registry: dict[str, ProfileDefinition] | None = None,
    available_procedures: set[str] | None = None,
    _active_definitions: tuple[int, ...] = (),
    _defer_values: dict[tuple[str | int, int], bool] | None = None,
) -> AgentProfile:
    identity = id(definition)
    if identity in _active_definitions:
        raise KediExecutionError(f"Cyclic profile use reference involving {definition.name!r}")
    active_definitions = (*_active_definitions, identity)
    defer_values = {} if _defer_values is None else _defer_values
    defer_keys = {
        id(member): (definition.identity or identity, index)
        for index, member in enumerate(iter_profile_members(definition.members))
        if member.kind == "defer"
    }
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
    deferred = False
    selected_kinds: set[str] = set()

    def apply_member(current: AgentProfile, member: Any) -> AgentProfile:
        nonlocal deferred
        if member.kind == "conditional":
            with rt.trace_frame(
                KediTraceFrame(
                    kind="profile_condition",
                    label=f"condition in {definition.name}",
                    span=member.condition_loc,
                )
            ):
                value = rt.executor.evaluate_inline(rt, member.backtick_code, env)
                if type(value) is not bool:
                    raise TypeError("profile condition must evaluate to a bool")
            branch = member.consequence if value else member.alternative
            return apply_profile_members(branch, current, apply_member, apply_use)
        if member.kind in {"output", "workflow"} and member.kind in selected_kinds:
            label = "output type" if member.kind == "output" else "workflow mode"
            raise KediExecutionError(f"profile can define only one {label} on the selected path")
        if (
            member.kind == "agent"
            and "adapter" in selected_kinds
            or member.kind == "adapter"
            and "agent" in selected_kinds
        ):
            raise KediExecutionError(
                f"Profile {definition.name!r} cannot contain both '> agent:' and '> adapter:' members"
            )
        selected_kinds.add(member.kind)
        if member.kind == "defer":
            if definition.kind != "capability":
                raise KediExecutionError("defer is only valid inside a capability")
            key = defer_keys[id(member)]
            if key not in defer_values:
                with rt.trace_frame(
                    KediTraceFrame(
                        kind="capability_defer",
                        label=f"defer {definition.name}",
                        span=member.loc,
                    )
                ):
                    value = (
                        rt.executor.evaluate_inline(rt, member.backtick_code, env)
                        if member.backtick_code is not None
                        else member.enabled
                    )
                    if type(value) is not bool:
                        raise TypeError("capability defer must evaluate to a bool")
                defer_values[key] = value
            value = defer_values[key]
            deferred = value
            return current
        if definition.kind == "capability":
            if member.kind not in {
                "system",
                "tools",
                "mcp",
                "subagents",
                "approval",
                "hooks",
                "requires",
            }:
                raise KediExecutionError(f"{member.kind!r} is not allowed inside a capability")
        if member.kind == "tools":
            for name in member.tool_names:
                current = current.merge(
                    _eval_profile_use_member(
                        rt,
                        definition,
                        ProfileUseMember(name=name, loc=member.loc),
                        env,
                        registry=effective_registry,
                        available_procedures=effective_procedures,
                        active_definitions=active_definitions,
                        defer_values=defer_values,
                    )
                )
            return current
        patch = _member_to_patch(rt, member, env)
        if definition.kind == "capability" and member.kind == "subagents":
            child = definition.subagent_bindings.get(member.name) or effective_registry.get(
                member.name
            )
            if child is None or child.kind != "profile":
                raise KediExecutionError(
                    f"Unknown subagent profile {member.name!r} in capability {definition.name!r}"
                )
            patch = patch.merge(AgentProfile(subagent_bindings={member.name: child}))
        if definition.kind == "capability" and patch.hooks is not None:
            if patch.hooks.user_prompt_submit or patch.hooks.enabled is False:
                raise KediExecutionError("capabilities may only add owned tool hooks")
        return current.merge(patch)

    def apply_use(current: AgentProfile, member: ProfileUseMember) -> AgentProfile:
        return current.merge(
            _eval_profile_use_member(
                rt,
                definition,
                member,
                env,
                registry=effective_registry,
                available_procedures=effective_procedures,
                active_definitions=active_definitions,
                defer_values=defer_values,
            )
        )

    profile = apply_profile_members(
        definition.members, AgentProfile.empty(), apply_member, apply_use
    )
    profile = profile.merge(
        AgentProfile(
            name=definition.name,
            tool_bindings=definition.tool_bindings,
            subagent_bindings=definition.subagent_bindings,
        )
    )
    if definition.kind == "capability":
        from .capabilities import capability_patch

        return capability_patch(definition, profile, deferred=deferred)
    if profile.workflow_mode == "dynamic" and not profile.subagents:
        from .capabilities import deferred_capabilities

        if not any(capability.profile.subagents for capability in deferred_capabilities(profile)):
            raise KediExecutionError("dynamic workflow requires at least one direct subagent")
    return profile


def _eval_profile_use_member(
    rt: KediRuntime,
    owner: ProfileDefinition,
    member: ProfileUseMember,
    env: dict[str, Any],
    *,
    registry: dict[str, ProfileDefinition],
    available_procedures: set[str],
    active_definitions: tuple[int, ...],
    defer_values: dict[tuple[str | int, int], bool],
) -> AgentProfile:
    from .use_resolution import python_callable_for_tool, python_capability_for_use

    name = member.name
    if name in owner.capability_bindings:
        return owner.capability_bindings[name].as_profile()
    capability = (
        None if name in owner.profile_bindings else python_capability_for_use(rt, name, env)
    )
    if (
        name in owner.tool_bindings
        or name in available_procedures
        or callable(env.get(name))
        or (capability is None and python_callable_for_tool(rt, name) is not None)
    ):
        if owner.kind == "capability":
            return _bound_capability_tool(rt, owner, name, env)
        return AgentProfile(tools=(name,))
    nested = owner.profile_bindings.get(name)
    if nested is None:
        if capability is not None:
            return capability.as_profile()
        nested = registry.get(name)
    if nested is not None:
        if owner.kind == "capability" and nested.kind != "capability":
            raise KediExecutionError(
                f"Capability {owner.name!r} cannot use profile {name!r}; bind a subagent instead"
            )
        return eval_profile_definition(
            rt,
            nested,
            env,
            registry=registry,
            available_procedures=available_procedures,
            _active_definitions=active_definitions,
            _defer_values=defer_values,
        )
    raise KediExecutionError(f"Unknown use target {name!r} in profile {owner.name!r}")


def _bound_capability_tool(
    rt: KediRuntime, owner: ProfileDefinition, name: str, env: dict[str, Any]
) -> AgentProfile:
    from ..agent_adapter.tools.conversion import (
        kedi_procedure_tool_spec,
        python_callable_tool_spec,
    )
    from .use_resolution import procedure_def_for_tool, python_callable_for_tool

    spec = owner.tool_bindings.get(name)
    if spec is None:
        proc = procedure_def_for_tool(rt, name)
        if proc is not None:
            spec = kedi_procedure_tool_spec(rt, proc, local_env=env)
        else:
            fn = env.get(name)
            if not callable(fn):
                fn = python_callable_for_tool(rt, name)
            if fn is None:
                raise KediExecutionError(f"Unknown tool {name!r} in capability {owner.name!r}")
            spec = python_callable_tool_spec(name, fn)
    return AgentProfile(tools=(name,), tool_bindings={name: spec})


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
