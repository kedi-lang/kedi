from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..errors import KediExecutionError
from .definitions import iter_profile_members
from .evaluation import eval_profile_definition
from .profile import AgentProfile

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import ProcedureDef, UseDirectiveStmt

    from .capability import Capability
    from .definitions import ProfileDefinition


def eval_use_directive(
    rt: KediRuntime,
    stmt: UseDirectiveStmt,
    registry: dict[str, ProfileDefinition],
    env: dict[str, object],
    *,
    available_procedures: set[str],
    _defer_values: dict[tuple[str | int, int], bool] | None = None,
) -> AgentProfile:
    defer_values = {} if _defer_values is None else _defer_values
    if stmt.tool_names:
        from kedi.lang.ast import UseDirectiveStmt

        profile = AgentProfile.empty()
        for name in stmt.tool_names:
            profile = profile.merge(
                eval_use_directive(
                    rt,
                    UseDirectiveStmt(target=name, loc=stmt.loc),
                    registry,
                    env,
                    available_procedures=available_procedures,
                    _defer_values=defer_values,
                )
            )
        return profile
    name = stmt.name
    if not name:
        raise KediExecutionError("Empty '> use:' directive")
    if name in available_procedures:
        return AgentProfile(tools=(name,))
    if callable(env.get(name)):
        return AgentProfile(tools=(name,))
    capability = python_capability_for_use(rt, name, env)
    if capability is not None:
        return capability.as_profile()
    if python_callable_for_tool(rt, name) is not None:
        return AgentProfile(tools=(name,))
    definition = registry.get(name)
    if definition is not None:
        return eval_profile_definition(
            rt,
            definition,
            env,
            registry=registry,
            available_procedures=available_procedures,
            _defer_values=defer_values,
        )
    raise KediExecutionError(f"Unknown use target: {name!r}")


def use_needs_lazy_capture(
    rt: KediRuntime,
    stmt: UseDirectiveStmt,
    registry: dict[str, ProfileDefinition],
    env: dict[str, Any],
    *,
    available_procedures: set[str],
) -> bool:
    """Keep capability and conditional selection out of procedure preparation."""
    visited: set[int] = set()

    def visit(name: str, owner: ProfileDefinition | None = None) -> bool:
        if owner is not None and name in owner.capability_bindings:
            return True
        if (
            (owner is not None and name in owner.tool_bindings)
            or name in available_procedures
            or callable(env.get(name))
        ):
            return False
        if (owner is None or name not in owner.profile_bindings) and (
            python_capability_for_use(rt, name, env) is not None
        ):
            return True
        if python_callable_for_tool(rt, name) is not None:
            return False
        definition = (
            owner.profile_bindings.get(name) if owner is not None else None
        ) or registry.get(name)
        if definition is None or id(definition) in visited:
            return False
        visited.add(id(definition))
        if definition.kind == "capability":
            return True
        for member in iter_profile_members(definition.members):
            if member.kind == "conditional":
                return True
            if member.kind == "use" and visit(member.name, definition):
                return True
            if member.kind == "tools" and any(
                visit(target, definition) for target in member.tool_names
            ):
                return True
        return False

    return any(visit(name) for name in (stmt.tool_names or (stmt.name,)))


def python_capability_for_use(rt: KediRuntime, name: str, env: dict[str, Any]) -> Capability | None:
    from .capability import Capability

    for namespace in (
        env,
        rt._import_env,
        rt._prelude_env,
        rt._globals_env,
        rt._initial_globals_env,
    ):
        if isinstance(namespace, dict) and name in namespace:
            value = namespace[name]
            return value if isinstance(value, Capability) else None
    return None


def procedure_def_for_tool(rt: KediRuntime, name: str) -> ProcedureDef | None:
    from kedi.lang.procedure_introspection import get_procedure_def

    try:
        fn = rt._get_proc(name)
    except KeyError:
        return None
    proc = get_procedure_def(fn)
    return proc


def python_callable_for_tool(rt: KediRuntime, name: str) -> Callable[..., Any] | None:
    for env in (
        rt._import_env,
        rt._prelude_env,
        rt._globals_env,
        rt._initial_globals_env,
    ):
        if not isinstance(env, dict):
            continue
        value = env.get(name)
        if callable(value):
            return value
    return None
