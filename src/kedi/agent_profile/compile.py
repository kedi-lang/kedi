from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kedi.lang.ast import ModelDirectiveStmt, ProcedureDef, UseProfileStmt

from .profile import AgentProfile
from .registry import ProfileDefinition, ProfileModelMember

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import ProfileDefStmt, ProfileModelEntry, Program


def profile_stmt_to_definition(stmt: ProfileDefStmt) -> ProfileDefinition:
    members: list[ProfileModelMember] = []
    for entry in stmt.model_members:
        members.append(_entry_to_member(entry))
    return ProfileDefinition(name=stmt.name, members=members)


def _entry_to_member(entry: ProfileModelEntry) -> ProfileModelMember:
    return ProfileModelMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
    )


def build_profile_registry(prog: Program) -> dict[str, ProfileDefinition]:
    return {name: profile_stmt_to_definition(stmt) for name, stmt in prog.profile_defs.items()}


def effective_profile_registry(rt: KediRuntime) -> dict[str, ProfileDefinition]:
    registry = getattr(rt, "_profile_registry", None)
    if isinstance(registry, dict):
        return registry
    return {}


def build_capture_env(rt: KediRuntime) -> dict[str, Any]:
    env: dict[str, Any] = {}
    prelude_env = getattr(rt, "_prelude_env", None)
    if isinstance(prelude_env, dict):
        env.update(prelude_env)
    env.update(getattr(rt, "_initial_globals_env", {}))
    import_env = getattr(rt, "_import_env", None)
    if isinstance(import_env, dict):
        env.update(import_env)
    return env


def capture_procedure_profiles(rt: KediRuntime, prog: Program) -> None:
    from .model import eval_model_directive, eval_use_profile

    registry = effective_profile_registry(rt)
    env = build_capture_env(rt)
    active = AgentProfile.empty()

    for item in prog.ordered_toplevel:
        if isinstance(item, ModelDirectiveStmt):
            active = active.merge(eval_model_directive(rt, item, env))
        elif isinstance(item, UseProfileStmt):
            active = active.merge(eval_use_profile(rt, item, registry, env))
        elif isinstance(item, ProcedureDef):
            item.captured_agent_profile = active.frozen_copy()
