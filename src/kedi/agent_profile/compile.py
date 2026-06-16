from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .profile import AgentProfile
from .registry import (
    ProfileDefinition,
    ProfileEffortMember,
    ProfileMcpMember,
    ProfileModelMember,
    ProfileSystemMember,
    ProfileToolMember,
)

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import (
        ProfileDefStmt,
        ProfileEffortEntry,
        ProfileMcpEntry,
        ProfileModelEntry,
        ProfileSystemEntry,
        Program,
    )


def profile_stmt_to_definition(stmt: ProfileDefStmt) -> ProfileDefinition:
    members: list[
        ProfileModelMember
        | ProfileEffortMember
        | ProfileSystemMember
        | ProfileToolMember
        | ProfileMcpMember
    ] = []
    for entry in stmt.model_members:
        members.append(_model_entry_to_member(entry))
    for entry in stmt.effort_members:
        members.append(_effort_entry_to_member(entry))
    for entry in stmt.system_members:
        members.append(_system_entry_to_member(entry))
    for entry in stmt.mcp_members:
        members.append(_mcp_entry_to_member(entry))
    if stmt.tool_names:
        members.append(ProfileToolMember(tool_names=tuple(stmt.tool_names)))
    return ProfileDefinition(name=stmt.name, members=members)


def _model_entry_to_member(entry: ProfileModelEntry) -> ProfileModelMember:
    return ProfileModelMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
    )


def _effort_entry_to_member(entry: ProfileEffortEntry) -> ProfileEffortMember:
    return ProfileEffortMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
    )


def _system_entry_to_member(entry: ProfileSystemEntry) -> ProfileSystemMember:
    return ProfileSystemMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        template_lines=tuple(entry.template_lines),
    )


def _mcp_entry_to_member(entry: ProfileMcpEntry) -> ProfileMcpMember:
    return ProfileMcpMember(fields=tuple(entry.fields))


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
    from kedi.lang.ast import (
        EffortDirectiveStmt,
        McpDirectiveStmt,
        ModelDirectiveStmt,
        ProcedureDef,
        SystemDirectiveStmt,
        UseDirectiveStmt,
    )

    from .model import (
        eval_effort_directive,
        eval_mcp_directive,
        eval_model_directive,
        eval_system_directive,
    )
    from .tools import eval_use_directive

    if not any(isinstance(item, ProcedureDef) for item in prog.ordered_toplevel):
        return

    registry = effective_profile_registry(rt)
    env = build_capture_env(rt)
    active = AgentProfile.empty()
    defined_procedures: set[str] = set(rt._procedures.keys())

    for item in prog.ordered_toplevel:
        if isinstance(item, ModelDirectiveStmt):
            active = active.merge(eval_model_directive(rt, item, env))
        elif isinstance(item, EffortDirectiveStmt):
            active = active.merge(eval_effort_directive(rt, item, env))
        elif isinstance(item, SystemDirectiveStmt):
            active = active.merge(eval_system_directive(rt, item, env))
        elif isinstance(item, McpDirectiveStmt):
            active = active.merge(eval_mcp_directive(rt, item, env))
        elif isinstance(item, UseDirectiveStmt):
            active = active.merge(
                eval_use_directive(
                    rt,
                    item,
                    registry,
                    env,
                    available_procedures=defined_procedures,
                )
            )
        elif isinstance(item, ProcedureDef):
            item.captured_agent_profile = active.frozen_copy()
            defined_procedures.add(item.name)
