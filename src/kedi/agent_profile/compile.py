from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..errors import KediExecutionError
from .definitions import (
    ProfileAdapterMember,
    ProfileAgentMember,
    ProfileApprovalMember,
    ProfileArtifactsMember,
    ProfileBudgetMember,
    ProfileCodeModeMember,
    ProfileDefinition,
    ProfileEffortMember,
    ProfileHistoryMember,
    ProfileHooksMember,
    ProfileMaxAgentsMember,
    ProfileMcpMember,
    ProfileModelMember,
    ProfileOutputMember,
    ProfileRequiresMember,
    ProfileSettingsMember,
    ProfileSkillsMember,
    ProfileSubagentMember,
    ProfileSystemMember,
    ProfileToolMember,
    ProfileUseMember,
    ProfileWorkflowMember,
)

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import (
        ProfileAdapterEntry,
        ProfileAgentEntry,
        ProfileApprovalEntry,
        ProfileArtifactsEntry,
        ProfileBudgetEntry,
        ProfileCodeModeEntry,
        ProfileDefStmt,
        ProfileEffortEntry,
        ProfileHistoryEntry,
        ProfileHooksEntry,
        ProfileMcpEntry,
        ProfileModelEntry,
        ProfileOutputEntry,
        ProfileRequiresEntry,
        ProfileSettingsEntry,
        ProfileSkillsEntry,
        ProfileSystemEntry,
        Program,
    )


def profile_stmt_to_definition(stmt: ProfileDefStmt) -> ProfileDefinition:
    members: list[
        ProfileAgentMember
        | ProfileAdapterMember
        | ProfileModelMember
        | ProfileEffortMember
        | ProfileApprovalMember
        | ProfileHooksMember
        | ProfileHistoryMember
        | ProfileSkillsMember
        | ProfileCodeModeMember
        | ProfileRequiresMember
        | ProfileBudgetMember
        | ProfileSystemMember
        | ProfileToolMember
        | ProfileUseMember
        | ProfileMcpMember
        | ProfileSettingsMember
        | ProfileArtifactsMember
        | ProfileOutputMember
        | ProfileSubagentMember
        | ProfileMaxAgentsMember
        | ProfileWorkflowMember
    ] = []
    _assert_profile_adapter_selection_is_not_mixed(stmt)
    for entry in stmt.agent_members:
        members.append(_agent_entry_to_member(entry))
    for entry in stmt.adapter_members:
        members.append(_adapter_entry_to_member(entry))
    for entry in stmt.model_members:
        members.append(_model_entry_to_member(entry))
    for entry in stmt.effort_members:
        members.append(_effort_entry_to_member(entry))
    for entry in stmt.approval_members:
        members.append(_approval_entry_to_member(entry))
    for entry in stmt.hooks_members:
        members.append(_hooks_entry_to_member(entry))
    for entry in stmt.history_members:
        members.append(_history_entry_to_member(entry))
    for entry in stmt.skills_members:
        members.append(_skills_entry_to_member(entry))
    for entry in stmt.codemode_members:
        members.append(_codemode_entry_to_member(entry))
    for entry in stmt.requires_members:
        members.append(_requires_entry_to_member(entry))
    for entry in stmt.budget_members:
        members.append(_budget_entry_to_member(entry))
    for entry in stmt.system_members:
        members.append(_system_entry_to_member(entry))
    for entry in stmt.mcp_members:
        members.append(_mcp_entry_to_member(entry))
    for entry in stmt.settings_members:
        members.append(_settings_entry_to_member(entry))
    for entry in stmt.artifacts_members:
        members.append(_artifacts_entry_to_member(entry))
    if len(stmt.output_members) > 1:
        raise KediExecutionError("profile can define only one output type")
    if stmt.output_members:
        members.append(_output_entry_to_member(stmt.output_members[0]))
    for entry in stmt.subagent_members:
        members.append(ProfileSubagentMember(name=entry.name, loc=entry.loc))
    for entry in stmt.max_agents_members:
        members.append(ProfileMaxAgentsMember(value=entry.value))
    for target in stmt.use_targets:
        members.append(ProfileUseMember(name=target))
    if stmt.tool_names:
        members.append(ProfileToolMember(tool_names=tuple(stmt.tool_names)))
    if len(stmt.workflow_members) > 1:
        raise KediExecutionError("profile can define only one workflow mode")
    if stmt.workflow_members:
        workflow = stmt.workflow_members[0]
        if workflow.mode == "dynamic" and not stmt.subagent_members:
            raise KediExecutionError("dynamic workflow requires at least one direct subagent")
        members.append(ProfileWorkflowMember(mode=workflow.mode))
    return ProfileDefinition(name=stmt.name, description=stmt.docstring, members=members)


def _agent_entry_to_member(entry: ProfileAgentEntry) -> ProfileAgentMember:
    return ProfileAgentMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        command_plain_value=entry.command_plain_value,
        command_backtick_code=entry.command_backtick_code,
        connection_fields=tuple(entry.connection_fields),
    )


def _adapter_entry_to_member(entry: ProfileAdapterEntry) -> ProfileAdapterMember:
    return ProfileAdapterMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
    )


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


def _approval_entry_to_member(entry: ProfileApprovalEntry) -> ProfileApprovalMember:
    return ProfileApprovalMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
    )


def _hooks_entry_to_member(entry: ProfileHooksEntry) -> ProfileHooksMember:
    return ProfileHooksMember(enabled=entry.enabled, fields=tuple(entry.fields))


def _history_entry_to_member(entry: ProfileHistoryEntry) -> ProfileHistoryMember:
    return ProfileHistoryMember(enabled=entry.enabled, fields=tuple(entry.fields))


def _skills_entry_to_member(entry: ProfileSkillsEntry) -> ProfileSkillsMember:
    return ProfileSkillsMember(enabled=entry.enabled, fields=tuple(entry.fields))


def _codemode_entry_to_member(entry: ProfileCodeModeEntry) -> ProfileCodeModeMember:
    return ProfileCodeModeMember(enabled=entry.enabled, fields=tuple(entry.fields))


def _requires_entry_to_member(entry: ProfileRequiresEntry) -> ProfileRequiresMember:
    return ProfileRequiresMember(
        requirements=tuple(entry.requirements),
        requirement_locs=dict(entry.requirement_locs),
    )


def _budget_entry_to_member(entry: ProfileBudgetEntry) -> ProfileBudgetMember:
    return ProfileBudgetMember(
        request_limit=entry.request_limit,
        tool_attempt_limit=entry.tool_attempt_limit,
    )


def _system_entry_to_member(entry: ProfileSystemEntry) -> ProfileSystemMember:
    return ProfileSystemMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        template_lines=tuple(entry.template_lines),
    )


def _mcp_entry_to_member(entry: ProfileMcpEntry) -> ProfileMcpMember:
    return ProfileMcpMember(fields=tuple(entry.fields))


def _settings_entry_to_member(entry: ProfileSettingsEntry) -> ProfileSettingsMember:
    return ProfileSettingsMember(fields=tuple(entry.fields))


def _artifacts_entry_to_member(entry: ProfileArtifactsEntry) -> ProfileArtifactsMember:
    return ProfileArtifactsMember(fields=tuple(entry.fields))


def _output_entry_to_member(entry: ProfileOutputEntry) -> ProfileOutputMember:
    return ProfileOutputMember(
        type_annotation=entry.type_annotation,
        type_expr=entry.type_expr,
    )


def build_profile_registry(prog: Program) -> dict[str, ProfileDefinition]:
    return {name: profile_stmt_to_definition(stmt) for name, stmt in prog.profile_defs.items()}


def validate_subagent_graph(registry: dict[str, ProfileDefinition]) -> None:
    visited: set[int] = set()
    active: list[ProfileDefinition] = []
    active_ids: set[int] = set()

    def child_definition(
        definition: ProfileDefinition,
        child_name: str,
    ) -> ProfileDefinition:
        bound = definition.subagent_bindings.get(child_name)
        if bound is not None:
            return bound
        child = registry.get(child_name)
        if child is None:
            raise KediExecutionError(
                f"Unknown subagent profile {child_name!r} referenced by profile {definition.name!r}"
            )
        return child

    def visit(definition: ProfileDefinition) -> None:
        identity = id(definition)
        if identity in visited:
            return
        if identity in active_ids:
            cycle_start = next(
                index
                for index, active_definition in enumerate(active)
                if active_definition is definition
            )
            cycle = (*[item.name for item in active[cycle_start:]], definition.name)
            raise KediExecutionError(f"Cyclic subagent profile reference: {' -> '.join(cycle)}")
        active.append(definition)
        active_ids.add(identity)
        for member in definition.members:
            if member.kind == "subagents":
                visit(child_definition(definition, member.name))
        active.pop()
        active_ids.remove(identity)
        visited.add(identity)

    for definition in registry.values():
        visit(definition)


def _assert_profile_adapter_selection_is_not_mixed(stmt: ProfileDefStmt) -> None:
    if stmt.agent_members and stmt.adapter_members:
        from ..errors import KediExecutionError

        raise KediExecutionError(
            f"Profile {stmt.name!r} cannot contain both '> agent:' and '> adapter:' members"
        )


def effective_profile_registry(rt: KediRuntime) -> dict[str, ProfileDefinition]:
    registry = rt._profile_registry
    return registry if isinstance(registry, dict) else {}


def build_capture_env(rt: KediRuntime) -> dict[str, Any]:
    env: dict[str, Any] = {}
    if isinstance(rt._prelude_env, dict):
        env.update(rt._prelude_env)
    env.update(rt._initial_globals_env)
    if isinstance(rt._import_env, dict):
        env.update(rt._import_env)
    return env


def capture_procedure_profiles(rt: KediRuntime, prog: Program) -> dict[int, Callable[..., Any]]:
    from kedi.lang.ast import (
        AdapterDirectiveStmt,
        AgentDirectiveStmt,
        ApprovalDirectiveStmt,
        ArtifactsDirectiveStmt,
        BudgetDirectiveStmt,
        CodeModeDirectiveStmt,
        EffortDirectiveStmt,
        HistoryDirectiveStmt,
        HooksDirectiveStmt,
        McpDirectiveStmt,
        ModelDirectiveStmt,
        ProcedureDef,
        RequiresDirectiveStmt,
        SettingsDirectiveStmt,
        SkillsDirectiveStmt,
        SystemDirectiveStmt,
        UseDirectiveStmt,
    )

    from .directives import (
        eval_adapter_directive,
        eval_agent_directive,
        eval_approval_directive,
        eval_artifacts_directive,
        eval_budget_directive,
        eval_codemode_directive,
        eval_effort_directive,
        eval_history_directive,
        eval_hooks_directive,
        eval_mcp_directive,
        eval_model_directive,
        eval_requires_directive,
        eval_settings_directive,
        eval_skills_directive,
        eval_system_directive,
    )
    from .use_resolution import eval_use_directive

    if not any(isinstance(item, ProcedureDef) for item in prog.ordered_toplevel):
        return {}

    # Profile capture precedes compilation. These references are filled with the
    # exact compiled definitions, never resolved through a caller's shadowed name.
    bindings: dict[int, Callable[..., Any]] = {}

    def reference(definition: ProcedureDef) -> Callable[..., Any]:
        def invoke(*args: Any) -> Any:
            return rt._invoke_procedure(bindings[id(definition)], list(args))

        return invoke

    registry = effective_profile_registry(rt)
    env = build_capture_env(rt)
    active = rt.default_agent_profile
    adapter_selection_kind: str | None = None
    defined_procedures: set[str] = set(rt._procedures.keys())

    for item in prog.ordered_toplevel:
        if isinstance(item, AgentDirectiveStmt):
            _assert_adapter_selection_not_mixed(adapter_selection_kind, "agent")
            adapter_selection_kind = "agent"
            active = active.merge(eval_agent_directive(rt, item, env))
        elif isinstance(item, AdapterDirectiveStmt):
            _assert_adapter_selection_not_mixed(adapter_selection_kind, "adapter")
            adapter_selection_kind = "adapter"
            active = active.merge(eval_adapter_directive(rt, item, env))
        elif isinstance(item, ModelDirectiveStmt):
            active = active.merge(eval_model_directive(rt, item, env))
        elif isinstance(item, EffortDirectiveStmt):
            active = active.merge(eval_effort_directive(rt, item, env))
        elif isinstance(item, ApprovalDirectiveStmt):
            active = active.merge(eval_approval_directive(rt, item, env))
        elif isinstance(item, HooksDirectiveStmt):
            active = active.merge(eval_hooks_directive(rt, item, env).with_hook_enforcement())
        elif isinstance(item, HistoryDirectiveStmt):
            active = active.merge(eval_history_directive(rt, item, env))
        elif isinstance(item, SkillsDirectiveStmt):
            active = active.merge(eval_skills_directive(rt, item, env))
        elif isinstance(item, CodeModeDirectiveStmt):
            active = active.merge(eval_codemode_directive(rt, item, env))
        elif isinstance(item, RequiresDirectiveStmt):
            active = active.merge(eval_requires_directive(rt, item, env))
        elif isinstance(item, BudgetDirectiveStmt):
            active = active.merge(eval_budget_directive(rt, item, env))
        elif isinstance(item, SystemDirectiveStmt):
            active = active.merge(eval_system_directive(rt, item, env))
        elif isinstance(item, McpDirectiveStmt):
            active = active.merge(eval_mcp_directive(rt, item, env))
        elif isinstance(item, SettingsDirectiveStmt):
            active = active.merge(eval_settings_directive(rt, item, env))
        elif isinstance(item, ArtifactsDirectiveStmt):
            active = active.merge(eval_artifacts_directive(rt, item, env))
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
            if not item.auto_spec and not item.ai_prompt:
                env[item.name] = reference(item)

    return bindings


def _assert_adapter_selection_not_mixed(current: str | None, incoming: str) -> None:
    if current is not None and current != incoming:
        from ..errors import KediExecutionError

        raise KediExecutionError("Cannot mix '> agent:' and '> adapter:' in the same scope")
