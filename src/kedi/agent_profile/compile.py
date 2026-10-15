from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from ..errors import KediExecutionError
from .definitions import (
    ProfileAdapterMember,
    ProfileAgentMember,
    ProfileApprovalMember,
    ProfileArtifactsMember,
    ProfileBudgetMember,
    ProfileCodeModeMember,
    ProfileDeferMember,
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

    from .capture import CapturedAgentProfile


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
        | ProfileDeferMember
    ] = []
    _assert_profile_adapter_selection_is_not_mixed(stmt)
    for entry in stmt.defer_members:
        members.append(
            ProfileDeferMember(
                enabled=entry.enabled, backtick_code=entry.backtick_code, loc=entry.loc
            )
        )
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
        members.append(ProfileMaxAgentsMember(value=entry.value, loc=entry.loc))
    if stmt.use_members:
        for use in stmt.use_members:
            if use.tool_names:
                members.append(ProfileToolMember(tool_names=tuple(use.tool_names), loc=use.loc))
            else:
                members.append(ProfileUseMember(name=use.name, loc=use.loc))
    else:
        for target in stmt.use_targets:
            members.append(ProfileUseMember(name=target))
        if stmt.tool_names:
            members.append(ProfileToolMember(tool_names=tuple(stmt.tool_names)))
    if len(stmt.workflow_members) > 1:
        raise KediExecutionError("profile can define only one workflow mode")
    if stmt.workflow_members:
        workflow = stmt.workflow_members[0]
        if workflow.mode == "dynamic" and not stmt.subagent_members and not stmt.use_targets:
            raise KediExecutionError("dynamic workflow requires at least one direct subagent")
        members.append(ProfileWorkflowMember(mode=workflow.mode, loc=workflow.loc))
    # Member lists remain grouped in the AST for editor consumers; evaluation
    # must follow source order, including nested profile applications.
    members.sort(key=lambda member: member.loc.start_byte if member.loc is not None else -1)
    source = stmt.loc.source_path if stmt.loc is not None else None
    return ProfileDefinition(
        name=stmt.name,
        loc=stmt.loc,
        kind=stmt.kind,
        identity=f"{source or '<program>'}::{stmt.name}",
        description=stmt.docstring,
        members=members,
    )


def _agent_entry_to_member(entry: ProfileAgentEntry) -> ProfileAgentMember:
    return ProfileAgentMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        command_plain_value=entry.command_plain_value,
        command_backtick_code=entry.command_backtick_code,
        connection_fields=tuple(entry.connection_fields),
        loc=entry.loc,
    )


def _adapter_entry_to_member(entry: ProfileAdapterEntry) -> ProfileAdapterMember:
    return ProfileAdapterMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        loc=entry.loc,
    )


def _model_entry_to_member(entry: ProfileModelEntry) -> ProfileModelMember:
    return ProfileModelMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        loc=entry.loc,
    )


def _effort_entry_to_member(entry: ProfileEffortEntry) -> ProfileEffortMember:
    return ProfileEffortMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        loc=entry.loc,
    )


def _approval_entry_to_member(entry: ProfileApprovalEntry) -> ProfileApprovalMember:
    return ProfileApprovalMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        loc=entry.loc,
    )


def _hooks_entry_to_member(entry: ProfileHooksEntry) -> ProfileHooksMember:
    return ProfileHooksMember(enabled=entry.enabled, fields=tuple(entry.fields), loc=entry.loc)


def _history_entry_to_member(entry: ProfileHistoryEntry) -> ProfileHistoryMember:
    return ProfileHistoryMember(enabled=entry.enabled, fields=tuple(entry.fields), loc=entry.loc)


def _skills_entry_to_member(entry: ProfileSkillsEntry) -> ProfileSkillsMember:
    return ProfileSkillsMember(enabled=entry.enabled, fields=tuple(entry.fields), loc=entry.loc)


def _codemode_entry_to_member(entry: ProfileCodeModeEntry) -> ProfileCodeModeMember:
    return ProfileCodeModeMember(enabled=entry.enabled, fields=tuple(entry.fields), loc=entry.loc)


def _requires_entry_to_member(entry: ProfileRequiresEntry) -> ProfileRequiresMember:
    return ProfileRequiresMember(
        requirements=tuple(entry.requirements),
        requirement_locs=dict(entry.requirement_locs),
        loc=entry.loc,
    )


def _budget_entry_to_member(entry: ProfileBudgetEntry) -> ProfileBudgetMember:
    return ProfileBudgetMember(
        request_limit=entry.request_limit,
        tool_attempt_limit=entry.tool_attempt_limit,
        loc=entry.loc,
    )


def _system_entry_to_member(entry: ProfileSystemEntry) -> ProfileSystemMember:
    return ProfileSystemMember(
        plain_value=entry.plain_value,
        backtick_code=entry.backtick_code,
        template_lines=tuple(entry.template_lines),
        loc=entry.loc,
    )


def _mcp_entry_to_member(entry: ProfileMcpEntry) -> ProfileMcpMember:
    return ProfileMcpMember(fields=tuple(entry.fields), loc=entry.loc)


def _settings_entry_to_member(entry: ProfileSettingsEntry) -> ProfileSettingsMember:
    return ProfileSettingsMember(fields=tuple(entry.fields), loc=entry.loc)


def _artifacts_entry_to_member(entry: ProfileArtifactsEntry) -> ProfileArtifactsMember:
    return ProfileArtifactsMember(fields=tuple(entry.fields), loc=entry.loc)


def _output_entry_to_member(entry: ProfileOutputEntry) -> ProfileOutputMember:
    return ProfileOutputMember(
        type_annotation=entry.type_annotation,
        type_expr=entry.type_expr,
        loc=entry.loc,
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
        child = bound if bound is not None else registry.get(child_name)
        if child is None:
            raise KediExecutionError(
                f"Unknown subagent profile {child_name!r} referenced by profile {definition.name!r}"
            )
        if child.kind != "profile":
            raise KediExecutionError(
                f"Subagent {child_name!r} must name a profile, not a capability"
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


def capture_procedure_profiles(
    rt: KediRuntime,
    prog: Program,
    *,
    captures: dict[int, CapturedAgentProfile] | None = None,
) -> dict[int, Callable[..., Any]]:
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

    from .capture import CapturedAgentProfile
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
    from .profile import AgentProfile
    from .use_resolution import eval_use_directive, use_contains_capability

    last_procedure = max(
        (
            index
            for index, item in enumerate(prog.ordered_toplevel)
            if isinstance(item, ProcedureDef)
        ),
        default=-1,
    )
    if last_procedure < 0:
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
    active: AgentProfile | CapturedAgentProfile = rt.default_agent_profile
    defined_procedures: set[str] = set(rt._procedures.keys())

    evaluators: dict[type, Callable[..., AgentProfile]] = {
        AgentDirectiveStmt: eval_agent_directive,
        AdapterDirectiveStmt: eval_adapter_directive,
        ModelDirectiveStmt: eval_model_directive,
        EffortDirectiveStmt: eval_effort_directive,
        ApprovalDirectiveStmt: eval_approval_directive,
        HooksDirectiveStmt: lambda rt, stmt, env: eval_hooks_directive(
            rt, stmt, env
        ).with_hook_enforcement(),
        HistoryDirectiveStmt: eval_history_directive,
        SkillsDirectiveStmt: eval_skills_directive,
        CodeModeDirectiveStmt: eval_codemode_directive,
        RequiresDirectiveStmt: eval_requires_directive,
        BudgetDirectiveStmt: eval_budget_directive,
        SystemDirectiveStmt: eval_system_directive,
        McpDirectiveStmt: eval_mcp_directive,
        SettingsDirectiveStmt: eval_settings_directive,
        ArtifactsDirectiveStmt: eval_artifacts_directive,
    }

    for item in prog.ordered_toplevel[: last_procedure + 1]:
        if isinstance(item, ProcedureDef):
            item.captured_agent_profile = (
                active if isinstance(active, CapturedAgentProfile) else active.frozen_copy()
            )
            defined_procedures.add(item.name)
            if not item.auto_spec and not item.ai_prompt:
                env[item.name] = reference(item)
            continue

        pending = isinstance(active, CapturedAgentProfile)
        if isinstance(item, UseDirectiveStmt):
            pending = pending or use_contains_capability(
                rt, item, registry, env, available_procedures=defined_procedures
            )
            evaluate = partial(
                eval_use_directive,
                rt,
                item,
                dict(registry),
                available_procedures=set(defined_procedures),
            )
        else:
            evaluator = evaluators.get(type(item))
            if evaluator is None:
                continue
            evaluate = partial(evaluator, rt, item)
        if isinstance(active, CapturedAgentProfile) or pending:
            active = CapturedAgentProfile(active, evaluate, env)
            if captures is not None:
                captures[id(item)] = active
        else:
            active = active.merge(evaluate(dict(env)))

    return bindings
