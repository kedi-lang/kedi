from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..agent_adapter.approval import ApprovalPolicy
from ..agent_adapter.meta import AgentAdapterKind
from ..agent_adapter.registry import validate_adapter_kind
from ..artifacts import ArtifactPolicy, normalize_artifact_policy
from ..effort import EffortValue, validate_effort
from ..errors import KediExecutionError
from .compaction import CompactionSettings, normalize_compaction_settings
from .mcp import McpServerSpec, mcp_spec_from_mapping
from .profile import AgentCommand, AgentProfile
from .registry import (
    ProfileAdapterMember,
    ProfileAgentMember,
    ProfileApprovalMember,
    ProfileArtifactsMember,
    ProfileDefinition,
    ProfileEffortMember,
    ProfileHistoryMember,
    ProfileSkillsMember,
    ProfileMaxAgentsMember,
    ProfileMcpMember,
    ProfileModelMember,
    ProfileOutputMember,
    ProfileSettingsMember,
    ProfileSubagentMember,
    ProfileSystemMember,
    ProfileToolMember,
    ProfileUseMember,
    ProfileWorkflowMember,
)
from .settings import parse_plain_setting_value

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import (
        AdapterDirectiveStmt,
        AgentDirectiveStmt,
        ApprovalDirectiveStmt,
        ArtifactsDirectiveStmt,
        ArtifactsField,
        EffortDirectiveStmt,
        HistoryDirectiveStmt,
        SkillsDirectiveStmt,
        McpDirectiveStmt,
        McpField,
        ModelDirectiveStmt,
        SettingsDirectiveStmt,
        SettingsField,
        SystemDirectiveStmt,
        TemplateLine,
        UseDirectiveStmt,
    )


class AgentProfileOverrideUnsupported(KediExecutionError):
    """Raised when the active adapter cannot apply an agent profile override."""


class AgentToolOverrideUnsupported(KediExecutionError):
    """Raised when the active adapter cannot register tools for an agent run."""


class AgentSubagentUnsupported(KediExecutionError):
    """Raised when the active adapter cannot delegate to Kedi subagents."""


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


def eval_agent_command_value(
    rt: KediRuntime,
    *,
    directive_name: str,
    plain_value: str | None,
    backtick_code: str | None,
    env: dict[str, Any],
) -> AgentCommand:
    if backtick_code is not None:
        value = rt.executor.evaluate_type_expression(rt, backtick_code, env)
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parts = tuple(value)
            if all(isinstance(part, str) for part in parts):
                return parts
        raise TypeError(
            f"{directive_name} directive must evaluate to str or sequence[str], "
            f"got {type(value).__name__}"
        )
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
        from kedi.lang.compiler.evaluation import (
            merge_template_block_lines,
            render_expression,
        )

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


def eval_agent_directive(
    rt: KediRuntime,
    stmt: AgentDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return _eval_adapter_selection(
        rt,
        directive_name="agent",
        expected_kind="agent-harness",
        plain_value=stmt.plain_value,
        backtick_code=stmt.backtick_code,
        command_plain_value=stmt.command_plain_value,
        command_backtick_code=stmt.command_backtick_code,
        env=env,
    )


def eval_adapter_directive(
    rt: KediRuntime,
    stmt: AdapterDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return _eval_adapter_selection(
        rt,
        directive_name="adapter",
        expected_kind="agent-framework",
        plain_value=stmt.plain_value,
        backtick_code=stmt.backtick_code,
        env=env,
    )


def _eval_adapter_selection(
    rt: KediRuntime,
    *,
    directive_name: str,
    expected_kind: AgentAdapterKind,
    plain_value: str | None,
    backtick_code: str | None,
    env: dict[str, Any],
    command_plain_value: str | None = None,
    command_backtick_code: str | None = None,
) -> AgentProfile:
    shortname = eval_model_value(
        rt,
        directive_name=directive_name,
        plain_value=plain_value,
        backtick_code=backtick_code,
        env=env,
    )
    validate_adapter_kind(shortname, expected_kind)
    command: AgentCommand | None = None
    if command_plain_value is not None or command_backtick_code is not None:
        command = eval_agent_command_value(
            rt,
            directive_name=f"{directive_name} command",
            plain_value=command_plain_value,
            backtick_code=command_backtick_code,
            env=env,
        )
    return AgentProfile(
        adapter_kind=expected_kind,
        adapter_shortname=shortname,
        agent_command=command,
    )


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


def eval_approval_directive(
    rt: KediRuntime,
    stmt: ApprovalDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    if stmt.backtick_code is not None:
        handler = rt.executor.evaluate_inline(rt, stmt.backtick_code, env)
        if not callable(handler):
            raise TypeError(
                "approval directive inline Python must evaluate to a callable approval handler, "
                f"got {type(handler).__name__}"
            )
        return AgentProfile(approval=ApprovalPolicy.dynamic(handler))
    if stmt.plain_value is None:
        raise ValueError("approval directive has no value")
    if stmt.plain_value == "allow":
        return AgentProfile(approval=ApprovalPolicy.allow())
    if stmt.plain_value == "deny":
        return AgentProfile(approval=ApprovalPolicy.deny())
    raise ValueError("approval directive must be 'allow', 'deny', or an inline Python handler")


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


def eval_mcp_field_value(rt: KediRuntime, field: McpField, env: dict[str, Any]) -> Any:
    if field.backtick_code is not None:
        return rt.executor.evaluate_type_expression(rt, field.backtick_code, env)
    if field.plain_value is not None:
        return field.plain_value.strip()
    return None


def eval_mcp_fields(
    rt: KediRuntime, fields: list[McpField] | tuple[McpField, ...], env: dict[str, Any]
) -> McpServerSpec:
    values = {field.name: eval_mcp_field_value(rt, field, env) for field in fields}
    return mcp_spec_from_mapping(values)


def eval_mcp_directive(
    rt: KediRuntime,
    stmt: McpDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return AgentProfile(mcp_servers=(eval_mcp_fields(rt, stmt.fields, env),))


def eval_settings_field_value(rt: KediRuntime, field: SettingsField, env: dict[str, Any]) -> Any:
    if field.backtick_code is not None:
        return rt.executor.evaluate_type_expression(rt, field.backtick_code, env)
    if field.plain_value is not None:
        return parse_plain_setting_value(field.plain_value)
    return None


def eval_settings_fields(
    rt: KediRuntime,
    fields: list[SettingsField] | tuple[SettingsField, ...],
    env: dict[str, Any],
) -> dict[str, Any]:
    return {field.name: eval_settings_field_value(rt, field, env) for field in fields}


def eval_settings_directive(
    rt: KediRuntime,
    stmt: SettingsDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return AgentProfile(settings=eval_settings_fields(rt, stmt.fields, env))


def eval_artifacts_field_value(
    rt: KediRuntime,
    field: ArtifactsField,
    env: dict[str, Any],
) -> Any:
    if field.backtick_code is not None:
        return rt.executor.evaluate_type_expression(rt, field.backtick_code, env)
    if field.plain_value is not None:
        return parse_plain_setting_value(field.plain_value)
    return None


def eval_artifacts_fields(
    rt: KediRuntime,
    fields: list[ArtifactsField] | tuple[ArtifactsField, ...],
    env: dict[str, Any],
) -> ArtifactPolicy:
    values = {field.name: eval_artifacts_field_value(rt, field, env) for field in fields}
    policy = normalize_artifact_policy(values)
    if policy is None:
        raise KediExecutionError("artifacts directive did not produce a policy")
    return policy


def eval_artifacts_directive(
    rt: KediRuntime,
    stmt: ArtifactsDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return AgentProfile(artifacts=eval_artifacts_fields(rt, stmt.fields, env))


def _eval_history_fields(
    rt: KediRuntime,
    *,
    enabled: bool,
    fields: list[SettingsField] | tuple[SettingsField, ...],
    env: dict[str, Any],
) -> CompactionSettings | None:
    values = eval_settings_fields(rt, fields, env)
    compaction_values = {
        name.removeprefix("compaction_"): value
        for name, value in values.items()
        if name in {"compaction_mode", "compaction_threshold"}
    }
    compaction = normalize_compaction_settings(compaction_values or None)
    if not enabled and compaction is not None and compaction.mode == "native":
        raise KediExecutionError("native compaction requires history to be enabled")
    return compaction


def eval_history_directive(
    rt: KediRuntime,
    stmt: HistoryDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return AgentProfile(
        history=stmt.enabled,
        compaction=_eval_history_fields(
            rt,
            enabled=stmt.enabled,
            fields=stmt.fields,
            env=env,
        ),
    )


def eval_skills_directive(
    rt: KediRuntime,
    stmt: SkillsDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    from .skills import normalize_skills_settings

    values = eval_settings_fields(rt, stmt.fields, env)
    return AgentProfile(
        skills=normalize_skills_settings(values, enabled=stmt.enabled),
    )


def _member_to_patch(
    rt: KediRuntime,
    member: (
        ProfileModelMember
        | ProfileAgentMember
        | ProfileAdapterMember
        | ProfileEffortMember
        | ProfileApprovalMember
        | ProfileHistoryMember
        | ProfileSkillsMember
        | ProfileSystemMember
        | ProfileToolMember
        | ProfileMcpMember
        | ProfileSettingsMember
        | ProfileArtifactsMember
        | ProfileOutputMember
        | ProfileSubagentMember
        | ProfileMaxAgentsMember
        | ProfileWorkflowMember
    ),
    env: dict[str, Any],
) -> AgentProfile:
    if member.kind == "agent":
        return _eval_adapter_selection(
            rt,
            directive_name="agent",
            expected_kind="agent-harness",
            plain_value=member.plain_value,
            backtick_code=member.backtick_code,
            command_plain_value=member.command_plain_value,
            command_backtick_code=member.command_backtick_code,
            env=env,
        )
    if member.kind == "adapter":
        return _eval_adapter_selection(
            rt,
            directive_name="adapter",
            expected_kind="agent-framework",
            plain_value=member.plain_value,
            backtick_code=member.backtick_code,
            env=env,
        )
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
    if member.kind == "approval":
        if member.backtick_code is not None:
            handler = rt.executor.evaluate_inline(rt, member.backtick_code, env)
            if not callable(handler):
                raise TypeError(
                    "profile approval inline Python must evaluate to a callable approval handler, "
                    f"got {type(handler).__name__}"
                )
            return AgentProfile(approval=ApprovalPolicy.dynamic(handler))
        if member.plain_value == "allow":
            return AgentProfile(approval=ApprovalPolicy.allow())
        if member.plain_value == "deny":
            return AgentProfile(approval=ApprovalPolicy.deny())
        raise ValueError("profile approval must be 'allow' or 'deny'")
    if member.kind == "history":
        return AgentProfile(
            history=member.enabled,
            compaction=_eval_history_fields(
                rt,
                enabled=member.enabled,
                fields=member.fields,
                env=env,
            ),
        )
    if member.kind == "skills":
        from .skills import normalize_skills_settings

        values = eval_settings_fields(rt, member.fields, env)
        return AgentProfile(
            skills=normalize_skills_settings(values, enabled=member.enabled),
        )
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
    if member.kind == "mcp":
        return AgentProfile(mcp_servers=(eval_mcp_fields(rt, member.fields, env),))
    if member.kind == "settings":
        return AgentProfile(settings=eval_settings_fields(rt, member.fields, env))
    if member.kind == "artifacts":
        return AgentProfile(artifacts=eval_artifacts_fields(rt, member.fields, env))
    if member.kind == "output":
        from ..lang.compiler.type_system import KediType

        output_type = KediType.from_node(
            member.type_expr,
            member.type_annotation,
        ).resolve_required(rt, env)
        return AgentProfile(output_type=output_type)
    if member.kind == "subagents":
        return AgentProfile(subagents=(member.name,))
    if member.kind == "max_agents":
        return AgentProfile(max_agents=member.value)
    if member.kind == "workflow":
        return AgentProfile(workflow_mode=member.mode)
    raise KediExecutionError(f"Unknown profile member kind: {member.kind!r}")


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
    from .tools import python_callable_for_tool

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
