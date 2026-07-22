from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..agent_adapter.meta import AgentAdapterKind
from ..agent_adapter.registry import validate_adapter_kind
from ..agent_adapter.approval import ApprovalPolicy
from ..effort import EffortValue, validate_effort
from ..errors import KediExecutionError
from .mcp import McpServerSpec, mcp_spec_from_mapping
from .profile import AgentCommand, AgentProfile, SKILLS_USE_TARGET
from .registry import (
    ProfileAdapterMember,
    ProfileAgentMember,
    ProfileApprovalMember,
    ProfileDefinition,
    ProfileEffortMember,
    ProfileMcpMember,
    ProfileModelMember,
    ProfileSettingsMember,
    ProfileSystemMember,
    ProfileToolMember,
)
from .settings import parse_plain_setting_value

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import (
        AdapterDirectiveStmt,
        AgentDirectiveStmt,
        ApprovalDirectiveStmt,
        EffortDirectiveStmt,
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


def _member_to_patch(
    rt: KediRuntime,
    member: (
        ProfileModelMember
        | ProfileAgentMember
        | ProfileAdapterMember
        | ProfileEffortMember
        | ProfileApprovalMember
        | ProfileSystemMember
        | ProfileToolMember
        | ProfileMcpMember
        | ProfileSettingsMember
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
        tool_names = tuple(name for name in member.tool_names if name != SKILLS_USE_TARGET)
        return AgentProfile(
            tools=tool_names, skills=True if SKILLS_USE_TARGET in member.tool_names else None
        )
    if member.kind == "mcp":
        return AgentProfile(mcp_servers=(eval_mcp_fields(rt, member.fields, env),))
    if member.kind == "settings":
        return AgentProfile(settings=eval_settings_fields(rt, member.fields, env))
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
