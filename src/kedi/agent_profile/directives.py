from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ..agent_adapter.approvals.policy import ApprovalHandler, ApprovalPolicy
from ..agent_adapter.meta import AgentAdapterKind
from ..agent_adapter.registry import validate_adapter_kind
from ..agent_adapter.requirements import normalize_adapter_requirements
from ..artifacts import ArtifactPolicy, normalize_artifact_policy
from ..directive_values import parse_field, validate_field
from ..effort import EffortValue, validate_effort
from ..errors import KediExecutionError
from ..run_budget import RunBudget
from .codemode import normalize_codemode_settings
from .compaction import normalize_compaction_settings
from .connection import A2AAuthConfig, A2AAuthScheme, A2AConnection
from .definitions import (
    ProfileAdapterMember,
    ProfileAgentMember,
    ProfileApprovalMember,
    ProfileArtifactsMember,
    ProfileBudgetMember,
    ProfileCodeModeMember,
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
    ProfileWorkflowMember,
)
from .hooks import HookSettings, normalize_hook_settings
from .mcp import McpServerSpec, mcp_spec_from_mapping
from .profile import AgentCommand, AgentProfile

if TYPE_CHECKING:
    from kedi.core import KediRuntime
    from kedi.lang.ast import (
        AdapterDirectiveStmt,
        AgentDirectiveStmt,
        ApprovalDirectiveStmt,
        ArtifactsDirectiveStmt,
        ArtifactsField,
        BudgetDirectiveStmt,
        CodeModeDirectiveStmt,
        EffortDirectiveStmt,
        HistoryDirectiveStmt,
        HooksDirectiveStmt,
        McpDirectiveStmt,
        McpField,
        ModelDirectiveStmt,
        RequiresDirectiveStmt,
        SettingsDirectiveStmt,
        SettingsField,
        SkillsDirectiveStmt,
        SystemDirectiveStmt,
        TemplateLine,
    )


def _evaluate_directive_expression(rt: KediRuntime, code: str, env: dict[str, Any]) -> Any:
    from ..lang.compiler.environment import build_type_env

    return rt.executor.evaluate_type_expression(rt, code, build_type_env(rt, env))


def eval_model_value(
    rt: KediRuntime,
    *,
    directive_name: str = "model",
    plain_value: str | None,
    backtick_code: str | None,
    env: dict[str, Any],
) -> str:
    if backtick_code is not None:
        value = _evaluate_directive_expression(rt, backtick_code, env)
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
        value = _evaluate_directive_expression(rt, backtick_code, env)
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
        connection_fields=stmt.connection_fields,
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
    connection_fields: Sequence[Any] = (),
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
    a2a_connection: A2AConnection | None = None
    if connection_fields:
        if command_plain_value is not None or command_backtick_code is not None:
            raise KediExecutionError(
                "agent connection fields cannot be combined with the legacy command form"
            )
        if shortname == "acp":
            values = _agent_connection_values(rt, connection_fields, env=env)
            unknown = set(values).difference({"command"})
            if unknown:
                raise KediExecutionError(
                    f"ACP agent connection has unsupported field(s): {', '.join(sorted(unknown))}"
                )
            if "command" not in values:
                raise KediExecutionError("ACP agent connection requires 'command'")
            command = _normalize_agent_command(values["command"], directive_name="agent command")
        elif shortname == "a2a":
            a2a_connection = _eval_a2a_connection(rt, connection_fields, env=env)
        else:
            raise KediExecutionError(f"Agent {shortname!r} does not accept a connection body")
    elif command_plain_value is not None or command_backtick_code is not None:
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
        a2a_connection=a2a_connection,
    )


def _normalize_agent_command(value: Any, *, directive_name: str) -> AgentCommand:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = tuple(value)
        if all(isinstance(part, str) for part in parts):
            return parts
    raise TypeError(
        f"{directive_name} must evaluate to str or sequence[str], got {type(value).__name__}"
    )


def _agent_connection_values(
    rt: KediRuntime,
    fields: Sequence[Any],
    *,
    env: dict[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in fields:
        if item.name in values:
            raise KediExecutionError(f"duplicate agent connection field {item.name!r}")
        if item.fields:
            values[item.name] = _agent_connection_values(rt, item.fields, env=env)
            continue
        if item.backtick_code is not None:
            values[item.name] = _evaluate_directive_expression(rt, item.backtick_code, env)
        elif item.plain_value is not None:
            values[item.name] = item.plain_value.strip()
        else:
            raise KediExecutionError(f"agent connection field {item.name!r} has no value")
    return values


def _eval_a2a_connection(
    rt: KediRuntime,
    fields: Sequence[Any],
    *,
    env: dict[str, Any],
) -> A2AConnection:
    values = _agent_connection_values(rt, fields, env=env)
    allowed = {"endpoint", "auth", "request_timeout", "task_timeout", "poll_interval"}
    unknown = set(values).difference(allowed)
    if unknown:
        raise KediExecutionError(
            f"A2A agent connection has unsupported field(s): {', '.join(sorted(unknown))}"
        )
    endpoint = values.get("endpoint")
    if not isinstance(endpoint, str):
        raise KediExecutionError("A2A agent connection requires a string 'endpoint'")
    auth_value = values.get("auth")
    if auth_value is not None and not isinstance(auth_value, dict):
        raise KediExecutionError("A2A agent 'auth' must be a nested block")
    auth = None if auth_value is None else _a2a_auth_from_values(auth_value)
    timeout_values: dict[str, float] = {}
    for name in ("request_timeout", "task_timeout", "poll_interval"):
        if name not in values:
            continue
        value = values[name]
        try:
            timeout_values[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise KediExecutionError(f"A2A agent {name} must be a positive number") from exc
    return A2AConnection(endpoint=endpoint, auth=auth, **timeout_values)


def _a2a_auth_from_values(values: dict[str, Any]) -> A2AAuthConfig:
    allowed = {
        "scheme",
        "username",
        "password_env",
        "password_file",
        "token_env",
        "token_file",
        "api_key_env",
        "api_key_file",
        "credential_env",
        "credential_file",
        "provider",
        "header",
    }
    unknown = set(values).difference(allowed)
    if unknown:
        raise KediExecutionError(f"A2A auth has unsupported field(s): {', '.join(sorted(unknown))}")
    scheme = values.get("scheme")
    if not isinstance(scheme, str):
        raise KediExecutionError("A2A auth requires a string 'scheme'")
    source_names = {
        "basic": ("password_env", "password_file"),
        "bearer": ("token_env", "token_file"),
        "api_key": ("api_key_env", "api_key_file"),
    }.get(scheme)
    if source_names is None:
        raise KediExecutionError("A2A auth scheme must be basic, bearer, or api_key")
    credential_fields = {
        "password_env",
        "password_file",
        "token_env",
        "token_file",
        "api_key_env",
        "api_key_file",
        "credential_env",
        "credential_file",
        "provider",
    }
    supplied = credential_fields.intersection(values)
    if len(supplied) != 1:
        raise ValueError("A2A auth must define exactly one credential source")
    if not supplied.issubset({*source_names, "credential_env", "credential_file", "provider"}):
        raise ValueError(f"A2A credential source does not match auth scheme {scheme!r}")
    env_name = values.get(source_names[0], values.get("credential_env"))
    file_name = values.get(source_names[1], values.get("credential_file"))
    for name, value in (("credential environment", env_name), ("credential file", file_name)):
        if value is not None and not isinstance(value, str):
            raise KediExecutionError(f"A2A auth {name} must be a string")
    username = values.get("username")
    header = values.get("header")
    if username is not None and not isinstance(username, str):
        raise KediExecutionError("A2A auth username must be a string")
    if header is not None and not isinstance(header, str):
        raise KediExecutionError("A2A auth header must be a string")
    return A2AAuthConfig(
        scheme=cast(A2AAuthScheme, scheme),
        username=username,
        secret_env=env_name,
        secret_file=file_name,
        provider=values.get("provider"),
        header=header,
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
        return AgentProfile(approval=ApprovalPolicy.dynamic(cast(ApprovalHandler, handler)))
    if stmt.plain_value is None:
        raise ValueError("approval directive has no value")
    if stmt.plain_value == "allow":
        return AgentProfile(approval=ApprovalPolicy.allow())
    if stmt.plain_value == "deny":
        return AgentProfile(approval=ApprovalPolicy.deny())
    raise ValueError("approval directive must be 'allow', 'deny', or an inline Python handler")


def eval_hooks_directive(
    rt: KediRuntime,
    stmt: HooksDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return _eval_hooks_settings(rt, enabled=stmt.enabled, fields=stmt.fields, env=env)


def _eval_hooks_settings(
    rt: KediRuntime,
    *,
    enabled: bool | None,
    fields: list[SettingsField] | tuple[SettingsField, ...],
    env: dict[str, Any],
) -> AgentProfile:
    settings = HookSettings(enabled=enabled)
    for field in fields:
        if field.name == "enabled":
            value = eval_settings_field_value(rt, field, env, scope="hooks")
            settings = settings.merge(HookSettings(enabled=value))
            continue
        if field.backtick_code is None:
            raise TypeError(f"Hook event {field.name!r} requires an inline Python value")
        value = rt.executor.evaluate_inline(rt, field.backtick_code, env)
        patch = normalize_hook_settings({field.name: value})
        assert patch is not None
        settings = settings.merge(patch)
    return AgentProfile(hooks=settings)


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
        return validate_field(
            "mcp", field.name, rt.executor.evaluate_inline(rt, field.backtick_code, env)
        )
    if field.plain_value is not None:
        return parse_field("mcp", field.name, field.plain_value)
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


def eval_settings_field_value(
    rt: KediRuntime, field: SettingsField, env: dict[str, Any], *, scope: str = "settings"
) -> Any:
    if field.backtick_code is None and field.plain_value is None:
        return tuple(field.plain_values) if field.plain_values is not None else None
    # Profiles are reusable patches. Their SDK option domains belong to the
    # final adapter, which may differ from the caller evaluating this patch.
    if field.backtick_code is not None:
        return validate_field(
            scope,
            field.name,
            rt.executor.evaluate_inline(rt, field.backtick_code, env),
        )
    assert field.plain_value is not None
    return parse_field(scope, field.name, field.plain_value)


def eval_settings_fields(
    rt: KediRuntime,
    fields: list[SettingsField] | tuple[SettingsField, ...],
    env: dict[str, Any],
    *,
    scope: str = "settings",
) -> dict[str, Any]:
    return {field.name: eval_settings_field_value(rt, field, env, scope=scope) for field in fields}


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
        value = validate_field(
            "artifacts", field.name, _evaluate_directive_expression(rt, field.backtick_code, env)
        )
    elif field.plain_value is not None:
        value = parse_field("artifacts", field.name, field.plain_value)
    else:
        return None
    if field.name == "query_artifacts":
        return "enabled" if value else "disabled"
    return value


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
) -> AgentProfile:
    from .history import HistoryProcessingSettings, bind_history_processor

    values = eval_settings_fields(rt, fields, env, scope="history")
    enabled = values.pop("enabled", enabled)
    compaction_values = {
        name.removeprefix("compaction_"): value
        for name, value in values.items()
        if name in {"compaction_mode", "compaction_threshold"}
    }
    compaction = normalize_compaction_settings(compaction_values or None)
    if not enabled and compaction is not None and compaction.mode == "native":
        raise KediExecutionError("native compaction requires history to be enabled")
    return AgentProfile(
        history=enabled,
        compaction=compaction,
        history_processing=(
            HistoryProcessingSettings(
                processor=(
                    bind_history_processor(rt, values["processor"])
                    if "processor" in values
                    else None
                ),
                condition=(
                    bind_history_processor(rt, values["processor_condition"])
                    if "processor_condition" in values
                    else None
                ),
                inherit_processor="processor" not in values,
            )
            if "processor" in values or "processor_condition" in values
            else None
        ),
    )


def eval_history_directive(
    rt: KediRuntime,
    stmt: HistoryDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    return _eval_history_fields(
        rt,
        enabled=stmt.enabled,
        fields=stmt.fields,
        env=env,
    )


def eval_skills_directive(
    rt: KediRuntime,
    stmt: SkillsDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    from .skills import normalize_skills_settings

    values = eval_settings_fields(rt, stmt.fields, env, scope="skills")
    return AgentProfile(
        skills=normalize_skills_settings(values, enabled=values.pop("enabled", stmt.enabled)),
    )


def eval_codemode_directive(
    rt: KediRuntime,
    stmt: CodeModeDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    values = eval_settings_fields(rt, stmt.fields, env, scope="codemode")
    values.setdefault("enabled", stmt.enabled)
    return AgentProfile(codemode=normalize_codemode_settings(values))


def eval_requires_directive(
    rt: KediRuntime,
    stmt: RequiresDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    del rt, env
    requirements = normalize_adapter_requirements(stmt.requirements)
    return AgentProfile(
        requirements=requirements,
        requirement_origins={
            requirement: (stmt.requirement_locs[requirement],)
            for requirement in requirements
            if requirement in stmt.requirement_locs
        },
    )


def eval_budget_directive(
    rt: KediRuntime,
    stmt: BudgetDirectiveStmt,
    env: dict[str, Any],
) -> AgentProfile:
    del rt, env
    return AgentProfile(
        run_budget=RunBudget(
            request_limit=stmt.request_limit,
            tool_attempt_limit=stmt.tool_attempt_limit,
        )
    )


def _member_to_patch(
    rt: KediRuntime,
    member: (
        ProfileModelMember
        | ProfileAgentMember
        | ProfileAdapterMember
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
            connection_fields=member.connection_fields,
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
            return AgentProfile(approval=ApprovalPolicy.dynamic(cast(ApprovalHandler, handler)))
        if member.plain_value == "allow":
            return AgentProfile(approval=ApprovalPolicy.allow())
        if member.plain_value == "deny":
            return AgentProfile(approval=ApprovalPolicy.deny())
        raise ValueError("profile approval must be 'allow' or 'deny'")
    if member.kind == "hooks":
        return _eval_hooks_settings(
            rt,
            enabled=member.enabled,
            fields=member.fields,
            env=env,
        )
    if member.kind == "history":
        return _eval_history_fields(
            rt,
            enabled=member.enabled,
            fields=member.fields,
            env=env,
        )
    if member.kind == "skills":
        from .skills import normalize_skills_settings

        values = eval_settings_fields(rt, member.fields, env, scope="skills")
        return AgentProfile(
            skills=normalize_skills_settings(values, enabled=values.pop("enabled", member.enabled)),
        )
    if member.kind == "codemode":
        values = eval_settings_fields(rt, member.fields, env, scope="codemode")
        values.setdefault("enabled", member.enabled)
        return AgentProfile(codemode=normalize_codemode_settings(values))
    if member.kind == "requires":
        requirements = normalize_adapter_requirements(list(member.requirements))
        return AgentProfile(
            requirements=requirements,
            requirement_origins={
                requirement: (member.requirement_locs[requirement],)
                for requirement in requirements
                if requirement in member.requirement_locs
            },
        )
    if member.kind == "budget":
        return AgentProfile(
            run_budget=RunBudget(
                request_limit=member.request_limit,
                tool_attempt_limit=member.tool_attempt_limit,
            )
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
