from __future__ import annotations

from kedi.agent_profile.settings import (
    ACP_AGENT_SETTING_KEYS,
    CLAUDE_AGENT_SETTING_KEYS,
    CODEX_AGENT_SETTING_KEYS,
    DSPY_LM_SETTING_KEYS,
    LANGCHAIN_MODEL_SETTING_KEYS,
    MODEL_SETTING_KEYS,
    PYDANTIC_MODEL_SETTING_KEYS,
)
from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    ArtifactsField,
    CodeModeDirectiveStmt,
    EffortDirectiveStmt,
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    InputSegment,
    McpDirectiveStmt,
    McpField,
    ModelDirectiveStmt,
    ProcedureDef,
    ProfileAdapterEntry,
    ProfileAgentEntry,
    ProfileApprovalEntry,
    ProfileArtifactsEntry,
    ProfileCodeModeEntry,
    ProfileDefStmt,
    ProfileEffortEntry,
    ProfileHistoryEntry,
    ProfileHooksEntry,
    ProfileMcpEntry,
    ProfileModelEntry,
    ProfileSettingsEntry,
    ProfileSkillsEntry,
    ProfileSystemEntry,
    SettingsDirectiveStmt,
    SettingsField,
    SkillsDirectiveStmt,
    SystemDirectiveStmt,
    TextSegment,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypeRef,
    TypeUnion,
    UseDirectiveStmt,
)


def _agent_tool_markdown(name: str, proc: ProcedureDef) -> str:
    return (
        f"**Agent tool** `> use: {name}`\n\n{_proc_signature_markdown(proc)}\n\n"
        "Available to agent calls in the current scope."
    )


def _use_directive_markdown(stmt: UseDirectiveStmt) -> str:
    if stmt.tool_names:
        tools = ", ".join(f"`{name}`" for name in stmt.tool_names)
        return (
            f"`> use:`\n\nRegisters Kedi procedures as agent tools: {tools}.\n\n"
            "Inner `> use:` directives override outer tools in the current scope."
        )
    return (
        f"`> use: {stmt.name}`\n\n"
        "Configures the current scope for subsequent agent calls. A procedure target is "
        "registered as an agent tool; a profile target applies its agent configuration."
    )


def _model_directive_markdown(stmt: ModelDirectiveStmt) -> str:
    value = _model_value_preview(stmt)
    return f"`> model:` {value}\n\nUses this model for subsequent agent calls in the current scope."


def _agent_directive_markdown(stmt: AgentDirectiveStmt) -> str:
    value = _adapter_value_preview(stmt)
    command = _agent_command_preview(stmt)
    if command is not None:
        return (
            f"`> agent:` {value} `:` {command}\n\n"
            "Uses this command to run the selected agent harness for subsequent agent calls "
            "in the current scope."
        )
    return (
        f"`> agent:` {value}\n\n"
        "Routes subsequent agent calls in the current scope through this agent harness."
    )


def _adapter_directive_markdown(stmt: AdapterDirectiveStmt) -> str:
    value = _adapter_value_preview(stmt)
    return (
        f"`> adapter:` {value}\n\n"
        "Uses this model framework for subsequent model calls in the current scope."
    )


def _effort_directive_markdown(stmt: EffortDirectiveStmt) -> str:
    value = _effort_value_preview(stmt)
    return (
        f"`> effort:` {value}\n\n"
        "Requests this reasoning effort for subsequent model calls in the current scope. "
        "Support depends on the active adapter and model."
    )


def _approval_directive_markdown(stmt: ApprovalDirectiveStmt) -> str:
    value = _approval_value_preview(stmt)
    return (
        f"`> approval:` {value}\n\n"
        "Controls registered tool calls in the current scope. `allow` permits mutating "
        "tools; `deny` refuses them. An inline Python callable may return an "
        "`ApprovalDecision` to allow, deny, or edit arguments."
    )


def _hooks_directive_markdown(stmt: HooksDirectiveStmt) -> str:
    if stmt.enabled is False:
        value = "disabled"
    elif stmt.fields:
        value = ", ".join(f"`{field.name}`" for field in stmt.fields)
    else:
        value = "enabled"
    return (
        f"`> hooks:` {value}\n\n"
        "Registers ordered agent lifecycle handlers in the current scope. "
        "Handlers may inspect prompt submission and tool execution, edit prompts or tool "
        "arguments before transport, deny requests, and observe tool success or failure."
    )


def _history_directive_markdown(stmt: HistoryDirectiveStmt) -> str:
    value = "enabled" if stmt.enabled else "disabled"
    compaction = next(
        (field for field in stmt.fields if field.name.startswith("compaction_")),
        None,
    )
    configured = (
        " The expanded form also configures provider-native compaction."
        if compaction is not None
        else ""
    )
    return (
        f"`> history: {value}`\n\n"
        "Controls conversation continuity through stateful history for subsequent model calls "
        "in this lexical scope. "
        "`enabled` reuses adapter-scoped model history and provider cache identity; `disabled` "
        "runs stateless calls without mutating the surrounding history. The default is disabled."
        f"{configured}"
    )


def _skills_directive_markdown(stmt: SkillsDirectiveStmt) -> str:
    value = "enabled" if stmt.enabled else "disabled"
    return (
        f"`> skills: {value}`\n\n"
        "Controls on-demand skill discovery for subsequent agent calls in this lexical scope. "
        "Enabled scopes expose `list_skills` and `read_skill`. Sources are checked in order: "
        "the Kedi registry, project `.agents/skills`, then user `.agents/skills`. The expanded "
        "form configures `cwd`, `max_skills`, source merging, and excluded paths."
    )


def _codemode_directive_markdown(stmt: CodeModeDirectiveStmt) -> str:
    value = "enabled" if stmt.enabled else "disabled"
    return (
        f"`> codemode: {value}`\n\n"
        "Controls progressive tool discovery and bounded Monty execution for subsequent agent "
        "calls in this lexical scope. Enabled scopes expose `search_tools`, `get_tool_schema`, "
        "and `execute_code`; the expanded form configures discovery, hydration, execution, "
        "result, concurrency, and timeout limits. The default is disabled."
    )


def _system_directive_markdown(stmt: SystemDirectiveStmt) -> str:
    value = _system_value_preview(stmt)
    return (
        f"**`> system:`** {value}\n\n"
        "System instructions for subsequent model calls in the current scope. "
        "Text and `<substitutions>` are evaluated when this directive is applied."
    )


def _mcp_directive_markdown(stmt: McpDirectiveStmt) -> str:
    fields = ", ".join(f"`{field.name}`" for field in stmt.fields) or "`<empty>`"
    return (
        "**`> mcp:`**\n\n"
        "Adds tools from this MCP server to agent calls in the current scope. "
        f"Configured fields: {fields}."
    )


def _mcp_field_markdown(field: McpField) -> str:
    value = _mcp_field_value_preview(field)
    descriptions = {
        "transport": (
            "Connection transport for this MCP server. `http` is an alias for `streamable-http`."
        ),
        "command": "Command used to start a stdio MCP server.",
        "args": "String argument list passed to the stdio MCP command.",
        "env": "String environment dictionary passed to the stdio MCP process.",
        "url": "HTTP/SSE MCP server URL.",
        "headers": "String header dictionary sent to HTTP/SSE MCP servers.",
    }
    desc = descriptions.get(field.name, "MCP server configuration field.")
    if field.name == "transport":
        return (
            f'`transport: Literal["stdio", "sse", "http", "streamable-http"] =` {value}\n\n{desc}'
        )
    return f"`{field.name}` = {value}\n\n{desc}"


def _settings_directive_markdown(stmt: SettingsDirectiveStmt) -> str:
    fields = ", ".join(f"`{field.name}`" for field in stmt.fields) or "`<empty>`"
    return (
        "**`> settings:`**\n\nSet active model configuration for subsequent "
        "agent calls in the current scope. Configured fields: "
        f"{fields}."
    )


def _settings_field_markdown(field: SettingsField) -> str:
    value = _settings_field_value_preview(field)
    hook_descriptions = {
        "user_prompt_submit": "Runs before the rendered user prompt crosses the model transport boundary.",
        "pre_tool_use": "Runs before approval and tool execution; it may edit or deny canonical arguments.",
        "post_tool_use": "Observes the final model-visible result after successful tool execution.",
        "post_tool_use_failure": "Observes execution failures, excluding hook and approval denials.",
    }
    if field.name in hook_descriptions:
        return f"`{field.name}` = {value}\n\n{hook_descriptions[field.name]}"
    skill_descriptions = {
        "max_skills": "Maximum number of skill names exposed by `list_skills` in this scope.",
        "include_registry": "Include the user Kedi skill registry as the highest-priority source.",
        "include_all": "Merge all configured skill sources instead of using the first non-empty source.",
        "exclude_paths": "Paths excluded from skill discovery and direct skill reads.",
    }
    if field.name in skill_descriptions:
        return f"`{field.name}` = {value}\n\n{skill_descriptions[field.name]}"
    codemode_descriptions = {
        "default_search_limit": "Default number of tool names returned by `search_tools`.",
        "max_search_limit": "Maximum page size accepted by `search_tools`.",
        "max_hydrated_tools": "Maximum exact tool schemas hydrated during one agent run.",
        "max_discovery_result_bytes": "Maximum serialized result size for discovery controls.",
        "max_code_chars": "Maximum Monty source length accepted by `execute_code`.",
        "max_nested_calls": "Maximum host tool calls made by one `execute_code` invocation.",
        "max_concurrent_calls": "Maximum concurrently active host tool calls.",
        "max_tool_result_bytes": "Maximum serialized size of one nested host tool result.",
        "max_total_tool_result_bytes": "Maximum aggregate nested tool-result size per execution.",
        "max_print_bytes": "Maximum captured standard output from one Monty execution.",
        "request_timeout": "Timeout in seconds for one nested host tool call.",
    }
    if field.name in codemode_descriptions:
        return f"`{field.name}` = {value}\n\n{codemode_descriptions[field.name]}"
    if field.name == "compaction_mode":
        return (
            f'`compaction_mode: Literal["native", "disabled"]` = {value}\n\n'
            "Select provider-native conversation compaction or disable the inherited "
            "compaction policy while retaining stateful history."
        )
    if field.name == "compaction_threshold":
        return (
            f"`compaction_threshold: int | None` = {value}\n\n"
            "Positive input-token threshold that triggers native compaction. When omitted, "
            "the provider default is used."
        )
    adapters = _setting_adapter_label(field.name)
    return f"`{field.name}` = {value}\n\nSupported by: {adapters}."


def _artifacts_directive_markdown(stmt: ArtifactsDirectiveStmt) -> str:
    fields = ", ".join(f"`{field.name}`" for field in stmt.fields) or "`<empty>`"
    return (
        "**`> artifacts:`**\n\nConfigure large-value storage for subsequent "
        "model calls in the current scope. Native Kedi and Python reads retain "
        f"the original logical value. Configured fields: {fields}."
    )


def _artifacts_field_markdown(field: ArtifactsField) -> str:
    value = _artifacts_field_value_preview(field)
    descriptions = {
        "enabled": "Enable artifact creation and agent-facing artifact tools.",
        "query_artifacts": (
            "Expose the bounded lexical `query_artifact` tool: `enabled` or `disabled`. "
            "Defaults to `disabled`; an explicit scoped value overrides the adapter default."
        ),
        "store": "Payload backend: `memory` or `file`.",
        "path": "Root directory used by the file artifact store.",
        "threshold": "Minimum payload size before a value becomes an artifact.",
        "ttl": "Maximum lifetime measured from artifact creation.",
        "idle_ttl": "Optional expiry duration measured from the last access.",
        "preview_chars": "Maximum model-visible preview length.",
        "read_max_chars": "Hard character limit for one `read_artifact` call.",
        "session_quota": "Maximum active payload bytes in this artifact session.",
        "max_artifacts": "Maximum active artifact count in this session.",
        "cleanup_interval": "Interval used by background expiry cleanup.",
    }
    return f"`{field.name}` = {value}\n\n{descriptions[field.name]}"


def _profile_mcp_entry_markdown(entry: ProfileMcpEntry) -> str:
    fields = ", ".join(f"`{field.name}`" for field in entry.fields) or "`<empty>`"
    return (
        f"**`> mcp:`**\n\nMCP server configuration stored in this profile. "
        f"Fields: {fields}.\n\nApplied when the profile is used."
    )


def _profile_settings_entry_markdown(entry: ProfileSettingsEntry) -> str:
    fields = ", ".join(f"`{field.name}`" for field in entry.fields) or "`<empty>`"
    return (
        f"**`> settings:`**\n\nModel configuration stored in this profile. "
        f"Fields: {fields}.\n\nApplied when the profile is used."
    )


def _profile_artifacts_entry_markdown(entry: ProfileArtifactsEntry) -> str:
    fields = ", ".join(f"`{field.name}`" for field in entry.fields) or "`<empty>`"
    return (
        "**`> artifacts:`**\n\nArtifact policy stored in this profile. "
        f"Fields: {fields}.\n\nApplied when the profile is used."
    )


def _profile_agent_entry_markdown(entry: ProfileAgentEntry) -> str:
    value = _adapter_value_preview(entry)
    command = _agent_command_preview(entry)
    if command is not None:
        return (
            f"`> agent:` {value} `:` {command}\n\n"
            "Agent harness stored in this profile. Applied when the profile is used."
        )
    return (
        f"`> agent:` {value}\n\n"
        "Agent harness stored in this profile. Applied when the profile is used."
    )


def _profile_adapter_entry_markdown(entry: ProfileAdapterEntry) -> str:
    value = _adapter_value_preview(entry)
    return (
        f"`> adapter:` {value}\n\n"
        "Model framework stored in this profile. Applied when the profile is used."
    )


def _profile_model_entry_markdown(entry: ProfileModelEntry) -> str:
    value = _model_value_preview(entry)
    return (
        f"`> model:` {value}\n\n"
        "Model override stored in this profile. Applied when the profile is used."
    )


def _profile_system_entry_markdown(entry: ProfileSystemEntry) -> str:
    value = _system_value_preview(entry)
    return (
        f"**`> system:`** {value}\n\n"
        "System instructions stored in this profile. Applied when the profile is used."
    )


def _profile_effort_entry_markdown(entry: ProfileEffortEntry) -> str:
    value = _effort_value_preview(entry)
    return (
        f"`> effort:` {value}\n\n"
        "Reasoning effort stored in this profile. Applied when the profile is used."
    )


def _profile_approval_entry_markdown(entry: ProfileApprovalEntry) -> str:
    value = _approval_value_preview(entry)
    return (
        f"`> approval:` {value}\n\n"
        "Tool-call approval policy stored in this profile. Applied when the profile is used."
    )


def _profile_hooks_entry_markdown(entry: ProfileHooksEntry) -> str:
    if entry.enabled is False:
        value = "disabled"
    elif entry.fields:
        value = ", ".join(f"`{field.name}`" for field in entry.fields)
    else:
        value = "enabled"
    return (
        f"`> hooks:` {value}\n\n"
        "Ordered agent lifecycle handlers stored in this profile. Applied when the profile is used."
    )


def _profile_history_entry_markdown(entry: ProfileHistoryEntry) -> str:
    value = "enabled" if entry.enabled else "disabled"
    configured = " Includes a compaction policy." if entry.fields else ""
    return (
        f"`> history: {value}`\n\n"
        "Conversation continuity policy stored in this profile. Applied when the profile is used."
        f"{configured}"
    )


def _profile_skills_entry_markdown(entry: ProfileSkillsEntry) -> str:
    value = "enabled" if entry.enabled else "disabled"
    return (
        f"`> skills: {value}`\n\n"
        "Skill discovery policy stored in this profile. Applied when the profile is used."
    )


def _profile_codemode_entry_markdown(entry: ProfileCodeModeEntry) -> str:
    value = "enabled" if entry.enabled else "disabled"
    configured = " Includes bounded execution settings." if entry.fields else ""
    return (
        f"`> codemode: {value}`\n\n"
        "CodeMode tool-discovery and execution policy stored in this profile. "
        f"Applied when the profile is used.{configured}"
    )


def _effort_value_preview(stmt: EffortDirectiveStmt | ProfileEffortEntry) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _approval_value_preview(stmt: ApprovalDirectiveStmt | ProfileApprovalEntry) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _model_value_preview(stmt: ModelDirectiveStmt | ProfileModelEntry) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _adapter_value_preview(
    stmt: (AgentDirectiveStmt | AdapterDirectiveStmt | ProfileAgentEntry | ProfileAdapterEntry),
) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _agent_command_preview(stmt: AgentDirectiveStmt | ProfileAgentEntry) -> str | None:
    if stmt.command_backtick_code is not None:
        return f"`{stmt.command_backtick_code}`"
    if stmt.command_plain_value is not None:
        return f"`{stmt.command_plain_value}`"
    return None


def _system_value_preview(stmt: SystemDirectiveStmt | ProfileSystemEntry) -> str:
    if stmt.template_lines:
        parts: list[str] = []
        for index, line in enumerate(stmt.template_lines):
            if index:
                parts.append("\\n")
            for seg in line.segments:
                if isinstance(seg, TextSegment):
                    parts.append(seg.text)
                elif isinstance(seg, InputSegment):
                    parts.append(f"<{seg.name}>")
        text = "".join(parts).strip()
        return f"`{text}`" if text else "`<empty>`"
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _mcp_field_value_preview(field: McpField) -> str:
    if field.backtick_code is not None:
        return f"`{field.backtick_code}`"
    return f"`{field.plain_value or ''}`"


def _settings_field_value_preview(field: SettingsField) -> str:
    if field.backtick_code is not None:
        return f"`{field.backtick_code}`"
    return f"`{field.plain_value or ''}`"


def _artifacts_field_value_preview(field: ArtifactsField) -> str:
    if field.backtick_code is not None:
        return f"`{field.backtick_code}`"
    return f"`{field.plain_value or ''}`"


def _setting_adapter_label(name: str) -> str:
    if name == "tool_reason":
        return (
            "Kedi approval setting for Pydantic AI and LangChain; enabled or disabled "
            "(default disabled), or a backtick Python expression returning a boolean. "
            "Adds an optional reason to tools that may require approval. "
            "Overrides the adapter constructor in this scope; not sent to the model provider."
        )
    labels: list[str] = []
    if name in PYDANTIC_MODEL_SETTING_KEYS:
        labels.append("Pydantic AI `ModelSettings`")
    if name in DSPY_LM_SETTING_KEYS:
        labels.append("DSPy `LM`")
    if name in LANGCHAIN_MODEL_SETTING_KEYS:
        labels.append("LangChain model settings")
    if name in CLAUDE_AGENT_SETTING_KEYS:
        labels.append("Claude Agent SDK settings")
    if name in CODEX_AGENT_SETTING_KEYS:
        labels.append("Codex App Server settings")
    if name in ACP_AGENT_SETTING_KEYS:
        labels.append("ACP agent settings")
    if len(labels) == 1:
        return f"{labels[0]} only"
    if labels:
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"
    expected = ", ".join(sorted(MODEL_SETTING_KEYS))
    return f"no adapter; valid keys are {expected}"


def _profile_signature_markdown(profile: ProfileDefStmt) -> str:
    model_parts: list[str] = []
    for member in profile.model_members:
        if member.backtick_code is not None:
            model_parts.append(f"`{member.backtick_code}`")
        elif member.plain_value is not None:
            model_parts.append(f"`{member.plain_value}`")
    effort_parts: list[str] = []
    for member in profile.effort_members:
        effort_parts.append(_effort_value_preview(member))
    summary: list[str] = []
    if model_parts:
        summary.append(f"Model: {', '.join(model_parts)}")
    if effort_parts:
        summary.append(f"Effort: {', '.join(effort_parts)}")
    if any(
        member.template_lines or member.backtick_code is not None or member.plain_value is not None
        for member in profile.system_members
    ):
        summary.append("System instructions: configured")
    explicit_tools = (*profile.use_targets, *profile.tool_names)
    if explicit_tools:
        targets = ", ".join(f"`{name}`" for name in explicit_tools)
        summary.append(f"Uses: {targets}")
    if profile.mcp_members:
        summary.append(f"MCP servers: {len(profile.mcp_members)}")
    setting_count = sum(len(member.fields) for member in profile.settings_members)
    if setting_count:
        summary.append(f"Settings: {setting_count}")
    artifact_field_count = sum(len(member.fields) for member in profile.artifacts_members)
    if artifact_field_count:
        summary.append(f"Artifact settings: {artifact_field_count}")
    if profile.output_members:
        summary.append(f"Subagent output: `{profile.output_members[-1].type_annotation}`")
    workflow_mode = profile.workflow_members[-1].mode if profile.workflow_members else "delegate"
    summary.append(f"Subagent workflow: `{workflow_mode}`")

    text = f"**Agent profile** `> profile: {profile.name}:`"
    if profile.docstring:
        text += f"\n\n{profile.docstring}"
    else:
        text += "\n\nReusable agent profile."
    text += "\n\n" + "\n\n".join(summary)
    return text + "\n\nApplied when `> use:` runs."


def _proc_signature_markdown(proc: ProcedureDef) -> str:
    params = ", ".join(_format_proc_param(proc, p) for p in proc.params)
    sig = f"`@{proc.name}({params})`"
    if proc.return_type_annotation:
        sig += f" → `{proc.return_type_annotation}`"
    desc = "Kedi procedure."
    if proc.auto_spec is not None:
        desc = "AI-generated procedure (`> auto:`)."
    elif proc.optimize_spans:
        desc = (
            "Procedure with optimization spans: "
            + ", ".join(s.name for s in proc.optimize_spans)
            + "."
        )
    elif proc.captured_agent_profile is not None:
        captured = proc.captured_agent_profile
        profile_details: list[str] = []
        if captured.model is not None:
            profile_details.append(f"Model in this scope: `{captured.model}`.")
        if captured.effort is not None:
            profile_details.append(f"Reasoning effort in this scope: `{captured.effort}`.")
        if profile_details:
            desc = "Kedi procedure.\n\n" + "\n\n".join(profile_details)
    if proc.docstring:
        return f"{sig}\n\n{desc}\n\n{proc.docstring}"
    return f"{sig}\n\n{desc}"


def _type_signature_markdown(td: TypeDef) -> str:
    fields = ", ".join(_format_type_field(f) for f in td.fields)
    return (
        f"`~{td.name}({fields})`\n\n"
        "Structured Kedi type definition used for validation and output schema generation."
    )


def _type_expr_to_str(expr: TypeExpr | None) -> str:
    if expr is None:
        return "None"
    if isinstance(expr, TypeRef):
        return expr.name
    if isinstance(expr, TypeApply):
        return f"{expr.name}[{', '.join(_type_expr_to_str(arg) for arg in expr.args)}]"
    if isinstance(expr, TypeUnion):
        return f"{_type_expr_to_str(expr.left)} | {_type_expr_to_str(expr.right)}"
    return repr(expr)


def _format_proc_param(proc: ProcedureDef, name: str) -> str:
    piece = f"{name}: {_proc_param_annotation(proc, name)}"
    default = (proc.param_defaults or {}).get(name)
    if default is not None:
        piece += f" = `{default}`"
    return piece


def _format_type_field(field: TypeField) -> str:
    piece = f"{field.name}: {_type_field_annotation(field)}"
    if field.default_expr is not None:
        piece += f" = `{field.default_expr}`"
    return piece


def _proc_param_annotation(proc: ProcedureDef, name: str) -> str:
    return ((proc.param_type_annotations or {}).get(name) or "str").strip()


def _type_field_annotation(field: TypeField) -> str:
    return (field.type_annotation or "str").strip()


def _known_type_markdown(name: str, origin: str) -> str:
    if name == "Annotated":
        return (
            '`Annotated[T, "description"]`\n\n'
            "Adds field documentation without changing the runtime type `T`. The description "
            "is included in generated schemas and shown to compatible model adapters.\n\n"
            'Example: `Annotated[str, "A valid ISO country code"]`'
        )
    if origin == "datetime":
        descriptions = {
            "date": "Calendar date serialized as `YYYY-MM-DD`.",
            "time": "Time of day serialized in ISO 8601 format.",
            "datetime": "Date and time serialized in ISO 8601 format.",
            "timedelta": "Duration represented by Python's `datetime.timedelta`.",
        }
        return f"`{name}`\n\n{descriptions.get(name, f'Python datetime.{name} type.')}"
    if origin == "re":
        return "`Regex`\n\nCompiled regular-expression pattern."
    if origin == "pydantic":
        descriptions = {
            "Email": "Validated email address.",
            "HttpUrl": "Validated HTTP or HTTPS URL.",
            "FileUrl": "Validated file URL.",
        }
        return f"`{name}`\n\n{descriptions.get(name, 'Pydantic-validated value.')}"
    return f"`{name}`\n\nKedi type resolved to Python's {origin}.{name}."


def _prelude_type_markdown(name: str) -> str:
    return f"`{name}`\n\nPython type defined in this module's prelude and available to Kedi annotations."


def _python_module_scope_type_markdown(name: str) -> str:
    return (
        f"`{name}`\n\nPython type defined in a top-level Python block and available "
        "to Kedi annotations."
    )
