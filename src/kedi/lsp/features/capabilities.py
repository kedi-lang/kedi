from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from lsprotocol import types as lsp

from kedi.agent_adapter.history_processing import HISTORY_PROCESSOR_CACHE_WARNING
from kedi.agent_adapter.meta import AdapterCapability, AgentAdapterKind
from kedi.agent_adapter.registry import (
    adapter_capabilities,
    adapter_factory,
    known_adapter_shortnames,
)
from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    BudgetDirectiveStmt,
    CodeModeDirectiveStmt,
    ConditionalLoopStmt,
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    IfStmt,
    LoopStmt,
    McpDirectiveStmt,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    ProfileBudgetEntry,
    ProfileDefStmt,
    Program,
    ProgramWithValidation,
    RequiresDirectiveStmt,
    SettingsDirectiveStmt,
    SettingsField,
    SkillsDirectiveStmt,
    Span,
    TaskGroupStmt,
    TaskStmt,
    TemplateBlock,
    TemplateLine,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeLiteral,
    TypePython,
    TypeRef,
    TypeUnion,
    UseDirectiveStmt,
)
from kedi.lang.compiler.environment import KEDI_RUNTIME_TYPE_JSON_SCHEMA_FORMATS
from kedi.lsp.coordinates import span_to_range as _span_to_range
from kedi.lsp.features.capability_state import (
    _adapter_capability_status,
    _adapter_state_from_plain_value,
    _adapter_supports,
    _AdapterDiagnosticState,
    _merge_adapter_state,
    _with_history_state,
    _with_hooks_state,
    _with_requirements_state,
    _activation_state,
)
from kedi.lsp.features.symbols import (
    _approval_value_span,
    _directive_keyword_span,
    _directive_name_span,
    _use_single_target_span,
    _use_tool_entry_spans,
)


def _profile_capability_diagnostics(
    prog: Program | ProgramWithValidation,
    profile: ProfileDefStmt,
    state: _AdapterDiagnosticState,
    warned_approval_spans: set[tuple[str | None, int, int]],
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for member in profile.agent_members:
        diagnostics.extend(
            _adapter_selection_diagnostics(
                prog,
                shortname=member.plain_value,
                expected_kind="agent-harness",
                loc=member.loc,
            )
        )
    for member in profile.adapter_members:
        diagnostics.extend(
            _adapter_selection_diagnostics(
                prog,
                shortname=member.plain_value,
                expected_kind="agent-framework",
                loc=member.loc,
            )
        )
    for member in profile.mcp_members:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                state,
                "mcp",
                [_directive_keyword_span(prog, member.loc, "mcp") or member.loc],
                "MCP server directives",
            )
        )
    for member in profile.artifacts_members:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                state,
                "artifacts",
                [_directive_keyword_span(prog, member.loc, "artifacts") or member.loc],
                "model-facing artifact tools",
            )
        )
    for member in profile.budget_members:
        diagnostics.extend(_budget_capability_diagnostics(state, member))
    for member in profile.codemode_members:
        if _activation_state(member.enabled, member.fields):
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    state,
                    "codemode",
                    [_directive_keyword_span(prog, member.loc, "codemode") or member.loc],
                    "CodeMode tool execution",
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-ADAPTER-CODEMODE",
                )
            )
    diagnostics.extend(_explicit_requirement_diagnostics(state))
    diagnostics.extend(_hooks_capability_diagnostics(state))
    diagnostics.extend(_history_capability_diagnostics(state))
    for member in profile.history_members:
        diagnostics.extend(_compaction_capability_diagnostics(state, member.fields))
        diagnostics.extend(_history_prefix_cache_warning(member.fields))
    diagnostics.extend(
        _unsupported_capability_diagnostics(
            state,
            "subagents",
            [entry.loc for entry in profile.subagent_members],
            "Subagent delegation",
            severity=lsp.DiagnosticSeverity.Error,
            code="K-ADAPTER-SUBAGENTS",
        )
    )
    diagnostics.extend(
        _native_approval_capability_diagnostics(
            prog,
            state,
            warned_approval_spans,
        )
    )
    return diagnostics


def _scan_capability_scope(
    prog: Program | ProgramWithValidation,
    statements: Sequence[object],
    state: _AdapterDiagnosticState,
    profile_defs: dict[str, ProfileDefStmt],
    profile_states: dict[str, _AdapterDiagnosticState],
    proc_names: set[str],
    type_defs: dict[str, TypeDef],
    warned_approval_spans: set[tuple[str | None, int, int]] | None = None,
    active_profile_name: str | None = None,
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    active = state
    selected_profile = active_profile_name
    if warned_approval_spans is None:
        warned_approval_spans = set()

    expanded = []
    for stmt in statements:
        if isinstance(stmt, UseDirectiveStmt) and stmt.tool_names:
            entries = dict(_use_tool_entry_spans(prog, stmt))
            expanded.extend(
                UseDirectiveStmt(target=name, loc=entries.get(name, stmt.loc))
                for name in stmt.tool_names
            )
        else:
            expanded.append(stmt)
    for stmt in expanded:
        if isinstance(stmt, AgentDirectiveStmt):
            before_selection = active
            diagnostics.extend(
                _adapter_selection_diagnostics(
                    prog,
                    shortname=stmt.plain_value,
                    expected_kind="agent-harness",
                    loc=stmt.loc,
                )
            )
            selected = _adapter_state_from_plain_value(stmt.plain_value, "agent-harness")
            if selected is not None:
                active = _merge_adapter_state(active, selected)
                diagnostics.extend(
                    _newly_unsupported_requirement_diagnostics(before_selection, active)
                )
                diagnostics.extend(
                    _native_approval_capability_diagnostics(
                        prog,
                        active,
                        warned_approval_spans,
                    )
                )
                diagnostics.extend(_history_capability_diagnostics(active))
                diagnostics.extend(_codemode_capability_diagnostics(active))
                diagnostics.extend(_hooks_capability_diagnostics(active))
            continue
        if isinstance(stmt, AdapterDirectiveStmt):
            before_selection = active
            diagnostics.extend(
                _adapter_selection_diagnostics(
                    prog,
                    shortname=stmt.plain_value,
                    expected_kind="agent-framework",
                    loc=stmt.loc,
                )
            )
            selected = _adapter_state_from_plain_value(stmt.plain_value, "agent-framework")
            if selected is not None:
                active = _merge_adapter_state(active, selected)
                diagnostics.extend(
                    _newly_unsupported_requirement_diagnostics(before_selection, active)
                )
                diagnostics.extend(
                    _native_approval_capability_diagnostics(
                        prog,
                        active,
                        warned_approval_spans,
                    )
                )
                diagnostics.extend(_history_capability_diagnostics(active))
                diagnostics.extend(_codemode_capability_diagnostics(active))
                diagnostics.extend(_hooks_capability_diagnostics(active))
            continue
        if isinstance(stmt, ApprovalDirectiveStmt):
            active = replace(
                active,
                approval_loc=stmt.loc,
                approval_plain_value=stmt.plain_value,
            )
            diagnostics.extend(
                _native_approval_capability_diagnostics(
                    prog,
                    active,
                    warned_approval_spans,
                )
            )
            continue
        if isinstance(stmt, HooksDirectiveStmt):
            active = _with_hooks_state(active, stmt)
            diagnostics.extend(_hooks_capability_diagnostics(active))
            continue
        if isinstance(stmt, HistoryDirectiveStmt):
            active = _with_history_state(active, stmt)
            diagnostics.extend(_history_capability_diagnostics(active))
            diagnostics.extend(_compaction_capability_diagnostics(active, stmt.fields))
            diagnostics.extend(_history_prefix_cache_warning(stmt.fields))
            continue
        if isinstance(stmt, SkillsDirectiveStmt):
            if _activation_state(stmt.enabled, stmt.fields):
                diagnostics.extend(
                    _unsupported_capability_diagnostics(
                        active,
                        "tool_registration",
                        [_directive_keyword_span(prog, stmt.loc, "skills") or stmt.loc],
                        "skill tool registration",
                    )
                )
            continue
        if isinstance(stmt, CodeModeDirectiveStmt):
            active = replace(
                active,
                codemode_loc=stmt.loc,
                codemode_enabled=_activation_state(stmt.enabled, stmt.fields),
            )
            diagnostics.extend(_codemode_capability_diagnostics(active))
            continue
        if isinstance(stmt, RequiresDirectiveStmt):
            active = _with_requirements_state(active, stmt)
            requirement_entries: tuple[tuple[AdapterCapability, Span | None], ...] = tuple(
                (
                    cast(AdapterCapability, requirement),
                    stmt.requirement_locs.get(requirement),
                )
                for requirement in stmt.requirements
            )
            diagnostics.extend(
                _explicit_requirement_diagnostics(active, entries=requirement_entries)
            )
            continue
        if isinstance(stmt, BudgetDirectiveStmt):
            diagnostics.extend(_budget_capability_diagnostics(active, stmt))
            continue
        if isinstance(stmt, McpDirectiveStmt):
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    active,
                    "mcp",
                    [_directive_keyword_span(prog, stmt.loc, "mcp") or stmt.loc],
                    "MCP server directives",
                )
            )
            continue
        if isinstance(stmt, ArtifactsDirectiveStmt):
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    active,
                    "artifacts",
                    [_directive_keyword_span(prog, stmt.loc, "artifacts") or stmt.loc],
                    "model-facing artifact tools",
                )
            )
            continue
        if isinstance(stmt, SettingsDirectiveStmt):
            continue
        if isinstance(stmt, UseDirectiveStmt):
            diagnostics.extend(
                _use_capability_diagnostics(
                    prog,
                    stmt,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                )
            )
            if stmt.name and not stmt.tool_names and stmt.name in profile_defs:
                if profile_defs[stmt.name].kind != "capability":
                    selected_profile = stmt.name
                active = _merge_adapter_state(
                    active,
                    profile_states.get(stmt.name, _AdapterDiagnosticState()),
                )
                diagnostics.extend(
                    _native_approval_capability_diagnostics(
                        prog,
                        active,
                        warned_approval_spans,
                    )
                )
                diagnostics.extend(_history_capability_diagnostics(active))
            continue
        if isinstance(stmt, TemplateBlock):
            diagnostics.extend(_template_capability_diagnostics(active, stmt, type_defs))
            continue
        if isinstance(stmt, TaskStmt):
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    active,
                    "background_subagents",
                    [stmt.subagent_loc],
                    "background subagents",
                    severity=lsp.DiagnosticSeverity.Error,
                )
            )
            if stmt.subagent not in profile_defs and stmt.subagent_loc is not None:
                diagnostics.append(
                    lsp.Diagnostic(
                        range=_span_to_range(stmt.subagent_loc),
                        severity=lsp.DiagnosticSeverity.Error,
                        code="K-TASK-SUBAGENT",
                        source="kedi",
                        message=f"Unknown subagent profile {stmt.subagent!r}.",
                    )
                )
            elif (
                stmt.subagent_loc is not None
                and selected_profile in profile_defs
                and stmt.subagent
                not in {member.name for member in profile_defs[selected_profile].subagent_members}
            ):
                diagnostics.append(
                    lsp.Diagnostic(
                        range=_span_to_range(stmt.subagent_loc),
                        severity=lsp.DiagnosticSeverity.Error,
                        code="K-TASK-NOT-CHILD",
                        source="kedi",
                        message=(
                            f"Profile {stmt.subagent!r} is not a direct child "
                            f"of {selected_profile!r}."
                        ),
                    )
                )
            continue
        if isinstance(stmt, TemplateLine):
            diagnostics.extend(_template_line_capability_diagnostics(active, stmt, type_defs))
            continue
        if isinstance(stmt, TaskGroupStmt):
            for arm in stmt.arms:
                diagnostics.extend(
                    _scan_capability_scope(
                        prog,
                        arm.body or [],
                        active,
                        profile_defs,
                        profile_states,
                        proc_names,
                        type_defs,
                        warned_approval_spans,
                        selected_profile,
                    )
                )
            continue
        if isinstance(stmt, IfStmt):
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                    selected_profile,
                )
            )
            if stmt.else_body is not None:
                diagnostics.extend(
                    _scan_capability_scope(
                        prog,
                        stmt.else_body,
                        active,
                        profile_defs,
                        profile_states,
                        proc_names,
                        type_defs,
                        warned_approval_spans,
                        selected_profile,
                    )
                )
            continue
        if isinstance(stmt, LoopStmt):
            loop_statements = list(stmt.body)
            if stmt.map_body is not None:
                loop_statements.extend(stmt.map_body)
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    loop_statements,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                    selected_profile,
                )
            )
            continue
        if isinstance(stmt, ConditionalLoopStmt):
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                    selected_profile,
                )
            )
            continue
        if isinstance(stmt, ProcedureDef):
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                    selected_profile,
                )
            )
            continue
        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.proc.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                    selected_profile,
                )
            )
            continue
    return diagnostics


def _budget_capability_diagnostics(
    state: _AdapterDiagnosticState,
    directive: BudgetDirectiveStmt | ProfileBudgetEntry,
) -> list[lsp.Diagnostic]:
    loc = directive.loc
    if loc is None:
        return []
    diagnostics: list[lsp.Diagnostic] = []
    for field, capability in (
        ("request_limit", "request_budget"),
        ("tool_attempt_limit", "tool_attempt_budget"),
    ):
        if getattr(directive, field) is not None:
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    state,
                    cast(AdapterCapability, capability),
                    [loc],
                    f"{field} run budget",
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-ADAPTER-BUDGET",
                )
            )
    return diagnostics


def _compaction_capability_diagnostics(
    state: _AdapterDiagnosticState,
    fields: Sequence[SettingsField],
) -> list[lsp.Diagnostic]:
    mode = next((field for field in fields if field.name == "compaction_mode"), None)
    threshold = next(
        (field for field in fields if field.name == "compaction_threshold"),
        None,
    )
    configured = mode or threshold
    if configured is None or configured.loc is None:
        return []
    if mode is not None and mode.plain_value == "disabled":
        return []
    if mode is not None and mode.backtick_code is not None:
        return []
    return _unsupported_capability_diagnostics(
        state,
        "native_compaction",
        [configured.loc],
        "provider-native conversation compaction",
        severity=lsp.DiagnosticSeverity.Error,
        code="K-ADAPTER-COMPACTION",
    )


def _history_capability_diagnostics(
    state: _AdapterDiagnosticState,
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    if state.history_enabled is True and state.history_loc is not None:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                state,
                "stateful_history",
                [state.history_loc],
                "stateful conversation history",
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-HISTORY",
            )
        )
    processing_loc = (
        state.history_processor_loc
        if state.history_processor_enabled is True
        else state.history_condition_loc
        if state.history_condition_enabled is True
        else None
    )
    if processing_loc is not None:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                state,
                "history_processing",
                [processing_loc],
                "native history processing",
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-HISTORY-PROCESSOR",
            )
        )
    return diagnostics


def _history_prefix_cache_warning(fields: Sequence[SettingsField]) -> list[lsp.Diagnostic]:
    processor = next((field for field in fields if field.name == "processor"), None)
    if processor is None or processor.loc is None:
        return []
    if (processor.backtick_code or processor.plain_value or "").strip() == "None":
        return []
    return [
        lsp.Diagnostic(
            range=_span_to_range(processor.loc),
            severity=lsp.DiagnosticSeverity.Warning,
            code="K-HISTORY-PREFIX-CACHE",
            source="kedi",
            message=HISTORY_PROCESSOR_CACHE_WARNING,
        )
    ]


def _codemode_capability_diagnostics(
    state: _AdapterDiagnosticState,
) -> list[lsp.Diagnostic]:
    if state.codemode_enabled is not True or state.codemode_loc is None:
        return []
    return _unsupported_capability_diagnostics(
        state,
        "codemode",
        [state.codemode_loc],
        "CodeMode tool execution",
        severity=lsp.DiagnosticSeverity.Error,
        code="K-ADAPTER-CODEMODE",
    )


def _hooks_capability_diagnostics(
    state: _AdapterDiagnosticState,
    *,
    span: Span | None = None,
) -> list[lsp.Diagnostic]:
    if state.hooks_enabled is not True or state.hooks_loc is None:
        return []
    diagnostic_span = span or state.hooks_loc
    generic = _unsupported_capability_diagnostics(
        state,
        "hooks",
        [diagnostic_span],
        "agent lifecycle hooks",
        severity=lsp.DiagnosticSeverity.Error,
        code="K-ADAPTER-HOOKS",
    )
    if generic or state.shortname is None or not state.hook_events:
        return generic
    try:
        hook_capabilities = adapter_capabilities(state.shortname).hooks
    except Exception:  # pragma: no cover - keep LSP diagnostics resilient
        hook_capabilities = None
    if hook_capabilities is None:
        return generic
    unsupported = hook_capabilities.unsupported(state.hook_events)
    if not unsupported:
        return []
    names = ", ".join(unsupported)
    return [
        lsp.Diagnostic(
            range=_span_to_range(diagnostic_span),
            severity=lsp.DiagnosticSeverity.Error,
            code="K-ADAPTER-HOOKS",
            source="kedi",
            message=(f"Adapter '{state.shortname}' does not support agent hook events: {names}."),
        )
    ]


def _adapter_selection_diagnostics(
    prog: Program | ProgramWithValidation,
    *,
    shortname: str | None,
    expected_kind: AgentAdapterKind,
    loc: Span | None,
) -> list[lsp.Diagnostic]:
    """Report invalid literal ``> agent:`` / ``> adapter:`` selections.

    Backtick values are resolved by the runtime, so only literal shortnames can
    be diagnosed statically here.
    """
    if shortname is None or loc is None:
        return []

    directive = "agent" if expected_kind == "agent-harness" else "adapter"
    kind_label = "agent harness" if expected_kind == "agent-harness" else "agent framework"
    span = _directive_name_span(prog, loc, shortname) or loc
    try:
        factory = adapter_factory(shortname)
    except Exception:
        known = ", ".join(
            name
            for name in known_adapter_shortnames()
            if adapter_factory(name).kind == expected_kind
        )
        return [
            lsp.Diagnostic(
                range=_span_to_range(span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-NOT-FOUND",
                source="kedi",
                message=(
                    f"Unknown {kind_label} adapter {shortname!r}. "
                    f"Known {kind_label} adapters: {known}"
                ),
            )
        ]
    if factory.kind == expected_kind:
        return []

    actual = "agent harness" if factory.kind == "agent-harness" else "agent framework"
    return [
        lsp.Diagnostic(
            range=_span_to_range(span),
            severity=lsp.DiagnosticSeverity.Error,
            code="K-ADAPTER-KIND",
            source="kedi",
            message=(f"> {directive}: cannot select {shortname!r}; it is an {actual} adapter."),
        )
    ]


def _use_capability_diagnostics(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
    state: _AdapterDiagnosticState,
    profile_defs: dict[str, ProfileDefStmt],
    profile_states: dict[str, _AdapterDiagnosticState],
    proc_names: set[str],
) -> list[lsp.Diagnostic]:
    if stmt.tool_names:
        spans = [span for _, span in _use_tool_entry_spans(prog, stmt)]
        return _unsupported_capability_diagnostics(
            state,
            "tool_registration",
            spans or [stmt.loc],
            "agent tool registration",
        )
    if not stmt.name:
        return []

    target_span = _use_single_target_span(prog, stmt)
    if target_span is None:
        target_span = stmt.loc
    if target_span is None:
        return []
    if stmt.name == "skills" and stmt.name not in proc_names and stmt.name not in profile_defs:
        return [
            lsp.Diagnostic(
                range=_span_to_range(target_span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-SKILLS-USE",
                source="kedi",
                message="Skills use a standalone directive; replace '> use: skills' with '> skills: enabled'.",
            )
        ]
    if stmt.name in proc_names:
        return _unsupported_capability_diagnostics(
            state,
            "tool_registration",
            [target_span],
            "agent tool registration",
        )

    profile = profile_defs.get(stmt.name)
    if profile is None:
        return []
    merged = _merge_adapter_state(state, profile_states.get(stmt.name, _AdapterDiagnosticState()))
    capabilities = _profile_capability_usage(
        profile,
        profile_defs=profile_defs,
        proc_names=proc_names,
    )
    diagnostics: list[lsp.Diagnostic] = []
    for feature, description in (
        ("capabilities", "composable agent capabilities"),
        ("deferred_tools", "on-demand capability loading"),
    ):
        if feature in capabilities:
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    merged,
                    feature,
                    [target_span],
                    description,
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-ADAPTER-CAPABILITY",
                )
            )
    diagnostics.extend(
        _newly_unsupported_requirement_diagnostics(
            state,
            merged,
            span_override=target_span,
        )
    )
    if "tool_registration" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "tool_registration",
                [target_span],
                "agent tool registration from this profile",
            )
        )
    if "mcp" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "mcp",
                [target_span],
                "MCP server directives from this profile",
            )
        )
    if "subagents" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "subagents",
                [target_span],
                "subagent delegation from this profile",
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-SUBAGENTS",
            )
        )
    if "codemode" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "codemode",
                [target_span],
                "CodeMode tool execution from this profile",
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-CODEMODE",
            )
        )
    if "hooks" in capabilities:
        local = merged
        if profile.kind == "capability":
            for member in profile.hooks_members:
                local = _with_hooks_state(local, member)
        diagnostics.extend(_hooks_capability_diagnostics(local, span=target_span))
    return diagnostics


def _profile_capability_usage(
    profile: ProfileDefStmt,
    *,
    profile_defs: dict[str, ProfileDefStmt],
    proc_names: set[str],
    active: frozenset[str] = frozenset(),
) -> set[AdapterCapability]:
    """Return capabilities exercised by a profile and composed profiles."""
    if profile.name in active:
        return set()

    capabilities: set[AdapterCapability] = set()
    if profile.kind == "capability":
        capabilities.add("capabilities")
        if profile.defer_members and profile.defer_members[-1].enabled:
            capabilities.add("deferred_tools")
    if profile.mcp_members:
        capabilities.add("mcp")
    if profile.subagent_members:
        capabilities.add("subagents")
    if any(_activation_state(member.enabled, member.fields) for member in profile.skills_members):
        capabilities.add("tool_registration")
    if any(_activation_state(member.enabled, member.fields) for member in profile.codemode_members):
        capabilities.add("codemode")
    if any(_activation_state(member.enabled, member.fields) for member in profile.hooks_members):
        capabilities.add("hooks")

    nested_active = active | {profile.name}
    for target in (*profile.use_targets, *profile.tool_names):
        if target in proc_names:
            capabilities.add("tool_registration")
            continue
        nested = profile_defs.get(target)
        if nested is not None:
            capabilities.update(
                _profile_capability_usage(
                    nested,
                    profile_defs=profile_defs,
                    proc_names=proc_names,
                    active=nested_active,
                )
            )
        elif target in profile.tool_names:
            capabilities.add("tool_registration")
    return capabilities


def _template_capability_diagnostics(
    state: _AdapterDiagnosticState,
    stmt: TemplateBlock,
    type_defs: dict[str, TypeDef],
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for line in stmt.lines:
        diagnostics.extend(_template_line_capability_diagnostics(state, line, type_defs))
    return diagnostics


def _template_line_capability_diagnostics(
    state: _AdapterDiagnosticState,
    line: TemplateLine,
    type_defs: dict[str, TypeDef],
) -> list[lsp.Diagnostic]:
    output_segments = [seg for seg in line.segments if isinstance(seg, OutputSegment)]
    diagnostics = _unsupported_capability_diagnostics(
        state,
        "structured_output",
        [seg.loc for seg in output_segments],
        "structured template output fields",
    )
    if diagnostics or state.shortname is None:
        return diagnostics

    try:
        supported_formats = adapter_capabilities(state.shortname).supported_json_schema_formats
    except Exception:  # pragma: no cover - keep LSP diagnostics resilient
        supported_formats = None
    if supported_formats is None:
        return diagnostics

    for segment in output_segments:
        unsupported = _unsupported_type_formats(
            segment.type_expr,
            type_defs,
            supported_formats,
        )
        if not unsupported:
            continue
        type_name, schema_format = unsupported[0]
        span = segment.type_expr.loc if segment.type_expr is not None else segment.loc
        if span is None:
            continue
        diagnostics.append(
            lsp.Diagnostic(
                range=_span_to_range(span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-SCHEMA-FORMAT",
                source="kedi",
                message=(
                    f"Adapter '{state.shortname}' does not support Kedi type "
                    f"'{type_name}' in structured output (JSON Schema format "
                    f"'{schema_format}'). Use Annotated[str, \"field description\"] "
                    "when a string with semantic guidance is sufficient."
                ),
            )
        )
    return diagnostics


def _unsupported_type_formats(
    expr: TypeExpr | None,
    type_defs: dict[str, TypeDef],
    supported_formats: frozenset[str],
    seen_types: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    if expr is None or isinstance(expr, (TypeLiteral, TypePython)):
        return []
    if isinstance(expr, TypeRef):
        schema_format = KEDI_RUNTIME_TYPE_JSON_SCHEMA_FORMATS.get(expr.name)
        if schema_format is not None and schema_format not in supported_formats:
            return [(expr.name, schema_format)]
        typedef = type_defs.get(expr.name)
        if typedef is None or expr.name in seen_types:
            return []
        nested_seen = seen_types | {expr.name}
        return [
            unsupported
            for field in typedef.fields
            for unsupported in _unsupported_type_formats(
                field.type_expr,
                type_defs,
                supported_formats,
                nested_seen,
            )
        ]
    if isinstance(expr, TypeApply):
        return [
            unsupported
            for arg in expr.args
            for unsupported in _unsupported_type_formats(
                arg,
                type_defs,
                supported_formats,
                seen_types,
            )
        ]
    if isinstance(expr, TypeUnion):
        return [
            *_unsupported_type_formats(
                expr.left,
                type_defs,
                supported_formats,
                seen_types,
            ),
            *_unsupported_type_formats(
                expr.right,
                type_defs,
                supported_formats,
                seen_types,
            ),
        ]
    return []


def _unsupported_capability_diagnostics(
    state: _AdapterDiagnosticState,
    feature: AdapterCapability,
    spans: Sequence[Span | None],
    description: str,
    *,
    severity: lsp.DiagnosticSeverity | None = None,
    code: str = "K-ADAPTER-CAPABILITY",
) -> list[lsp.Diagnostic]:
    supported = _adapter_supports(state, feature)
    if supported is not False or state.shortname is None:
        return []
    if severity is None:
        severity = (
            lsp.DiagnosticSeverity.Error
            if feature == "structured_output"
            else lsp.DiagnosticSeverity.Warning
        )
    return [
        lsp.Diagnostic(
            range=_span_to_range(span),
            severity=severity,
            code=code,
            source="kedi",
            message=(
                f"Adapter '{state.shortname}' does not support {description}; "
                "this construct will not be handled by that adapter."
            ),
        )
        for span in spans
        if span is not None
    ]


def _explicit_requirement_diagnostics(
    state: _AdapterDiagnosticState,
    *,
    entries: Sequence[tuple[AdapterCapability, Span | None]] | None = None,
    span_override: Span | None = None,
) -> list[lsp.Diagnostic]:
    if state.shortname is None:
        return []
    diagnostics: list[lsp.Diagnostic] = []
    selected_entries = state.requirements if entries is None else entries
    for requirement, source_span in selected_entries:
        if _adapter_capability_status(state, requirement) is not False:
            continue
        span = span_override or source_span
        if span is None:
            continue
        diagnostics.append(
            lsp.Diagnostic(
                range=_span_to_range(span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-REQUIREMENT",
                source="kedi",
                message=(
                    f"Adapter '{state.shortname}' does not satisfy required capability "
                    f"'{requirement}'."
                ),
            )
        )
    return diagnostics


def _newly_unsupported_requirement_diagnostics(
    before: _AdapterDiagnosticState,
    after: _AdapterDiagnosticState,
    *,
    span_override: Span | None = None,
) -> list[lsp.Diagnostic]:
    entries = tuple(
        entry
        for entry in after.requirements
        if entry not in before.requirements
        or _adapter_capability_status(before, entry[0]) is not False
    )
    return _explicit_requirement_diagnostics(
        after,
        entries=entries,
        span_override=span_override,
    )


def _native_approval_capability_diagnostics(
    prog: Program | ProgramWithValidation,
    state: _AdapterDiagnosticState,
    warned_spans: set[tuple[str | None, int, int]],
) -> list[lsp.Diagnostic]:
    """Warn when a selected adapter cannot govern its own builtin or MCP tools.

    Kedi-registered ``ToolSpec`` callables still receive the common approval
    wrapper. The warning is intentionally limited to adapter-native calls,
    whose approval surface is unavailable for the selected adapter.
    """
    loc = state.approval_loc
    if loc is None or state.shortname is None:
        return []
    if _adapter_supports(state, "native_approvals") is not False:
        return []

    span = _approval_value_span(prog, loc, state.approval_plain_value) or loc
    key = (span.source_path, span.start_byte, span.end_byte)
    if key in warned_spans:
        return []
    warned_spans.add(key)
    return [
        lsp.Diagnostic(
            range=_span_to_range(span),
            severity=lsp.DiagnosticSeverity.Warning,
            code="K-ADAPTER-NATIVE-APPROVAL",
            source="kedi",
            message=(
                f"Adapter '{state.shortname}' cannot apply this approval policy to its "
                "builtin or MCP tools. Kedi-registered tools remain protected by Kedi."
            ),
        )
    ]
