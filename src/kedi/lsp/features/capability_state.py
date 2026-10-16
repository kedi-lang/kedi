from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

from kedi.agent_adapter.meta import (
    AdapterCapabilities,
    AdapterCapability,
    AgentAdapterKind,
    HookCapabilityEvent,
)
from kedi.agent_adapter.registry import adapter_capabilities, adapter_factory
from kedi.lang.ast import (
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    ProfileDefStmt,
    ProfileHistoryEntry,
    ProfileHooksEntry,
    ProfileRequiresEntry,
    RequiresDirectiveStmt,
    SettingsField,
    Span,
)


@dataclass(frozen=True)
class _AdapterDiagnosticState:
    shortname: str | None = None
    kind: AgentAdapterKind | None = None
    approval_loc: Span | None = None
    approval_plain_value: str | None = None
    hooks_loc: Span | None = None
    hooks_enabled: bool | None = None
    hook_events: frozenset[HookCapabilityEvent] = frozenset()
    cleared_hook_events: frozenset[HookCapabilityEvent] = frozenset()
    history_loc: Span | None = None
    history_enabled: bool | None = None
    history_processor_loc: Span | None = None
    history_processor_enabled: bool | None = None
    history_condition_loc: Span | None = None
    history_condition_enabled: bool | None = None
    codemode_loc: Span | None = None
    codemode_enabled: bool | None = None
    requirements: tuple[tuple[AdapterCapability, Span | None], ...] = ()


def _profile_adapter_state(profile: ProfileDefStmt) -> _AdapterDiagnosticState:
    members: list[
        tuple[
            int,
            AgentAdapterKind | None,
            str | None,
            Span | None,
        ]
    ] = []
    for member in profile.agent_members:
        if member.loc is not None:
            members.append((member.loc.start_byte, "agent-harness", member.plain_value, None))
    for member in profile.adapter_members:
        if member.loc is not None:
            members.append((member.loc.start_byte, "agent-framework", member.plain_value, None))
    for member in () if profile.kind == "capability" else profile.approval_members:
        if member.loc is not None:
            members.append((member.loc.start_byte, None, member.plain_value, member.loc))

    state = _AdapterDiagnosticState()
    for _, expected_kind, plain_value, approval_loc in sorted(members):
        if approval_loc is not None:
            state = replace(
                state,
                approval_loc=approval_loc,
                approval_plain_value=plain_value,
            )
            continue
        assert expected_kind is not None
        shortname = plain_value
        selected = _adapter_state_from_plain_value(shortname, expected_kind)
        if selected is not None:
            state = _merge_adapter_state(state, selected)
    hooks_members = sorted(
        (
            member
            for member in profile.hooks_members
            if member.loc is not None and profile.kind != "capability"
        ),
        key=lambda member: member.loc.start_byte if member.loc is not None else -1,
    )
    for member in hooks_members:
        state = _with_hooks_state(state, member)
    for member in sorted(
        profile.history_members, key=lambda member: member.loc.start_byte if member.loc else -1
    ):
        state = _with_history_state(state, member)
    codemode_members = [member for member in profile.codemode_members if member.loc is not None]
    if codemode_members:
        latest = max(
            codemode_members,
            key=lambda member: member.loc.start_byte if member.loc is not None else -1,
        )
        state = replace(
            state,
            codemode_loc=latest.loc,
            codemode_enabled=latest.enabled,
        )
    for member in profile.requires_members:
        state = _with_requirements_state(state, member)
    return state


def _adapter_state_from_plain_value(
    shortname: str | None,
    expected_kind: AgentAdapterKind,
) -> _AdapterDiagnosticState | None:
    if shortname is None:
        return None
    try:
        factory = adapter_factory(shortname)
    except Exception:  # pragma: no cover - parser/compiler surfaces unknown adapters elsewhere
        return None
    if factory.kind != expected_kind:
        return None
    return _AdapterDiagnosticState(shortname=shortname, kind=factory.kind)


def _merge_adapter_state(
    outer: _AdapterDiagnosticState,
    inner: _AdapterDiagnosticState,
) -> _AdapterDiagnosticState:
    selected = inner if inner.shortname is not None else outer
    approval_owner = inner if inner.approval_loc is not None else outer
    hooks_owner = inner if inner.hooks_enabled is not None else outer
    hook_events = outer.hook_events
    cleared_hook_events = outer.cleared_hook_events
    if inner.hooks_enabled is not None:
        hook_events = (hook_events - inner.cleared_hook_events) | inner.hook_events
        cleared_hook_events = (cleared_hook_events | inner.cleared_hook_events) - inner.hook_events
    history_owner = inner if inner.history_enabled is not None else outer
    processor_owner = inner if inner.history_processor_enabled is not None else outer
    condition_owner = inner if inner.history_condition_enabled is not None else outer
    codemode_owner = inner if inner.codemode_enabled is not None else outer
    return _AdapterDiagnosticState(
        shortname=selected.shortname,
        kind=selected.kind,
        approval_loc=approval_owner.approval_loc,
        approval_plain_value=approval_owner.approval_plain_value,
        hooks_loc=hooks_owner.hooks_loc,
        hooks_enabled=hooks_owner.hooks_enabled,
        hook_events=hook_events,
        cleared_hook_events=cleared_hook_events,
        history_loc=history_owner.history_loc,
        history_enabled=history_owner.history_enabled,
        history_processor_enabled=processor_owner.history_processor_enabled,
        history_processor_loc=processor_owner.history_processor_loc,
        history_condition_enabled=condition_owner.history_condition_enabled,
        history_condition_loc=condition_owner.history_condition_loc,
        codemode_loc=codemode_owner.codemode_loc,
        codemode_enabled=codemode_owner.codemode_enabled,
        requirements=_merge_requirement_entries(outer.requirements, inner.requirements),
    )


def _with_history_state(
    state: _AdapterDiagnosticState,
    directive: HistoryDirectiveStmt | ProfileHistoryEntry,
) -> _AdapterDiagnosticState:
    state = replace(state, history_loc=directive.loc, history_enabled=directive.enabled)
    processor = next((field for field in directive.fields if field.name == "processor"), None)
    if processor is not None:
        state = replace(
            state,
            history_processor_loc=processor.loc,
            history_processor_enabled=(
                processor.backtick_code or processor.plain_value or ""
            ).strip()
            != "None",
            history_condition_loc=None,
            history_condition_enabled=False,
        )
    condition = next(
        (field for field in directive.fields if field.name == "processor_condition"), None
    )
    if condition is not None:
        state = replace(
            state,
            history_condition_loc=condition.loc,
            history_condition_enabled=(condition.backtick_code or "").strip() != "None",
        )
    return state


def _merge_requirement_entries(
    left: tuple[tuple[AdapterCapability, Span | None], ...],
    right: tuple[tuple[AdapterCapability, Span | None], ...],
) -> tuple[tuple[AdapterCapability, Span | None], ...]:
    merged = list(left)
    for entry in right:
        if entry not in merged:
            merged.append(entry)
    return tuple(merged)


def _with_requirements_state(
    state: _AdapterDiagnosticState,
    directive: RequiresDirectiveStmt | ProfileRequiresEntry,
) -> _AdapterDiagnosticState:
    entries: tuple[tuple[AdapterCapability, Span | None], ...] = tuple(
        (cast(AdapterCapability, requirement), directive.requirement_locs.get(requirement))
        for requirement in directive.requirements
    )
    return replace(
        state,
        requirements=_merge_requirement_entries(state.requirements, entries),
    )


def _adapter_supports(
    state: _AdapterDiagnosticState,
    feature: AdapterCapability,
) -> bool | None:
    if state.shortname is None:
        return None
    try:
        capabilities = adapter_capabilities(state.shortname)
    except Exception:  # pragma: no cover - keep LSP diagnostics resilient
        capabilities = AdapterCapabilities()
    return capabilities.supports(feature)


def _adapter_capability_status(
    state: _AdapterDiagnosticState,
    feature: AdapterCapability,
) -> bool | None:
    if state.shortname is None:
        return None
    try:
        capabilities = adapter_capabilities(state.shortname)
    except Exception:  # pragma: no cover - keep LSP diagnostics resilient
        return None
    return capabilities.capability_status(feature)


def _hook_event_changes(
    fields: Sequence[SettingsField],
) -> tuple[frozenset[HookCapabilityEvent], frozenset[HookCapabilityEvent]]:
    enabled: set[HookCapabilityEvent] = set()
    cleared: set[HookCapabilityEvent] = set()
    for field in fields:
        if field.name not in {
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "post_tool_use_failure",
            "post_tool_result_failure",
        }:
            continue
        event = cast(HookCapabilityEvent, field.name)
        if (field.backtick_code or "").strip() == "[]":
            enabled.discard(event)
            cleared.add(event)
        else:
            enabled.add(event)
            cleared.discard(event)
    return frozenset(enabled), frozenset(cleared)


def _with_hooks_state(
    state: _AdapterDiagnosticState,
    directive: HooksDirectiveStmt | ProfileHooksEntry,
) -> _AdapterDiagnosticState:
    enabled, cleared = _hook_event_changes(directive.fields)
    return replace(
        state,
        hooks_loc=directive.loc,
        hooks_enabled=directive.enabled,
        hook_events=(state.hook_events - cleared) | enabled,
        cleared_hook_events=(state.cleared_hook_events | cleared) - enabled,
    )
