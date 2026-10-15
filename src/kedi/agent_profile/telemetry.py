from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

from kedi.telemetry import AttributeValue, capture_enabled, current_span

if TYPE_CHECKING:
    from .capabilities import AppliedCapability
    from .profile import AgentProfile


def capability_trace_id(identity: str) -> str:
    """Correlate ownership without leaking definition-site paths by default."""

    return sha256(identity.encode()).hexdigest()


def capability_attributes(capability: AppliedCapability) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {
        "kedi.capability.id": capability_trace_id(capability.identity),
        "kedi.capability.name": capability.name,
        "kedi.capability.deferred": capability.deferred,
        "kedi.capability.instructions.enabled": capability.instructions_enabled,
        "kedi.capability.instructions.restored": capability.instructions_restored,
    }
    if capture_enabled("source_path"):
        attributes["kedi.capability.identity"] = capability.identity
        if capability.loc is not None:
            attributes["kedi.source.line"] = capability.loc.start_line
            if capability.loc.source_path is not None:
                attributes["kedi.source.path"] = capability.loc.source_path
    return attributes


def merge_profile_with_telemetry(current: AgentProfile, patch: AgentProfile) -> AgentProfile:
    """Observe executed directives; speculative definition merges remain pure."""

    result = current.merge(patch)
    record_profile_transition(current, patch, result)
    return result


def record_profile_transition(
    current: AgentProfile, patch: AgentProfile, result: AgentProfile
) -> None:
    active = current_span()
    if not active.is_recording():
        return
    if patch.system is not None:
        active.add_event(
            "instructions replaced",
            {
                "kedi.instructions.suppressed_capabilities": tuple(
                    capability_trace_id(capability.identity)
                    for capability in current.capabilities
                    if capability.instructions_enabled
                ),
            },
        )
    previous = {capability.identity: capability for capability in current.capabilities}
    effective = {capability.identity: capability for capability in result.capabilities}
    for capability in patch.capabilities:
        old = previous.get(capability.identity)
        active.add_event(
            "capability applied",
            {
                **capability_attributes(effective[capability.identity]),
                "kedi.capability.reused": old is not None,
            },
        )


def record_capability_snapshot(profile: AgentProfile) -> None:
    active = current_span()
    if not active.is_recording():
        return
    for capability in profile.capabilities:
        active.add_event("capability snapshot", capability_attributes(capability))
