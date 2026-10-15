from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile.capabilities import capability_tool_policy, deferred_capabilities

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile
    from kedi.core.runtime import KediRuntime


def resolve_deferred_tool_specs(
    rt: KediRuntime, profile: AgentProfile, adapter: Any
) -> tuple[ToolSpec, ...]:
    """Bind hidden tools in lexical scope without making them eagerly executable."""

    specs: list[ToolSpec] = []
    for capability in deferred_capabilities(profile):
        bound = rt._resolve_tool_specs(capability.profile, adapter)
        for name, spec in zip(capability.profile.tools, bound, strict=True):
            specs.append(
                replace(
                    spec,
                    deferred_capability=capability.identity,
                    capability_policy=capability_tool_policy(
                        profile, name, include_deferred=True, tool=spec
                    ),
                )
            )
    return tuple(specs)


def subagent_catalog_profile(profile: AgentProfile) -> AgentProfile:
    """Capture all child bindings; model-facing availability is decided per step."""
    deferred = deferred_capabilities(profile)
    if not deferred:
        return profile
    names = dict.fromkeys(profile.subagents)
    bindings = dict(profile.subagent_bindings)
    for capability in deferred:
        names.update(dict.fromkeys(capability.profile.subagents))
        for name, definition in capability.profile.subagent_bindings.items():
            if name in bindings and bindings[name] != definition:
                from kedi.errors import KediExecutionError

                raise KediExecutionError(f"Conflicting deferred subagent binding {name!r}")
            bindings[name] = definition
    return replace(profile, subagents=tuple(names), subagent_bindings=bindings)
