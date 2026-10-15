"""Resolve capability ownership for shared child-agent control tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from kedi.agent_profile.capabilities import CapabilityToolPolicy, capability_subagent_policy

from ..tools.specs import ToolSpec

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile
    from kedi.core.runtime import KediRuntime


def with_subagent_ownership(
    tools: Sequence[ToolSpec],
    rt: KediRuntime,
    profile: AgentProfile,
    *,
    activation_profile: AgentProfile | None = None,
) -> tuple[ToolSpec, ...]:
    if not profile.capabilities:
        return tuple(tools)
    definitions = {
        name: profile.subagent_bindings.get(name) or rt._profile_registry.get(name)
        for name in profile.subagents
    }
    policies = {
        name: capability_subagent_policy(
            profile,
            name,
            definition=definitions[name],
            include_deferred=True,
        )
        for name in profile.subagents
    }

    def resolve(arguments: Mapping[str, Any]) -> CapabilityToolPolicy | None:
        name = arguments.get("subagent")
        if isinstance(name, str):
            if activation_profile is not None:
                from kedi.agent_profile.activation import require_subagent_activation

                require_subagent_activation(activation_profile, name)
            return policies.get(name)
        run_id = arguments.get("run_id", arguments.get("task_id"))
        if not isinstance(run_id, str):
            return None
        # The lifecycle implementation owns authorization and invalid-ID errors.
        # Do not expose another agent's run to capability-local hook handlers.
        try:
            run = rt._subagent_coordinator.runs.get(run_id)
        except KeyError:
            return None
        if run.owner_profile_name != profile.name:
            return None
        definition = definitions.get(run.profile_name)
        if definition is None or not rt._subagent_coordinator.runs.matches_profile(
            run_id, definition
        ):
            return None
        if activation_profile is not None:
            from kedi.agent_profile.activation import require_subagent_activation

            require_subagent_activation(activation_profile, run.profile_name)
        return policies.get(run.profile_name)

    return tuple(
        replace(tool, capability_policy_resolver=resolve)
        if (tool.metadata or {}).get("kind") in {"subagent", "subagent_lifecycle"}
        and tool.name != "send_parent_message"
        else tool
        for tool in tools
    )
