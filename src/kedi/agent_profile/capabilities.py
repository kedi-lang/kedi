from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from inspect import unwrap
from typing import TYPE_CHECKING

from ..agent_adapter.approvals.policy import ApprovalPolicy
from ..errors import KediExecutionError, KediTraceFrame
from .hooks import HookSettings
from .instructions import CapabilityInstruction

if TYPE_CHECKING:
    from ..agent_adapter.tools.specs import ToolSpec
    from ..lang.ast import Span
    from .definitions import ProfileDefinition
    from .mcp import McpServerSpec
    from .profile import AgentProfile


CAPABILITY_DIRECTIVES = frozenset(
    {"instructions", "use", "mcp", "subagent", "approval", "hooks", "requires", "defer"}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AppliedCapability:
    """Resolved definition-site contributions, separate from their activation."""

    identity: str
    name: str
    description: str | None
    profile: AgentProfile
    loc: Span | None = field(default=None, repr=False, compare=False)
    deferred: bool = False
    instructions_enabled: bool = True
    instructions_restored: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", self.profile.frozen_copy())


def _capability_conflict(
    message: str, capabilities: Iterable[AppliedCapability]
) -> KediExecutionError:
    frames: list[KediTraceFrame] = []
    for capability in capabilities:
        frame = KediTraceFrame(
            kind="capability", label=f"capability {capability.name}", span=capability.loc
        )
        if frame not in frames:
            frames.append(frame)
    return KediExecutionError(message, frames=tuple(frames))


def _capability_origins(
    profile: AgentProfile, identities: tuple[str, ...]
) -> Iterator[AppliedCapability]:
    for capability in profile.capabilities:
        if capability.identity in identities:
            yield capability
        yield from _capability_origins(capability.profile, identities)


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityToolPolicy:
    """Ownership retained independently of the host profile's default policy."""

    owners: tuple[str, ...]
    approval: ApprovalPolicy | None = None
    hook_contributions: tuple[tuple[str, HookSettings], ...] = ()

    @property
    def hooks(self) -> HookSettings:
        result = HookSettings()
        for _owner, hooks in self.hook_contributions:
            result = result.compose_enforced(hooks)
        return result


def capability_tool_policy(
    profile: AgentProfile,
    tool_name: str,
    *,
    include_deferred: bool = False,
    tool: ToolSpec | None = None,
) -> CapabilityToolPolicy | None:
    """Resolve tool ownership from the applied graph, not the last profile used."""

    if tool is not None:
        _validate_resolved_tool(profile, tool_name, tool, include_deferred=include_deferred)
    return _resource_policy(
        profile,
        owns=lambda owned: tool_name in owned.tools,
        label=f"tool {tool_name!r}",
        include_deferred=include_deferred,
    )


def _validate_resolved_tool(
    profile: AgentProfile, name: str, tool: ToolSpec, *, include_deferred: bool
) -> None:
    for capability in profile.capabilities:
        if capability.deferred and not include_deferred:
            continue
        expected = capability.profile.tool_bindings.get(name)
        if expected is not None and not _same_tool_binding(expected, tool):
            raise _capability_conflict(
                f"Conflicting capability tool binding {name!r}: "
                f"{capability.identity!r} owns a different callable; use distinct tool names",
                (capability,),
            )
        _validate_resolved_tool(capability.profile, name, tool, include_deferred=include_deferred)


def _same_tool_binding(left: ToolSpec, right: ToolSpec) -> bool:
    if left is right:
        return True
    return (
        left.proc is right.proc
        and unwrap(left.fn) is unwrap(right.fn)
        and all(
            getattr(left, field) == getattr(right, field)
            for field in (
                "name",
                "description",
                "json_schema",
                "return_schema",
                "metadata",
                "risk",
                "approval",
                "retry_policy",
                "artifact_result",
                "sequential",
            )
        )
    )


def mcp_server_identity(server: McpServerSpec) -> str:
    """Opaque stable identity; server credentials never enter tool metadata."""

    encoded = json.dumps(asdict(server), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def capability_mcp_policy(
    profile: AgentProfile, server_identity: str
) -> CapabilityToolPolicy | None:
    return _resource_policy(
        profile,
        owns=lambda owned: any(
            mcp_server_identity(server) == server_identity for server in owned.mcp_servers
        ),
        label=f"MCP server {server_identity!r}",
        include_deferred=True,
    )


def capability_subagent_policy(
    profile: AgentProfile,
    subagent: str,
    *,
    definition: ProfileDefinition | None = None,
    include_deferred: bool = False,
) -> CapabilityToolPolicy | None:
    if definition is not None:
        _validate_subagent_binding(profile, subagent, definition, include_deferred=include_deferred)
    return _resource_policy(
        profile,
        owns=lambda owned: subagent in owned.subagents,
        label=f"subagent {subagent!r}",
        include_deferred=include_deferred,
    )


def _validate_subagent_binding(
    profile: AgentProfile, name: str, definition: ProfileDefinition, *, include_deferred: bool
) -> None:
    for capability in profile.capabilities:
        if capability.deferred and not include_deferred:
            continue
        expected = capability.profile.subagent_bindings.get(name)
        if expected is not None and expected != definition:
            raise _capability_conflict(
                f"Conflicting capability subagent binding {name!r}: "
                f"{capability.identity!r} owns a different profile; use distinct subagent names",
                (capability,),
            )
        _validate_subagent_binding(
            capability.profile, name, definition, include_deferred=include_deferred
        )


def _resource_policy(
    profile: AgentProfile,
    *,
    owns: Callable[[AgentProfile], bool],
    label: str,
    include_deferred: bool,
) -> CapabilityToolPolicy | None:
    candidates: list[CapabilityToolPolicy] = []
    nested_ids = {
        child.identity
        for capability in profile.capabilities
        for child in capability.profile.capabilities
    }
    for capability in profile.capabilities:
        if capability.identity in nested_ids or (capability.deferred and not include_deferred):
            continue
        nested = _resource_policy(
            capability.profile, owns=owns, label=label, include_deferred=include_deferred
        )
        if not owns(capability.profile) and nested is None:
            continue
        hooks = (
            ()
            if capability.profile.hooks is None
            else ((capability.identity, capability.profile.hooks),)
        )
        candidates.append(
            CapabilityToolPolicy(
                owners=(capability.identity,)
                if nested is None
                else (
                    capability.identity,
                    *nested.owners,
                ),
                approval=(
                    nested.approval
                    if nested is not None and nested.approval is not None
                    else capability.profile.approval
                ),
                hook_contributions=hooks + (() if nested is None else nested.hook_contributions),
            )
        )
    if not candidates:
        return None
    first = candidates[0]
    for candidate in candidates[1:]:
        if candidate.approval != first.approval:
            raise _capability_conflict(
                f"Conflicting capability approval for {label}: "
                f"{first.owners!r} and {candidate.owners!r}",
                _capability_origins(profile, (*first.owners, *candidate.owners)),
            )
    contributions: dict[str, HookSettings] = {}
    for candidate in candidates:
        for identity, hooks in candidate.hook_contributions:
            previous = contributions.get(identity)
            if previous is not None and previous != hooks:
                raise _capability_conflict(
                    f"Conflicting capability hooks for {label} from {identity!r}",
                    _capability_origins(profile, (identity,)),
                )
            contributions[identity] = hooks
    return CapabilityToolPolicy(
        owners=tuple(
            dict.fromkeys(owner for candidate in candidates for owner in candidate.owners)
        ),
        approval=first.approval,
        hook_contributions=tuple(contributions.items()),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredCapabilityBoundary:
    capability: AppliedCapability
    paths: tuple[tuple[str, ...], ...]
    instruction_paths: tuple[tuple[str, ...], ...]

    def discoverable(self, active: frozenset[str]) -> bool:
        return any(all(identity in active for identity in path) for path in self.paths)


def deferred_boundaries(profile: AgentProfile) -> tuple[DeferredCapabilityBoundary, ...]:
    """Retain each activation boundary and every independently reachable path."""

    definitions: dict[str, AppliedCapability] = {}
    paths: dict[str, list[tuple[str, ...]]] = {}
    instruction_paths: dict[str, list[tuple[str, ...]]] = {}
    overrides = {capability.identity: capability for capability in profile.capabilities}

    def visit(
        capability: AppliedCapability,
        ancestors: tuple[str, ...],
        deferred_path: tuple[str, ...],
        enabled: bool,
        restored: bool,
    ) -> None:
        if capability.identity in ancestors:
            raise KediExecutionError(f"Cyclic capability graph involving {capability.name!r}")
        if capability.identity in overrides:
            capability = overrides[capability.identity]
        else:
            instructions_enabled = enabled and capability.instructions_enabled
            instructions_restored = enabled and (restored or capability.instructions_restored)
            if (
                instructions_enabled != capability.instructions_enabled
                or instructions_restored != capability.instructions_restored
            ):
                capability = replace(
                    capability,
                    instructions_enabled=instructions_enabled,
                    instructions_restored=instructions_restored,
                )
        if capability.deferred:
            previous = definitions.get(capability.identity)
            definitions[capability.identity] = (
                capability
                if previous is None
                else replace(
                    capability,
                    instructions_enabled=(
                        previous.instructions_enabled or capability.instructions_enabled
                    ),
                    instructions_restored=(
                        previous.instructions_restored or capability.instructions_restored
                    ),
                )
            )
            routes = paths.setdefault(capability.identity, [])
            if deferred_path not in routes:
                routes.append(deferred_path)
            guidance_routes = instruction_paths.setdefault(capability.identity, [])
            if capability.instructions_enabled and deferred_path not in guidance_routes:
                guidance_routes.append(deferred_path)
            deferred_path = (*deferred_path, capability.identity)
        for child in capability.profile.capabilities:
            visit(
                child,
                (*ancestors, capability.identity),
                deferred_path,
                capability.instructions_enabled,
                capability.instructions_restored,
            )

    for capability in profile.capabilities:
        visit(capability, (), (), True, False)
    return tuple(
        DeferredCapabilityBoundary(
            capability=capability,
            paths=tuple(paths[identity]),
            instruction_paths=tuple(instruction_paths[identity]),
        )
        for identity, capability in definitions.items()
    )


def deferred_capabilities(profile: AgentProfile) -> tuple[AppliedCapability, ...]:
    return tuple(boundary.capability for boundary in deferred_boundaries(profile))


def capability_patch(
    definition: ProfileDefinition,
    profile: AgentProfile,
    *,
    deferred: bool,
) -> AgentProfile:
    from .profile import AgentProfile

    identity = definition.identity or definition.name
    capability = AppliedCapability(
        identity=identity,
        name=definition.name,
        description=definition.description,
        loc=definition.loc,
        profile=profile,
        deferred=deferred,
    )
    requirement = "deferred_tools" if deferred else "capabilities"
    requirements = tuple(dict.fromkeys((*profile.requirements, "capabilities", requirement)))
    if deferred:
        return AgentProfile(capabilities=(capability,), requirements=requirements)
    instructions = profile.capability_instructions
    if profile.system is not None:
        instructions = (
            CapabilityInstruction(identity=identity, content=profile.system),
            *instructions,
        )
    return replace(
        profile,
        name=None,
        system=None,
        approval=None,
        hooks=None,
        capability_instructions=instructions,
        capabilities=(*profile.capabilities, capability),
        requirements=requirements,
    )


def merge_capabilities(
    current: tuple[AppliedCapability, ...],
    incoming: tuple[AppliedCapability, ...],
) -> tuple[AppliedCapability, ...]:
    by_identity = {capability.identity: capability for capability in current}
    for capability in incoming:
        previous = by_identity.get(capability.identity)
        if previous is not None:
            for name in (
                previous.profile.tool_bindings.keys() & capability.profile.tool_bindings.keys()
            ):
                if not _same_tool_binding(
                    previous.profile.tool_bindings[name], capability.profile.tool_bindings[name]
                ):
                    raise _capability_conflict(
                        f"Conflicting capability tool binding {name!r}: "
                        f"{capability.identity!r} was already bound to a different callable",
                        (previous, capability),
                    )
            for name in (
                previous.profile.subagent_bindings.keys()
                & capability.profile.subagent_bindings.keys()
            ):
                if (
                    previous.profile.subagent_bindings[name]
                    != capability.profile.subagent_bindings[name]
                ):
                    raise _capability_conflict(
                        f"Conflicting capability subagent binding {name!r}: "
                        f"{capability.identity!r} was already bound to a different profile",
                        (previous, capability),
                    )
        if previous is not None and previous.deferred != capability.deferred:
            raise _capability_conflict(
                f"Capability {capability.name!r} ({capability.identity}) was already used with "
                "a different defer mode in this scope; use a separate scope for a different mode",
                (previous, capability),
            )
        if (
            previous is not None
            and capability.instructions_enabled
            and (not previous.instructions_enabled or previous.instructions_restored)
        ):
            capability = replace(capability, instructions_restored=True)
        by_identity[capability.identity] = capability
    return tuple(by_identity.values())


def validate_capability_bindings(current: AgentProfile, incoming: AgentProfile) -> None:
    """Do not silently replace a private dependency while composing capabilities."""

    for name in current.tool_bindings.keys() & incoming.tool_bindings.keys():
        if not any(
            name in capability.profile.tools
            for capability in (*current.capabilities, *incoming.capabilities)
        ):
            continue
        left = current.tool_bindings[name]
        right = incoming.tool_bindings[name]
        if _same_tool_binding(left, right):
            continue
        raise _capability_conflict(
            f"Conflicting capability tool binding {name!r}: "
            f"{_binding_owners(current, name)!r} and {_binding_owners(incoming, name)!r}; "
            "use distinct exported tool names",
            (
                *_capability_origins(current, _binding_owners(current, name)),
                *_capability_origins(incoming, _binding_owners(incoming, name)),
            ),
        )
    for name in current.subagent_bindings.keys() & incoming.subagent_bindings.keys():
        if not any(
            name in capability.profile.subagents
            for capability in (*current.capabilities, *incoming.capabilities)
        ):
            continue
        if current.subagent_bindings[name] != incoming.subagent_bindings[name]:
            raise _capability_conflict(
                f"Conflicting capability subagent binding {name!r}: "
                f"{_binding_owners(current, name)!r} and {_binding_owners(incoming, name)!r}; "
                "use distinct subagent names",
                (
                    *_capability_origins(current, _binding_owners(current, name)),
                    *_capability_origins(incoming, _binding_owners(incoming, name)),
                ),
            )


def _binding_owners(profile: AgentProfile, name: str) -> tuple[str, ...]:
    owners = tuple(
        capability.identity
        for capability in profile.capabilities
        if name in capability.profile.tools or name in capability.profile.subagents
    )
    return owners or (profile.name or "agent scope",)
