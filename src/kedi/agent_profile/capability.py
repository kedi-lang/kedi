"""Python definitions for the same capability composition used by Kedi source."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.meta import AdapterCapability
from kedi.agent_adapter.requirements import normalize_adapter_requirements
from kedi.agent_adapter.tools.specs import ToolSpec

from .capabilities import capability_patch
from .definitions import ProfileDefinition
from .hooks import HookInput, HookSettings, normalize_hook_settings
from .mcp import McpServerSpec
from .profile import AgentProfile


@dataclass(frozen=True, slots=True, kw_only=True)
class Capability:
    """Reusable tools and guidance, with no host model or conversation overrides.

    Bind an instance into the Kedi environment and apply it with ``> use: name``.
    ``as_profile()`` exposes the same additive patch for low-level integrations.
    A qualified ``identity`` keeps separately published capabilities distinct.
    """

    name: str
    instructions: str | None = None
    description: str | None = None
    identity: str | None = None
    tools: Sequence[ToolSpec | Callable[..., Any]] = ()
    mcp_servers: Sequence[McpServerSpec] = ()
    subagents: Mapping[str, ProfileDefinition] = field(default_factory=dict)
    capabilities: Sequence[Capability] = ()
    approval: ApprovalPolicy | None = None
    hooks: HookSettings | HookInput = None
    requires: Sequence[AdapterCapability] = ()
    defer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("capability name must be an identifier")
        if self.identity is not None and (
            not isinstance(self.identity, str) or not self.identity.strip()
        ):
            raise ValueError("capability identity must be a nonempty string")
        for key in ("instructions", "description"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"capability {key} must be a string or None")
        if type(self.defer) is not bool:
            raise TypeError("capability defer must be a bool")
        if self.approval is not None and not isinstance(self.approval, ApprovalPolicy):
            raise TypeError("capability approval must be an ApprovalPolicy or None")
        hooks = normalize_hook_settings(self.hooks)
        if hooks is not None and (hooks.enabled is False or hooks.user_prompt_submit):
            raise ValueError("capabilities may only add owned tool hooks")
        object.__setattr__(self, "hooks", hooks)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "requires", normalize_adapter_requirements(self.requires))
        object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        for child in self.capabilities:
            if not isinstance(child, Capability):
                raise TypeError("nested capabilities must be Kedi Capability instances")
        for name, definition in self.subagents.items():
            if not isinstance(definition, ProfileDefinition) or definition.kind != "profile":
                raise TypeError(f"subagent {name!r} must be a ProfileDefinition")
            if name != definition.name:
                raise ValueError("subagent binding name must match its profile definition")
        object.__setattr__(self, "subagents", MappingProxyType(deepcopy(dict(self.subagents))))
        if not all(isinstance(server, McpServerSpec) for server in self.mcp_servers):
            raise TypeError("capability mcp_servers must contain McpServerSpec instances")
        object.__setattr__(self, "mcp_servers", deepcopy(self.mcp_servers))
        # Materialize and validate schemas once, retaining callable identity.
        object.__setattr__(self, "tools", self._tool_specs())

    def __deepcopy__(self, memo: dict[int, Any]) -> Capability:
        # Copy mutable definition data, not executable tool/hook objects.
        result = replace(
            self,
            capabilities=tuple(deepcopy(child, memo) for child in self.capabilities),
        )
        memo[id(self)] = result
        return result

    def _tool_specs(self) -> tuple[ToolSpec, ...]:
        from kedi.agent_adapter.tools.conversion import python_callable_tool_spec

        specs = []
        names: set[str] = set()
        for tool in self.tools:
            if isinstance(tool, ToolSpec):
                spec = tool
            elif callable(tool):
                name = getattr(
                    tool, "__kedi_tool_name__", getattr(tool, "__name__", type(tool).__name__)
                )
                spec = python_callable_tool_spec(name, tool)
            else:
                raise TypeError("capability tools must be callables or ToolSpec instances")
            if spec.name in names:
                raise ValueError(f"duplicate capability tool name: {spec.name!r}")
            names.add(spec.name)
            specs.append(
                replace(
                    spec,
                    json_schema=deepcopy(spec.json_schema),
                    return_schema=deepcopy(spec.return_schema),
                    metadata=deepcopy(spec.metadata),
                )
            )
        return tuple(specs)

    def as_profile(self) -> AgentProfile:
        """Lower to a fresh Kedi contribution; no model calls or native activation."""
        tools = self._tool_specs()
        profile = AgentProfile(
            system=self.instructions,
            tools=tuple(tool.name for tool in tools),
            tool_bindings={tool.name: tool for tool in tools},
            mcp_servers=tuple(self.mcp_servers),
            subagents=tuple(self.subagents),
            subagent_bindings=deepcopy(dict(self.subagents)),
            approval=self.approval,
            hooks=normalize_hook_settings(self.hooks),
            requirements=tuple(self.requires),
        )
        for child in self.capabilities:
            profile = profile.merge(child.as_profile())
        return capability_patch(
            ProfileDefinition(
                name=self.name,
                kind="capability",
                identity=self.identity or f"python:{self.name}",
                description=self.description,
            ),
            profile.frozen_copy(),
            deferred=self.defer,
        )


__all__ = ["Capability"]
