from __future__ import annotations

from hashlib import sha256
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CapabilityOrdering,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile.capabilities import AppliedCapability, deferred_boundaries
from kedi.agent_profile.activation import available_subagents, capability_activation_scope
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.telemetry import capability_attributes
from kedi.errors import KediExecutionError
from kedi.telemetry import current_span, span

from .tools import _approval_required_toolset, pydantic_toolset_from_specs
from .compat import NativeDeferredCapabilityLoader, native_deferred_catalog

if TYPE_CHECKING:
    from .adapter import PydanticAdapter


def native_capability_id(capability: AppliedCapability) -> str:
    """Stable loader identity without disclosing definition-site filesystem paths."""

    suffix = sha256(capability.identity.encode()).hexdigest()[:12]
    return f"kedi_{capability.name}_{suffix}"


def deferred_run_capabilities(
    adapter: PydanticAdapter[Any, Any], tools: Sequence[ToolSpec]
) -> list[AbstractCapability[Any]]:
    result: list[AbstractCapability[Any]] = []
    boundaries = deferred_boundaries(adapter.active_profile)
    definitions = tuple(boundary.capability for boundary in boundaries)
    identities = {capability.identity for capability in definitions}
    for tool in tools:
        if tool.deferred_capability is not None and tool.deferred_capability not in identities:
            raise KediExecutionError(
                f"Tool {tool.name!r} refers to an unregistered deferred capability "
                f"{tool.deferred_capability!r}"
            )
    if not definitions:
        return result
    result.append(_CapabilityLoadingTelemetry(definitions))
    graph = _DeferredCapabilityGraph(adapter.active_profile, tools)
    result.append(graph)
    result.append(_DeferredCapabilityTools(graph))
    if any(any(path for path in boundary.paths) for boundary in boundaries):
        result.append(_NestedCapabilityLoader(graph))
    for boundary in boundaries:
        capability = boundary.capability
        identity = native_capability_id(capability)
        scoped_guidance = (
            capability.instructions_restored or boundary.instruction_paths != boundary.paths
        )
        owned_tools = [tool for tool in tools if tool.deferred_capability == capability.identity]
        instructions = (
            [
                *([capability.profile.system] if capability.profile.system is not None else []),
                *(part.content for part in capability.profile.capability_instructions),
            ]
            if capability.instructions_enabled
            else []
        )
        toolsets = (
            [
                _approval_required_toolset(
                    pydantic_toolset_from_specs(owned_tools), adapter_shortname=adapter.shortname
                )
            ]
            if owned_tools
            else []
        )
        toolsets.extend(
            _approval_required_toolset(toolset, adapter_shortname=adapter.shortname)
            for toolset in adapter._mcp_toolsets_for_profile(capability.profile)
        )
        result.append(
            Capability(
                id=identity,
                description=capability.description or capability.name,
                instructions=None if scoped_guidance else instructions,
                toolsets=toolsets,
                defer_loading=True,
            )
        )
        if scoped_guidance and instructions:
            result.append(_scoped_guidance(identity, "\n\n".join(instructions), graph))
    return result


class _DeferredCapabilityGraph(AbstractCapability[Any]):
    """Apply Kedi reachability to native catalog, tools, and shared child controls."""

    def __init__(self, profile: AgentProfile, tools: Sequence[ToolSpec]) -> None:
        self.profile = profile
        self.boundaries = {
            native_capability_id(boundary.capability): boundary
            for boundary in deferred_boundaries(profile)
        }
        self.tools = {tool.name: tool for tool in tools}

    def active(self, ctx: RunContext[Any]) -> frozenset[str]:
        if not self.boundaries:
            return frozenset()
        native_active = ctx.active_capability_ids
        loaded = frozenset(
            boundary.capability.identity
            for identity, boundary in self.boundaries.items()
            if identity in native_active
        )
        return frozenset(
            boundary.capability.identity
            for boundary in self.boundaries.values()
            if boundary.capability.identity in loaded and boundary.discoverable(loaded)
        )

    def get_ordering(self) -> CapabilityOrdering:
        from .hooks import _KediHookCapability

        return CapabilityOrdering(wraps=(_KediHookCapability,))

    async def project_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        active = self.active(ctx)
        children = available_subagents(self.profile, active)
        result = []
        for definition in tool_defs:
            boundary = self.boundaries.get(definition.capability_id or "")
            if boundary is not None and not boundary.discoverable(active):
                continue
            metadata = definition.metadata or {}
            kind = metadata.get("kind")
            if kind in {"subagent", "subagent_lifecycle", "dynamic_workflow"}:
                if definition.name != "send_parent_message" and not children:
                    continue
                if definition.name == "delegate_task":
                    schema = dict(definition.parameters_json_schema)
                    properties = dict(schema["properties"])
                    properties["subagent"] = {**properties["subagent"], "enum": list(children)}
                    schema["properties"] = properties
                    catalog = metadata.get("kedi_subagent_catalog", {})
                    entries = [
                        f"- {name}: {catalog[name]}" if catalog.get(name) else f"- {name}"
                        for name in children
                    ]
                    definition = replace(
                        definition,
                        parameters_json_schema=schema,
                        description=metadata.get("kedi_subagent_preamble", "Delegate a task.")
                        + "\n\nAvailable subagents:\n"
                        + "\n".join(entries),
                    )
                elif kind == "dynamic_workflow":
                    from kedi.agent_adapter.subagents.workflow import (
                        _sandbox_names,
                        _workflow_description,
                    )

                    catalog = metadata.get("kedi_workflow_catalog", {})
                    visible = {name: catalog.get(name) for name in _sandbox_names(children)}
                    definition = replace(
                        definition,
                        description=_workflow_description(
                            visible, remaining=metadata.get("kedi_workflow_remaining", 0)
                        ),
                    )
                definition = replace(
                    definition,
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if key
                        not in {
                            "kedi_subagent_preamble",
                            "kedi_subagent_catalog",
                            "kedi_workflow_catalog",
                            "kedi_workflow_remaining",
                        }
                    },
                )
            result.append(definition)
        return result

    async def wrap_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        with capability_activation_scope(self.active(ctx)):
            try:
                arguments = call.args_as_dict(raise_if_invalid=True)
            except (ValueError, AssertionError):
                return await handler(args)
            self._check_call(ctx, tool_def, arguments)
            return await handler(args)

    def _check_call(
        self, ctx: RunContext[Any], tool_def: ToolDefinition, arguments: dict[str, Any]
    ) -> None:
        active = self.active(ctx)
        if tool_def.tool_kind == "capability-load":
            identity = arguments.get("id")
            boundary = self.boundaries.get(identity) if isinstance(identity, str) else None
            if boundary is not None and not boundary.discoverable(active):
                current_span().add_event(
                    "capability load rejected",
                    {
                        **capability_attributes(boundary.capability),
                        "kedi.capability.rejection": "parent_not_active",
                    },
                )
                raise ModelRetry("Load the parent capability before loading this capability.")
        elif (boundary := self.boundaries.get(tool_def.capability_id or "")) is not None:
            if boundary.capability.identity not in active:
                raise ModelRetry("Load this tool's capability and its parent capabilities first.")
        tool = self.tools.get(tool_def.name)
        if tool is not None and tool.capability_policy_resolver is not None:
            from kedi.agent_adapter.tools.errors import CapabilityActivationError

            try:
                tool.capability_policy_resolver(arguments)
            except CapabilityActivationError as exc:
                raise ModelRetry(str(exc)) from exc

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        with capability_activation_scope(self.active(ctx)):
            self._check_call(ctx, tool_def, args)
            return await handler(args)


class _NestedCapabilityLoader(NativeDeferredCapabilityLoader):
    def __init__(self, graph: _DeferredCapabilityGraph) -> None:
        self.graph = graph

    def get_instructions(self):
        async def instructions(ctx: RunContext[Any]) -> str:
            active = self.graph.active(ctx)
            visible = {
                identity: capability
                for identity, capability in ctx.capabilities.items()
                if identity not in self.graph.boundaries
                or self.graph.boundaries[identity].discoverable(active)
            }
            return await native_deferred_catalog(replace(ctx, capabilities=visible))

        return instructions


class _DeferredCapabilityTools(AbstractCapability[Any]):
    def __init__(self, graph: _DeferredCapabilityGraph) -> None:
        self.graph = graph

    def get_ordering(self) -> CapabilityOrdering:
        from kedi.agent_adapter.codemode.pydantic import PydanticCodeModeCapability

        return CapabilityOrdering(wrapped_by=(PydanticCodeModeCapability,))

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
        return _CapabilityGraphToolset(wrapped=toolset, graph=self.graph)


@dataclass
class _CapabilityGraphToolset(WrapperToolset[Any]):
    graph: _DeferredCapabilityGraph

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        tools = await self.wrapped.get_tools(ctx)
        definitions = await self.graph.project_tools(
            ctx, [tool.tool_def for tool in tools.values()]
        )
        return {
            definition.name: replace(tools[definition.name], tool_def=definition)
            for definition in definitions
        }


class _CapabilityLoadingTelemetry(AbstractCapability[Any]):
    """Observe native activation without driving or replacing its state machine."""

    def __init__(self, definitions: tuple[AppliedCapability, ...]) -> None:
        self.definitions = definitions
        self.by_id = {native_capability_id(capability): capability for capability in definitions}
        self.seen_active: set[str] = set()
        self.prepared = False

    async def for_run(self, ctx: RunContext[Any]) -> _CapabilityLoadingTelemetry:
        return type(self)(self.definitions)

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        active = ctx.active_capability_ids & self.by_id.keys()
        for identity in sorted(active - self.seen_active):
            current_span().add_event(
                "capability activated",
                {
                    **capability_attributes(self.by_id[identity]),
                    "kedi.capability.native_id": identity,
                    "kedi.capability.resumed": not self.prepared,
                },
            )
        self.seen_active = set(active)
        self.prepared = True
        return tool_defs

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        if tool_def.tool_kind != "capability-load":
            return await handler(args)
        identity = args.get("id")
        definition = self.by_id.get(identity) if isinstance(identity, str) else None
        attributes = {} if definition is None else capability_attributes(definition)
        attributes["kedi.capability.known"] = definition is not None
        attributes["kedi.capability.already_active"] = identity in ctx.active_capability_ids
        with span("agent", "load capability", "load_capability", attributes=attributes) as active:
            active.add_event("capability load requested")
            try:
                result = await handler(args)
            except BaseException as exc:
                active.set_attribute("kedi.capability.load.outcome", "error")
                active.set_attribute("error.type", type(exc).__name__)
                raise
            active.set_attribute("kedi.capability.load.outcome", "delivered")
            return result


def _scoped_guidance(
    identity: str, content: str, graph: _DeferredCapabilityGraph
) -> Capability[Any]:
    # Native loaders do not reload an already-active capability. An explicit Kedi
    # reuse restores guidance only through an activated path that still enables it.
    def instructions(ctx: RunContext[Any]) -> str | None:
        boundary = graph.boundaries[identity]
        active = graph.active(ctx)
        enabled = boundary.capability.identity in active and any(
            all(parent in active for parent in path) for path in boundary.instruction_paths
        )
        return content if enabled else None

    return Capability(id=f"{identity}_guidance", instructions=instructions)
