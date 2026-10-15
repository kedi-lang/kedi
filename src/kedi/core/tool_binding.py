from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from kedi.errors import KediExecutionError

if TYPE_CHECKING:
    from kedi.agent_adapter.request import InstructionPart
    from kedi.artifacts import ArtifactManager
    from kedi.core.runtime import KediRuntime


def _resolve_tool_specs(self: KediRuntime, profile: Any, adapter: Any) -> tuple[Any, ...]:
    """Resolve a template's tools to fully-bound ToolSpecs (on the main thread, in scope).

    Returns () when the template has no tools. Raises if the adapter cannot register tools or
    a named tool is unknown — surfaced synchronously at the call site, not deferred.
    """
    tool_names = tuple(profile.tools)
    if not tool_names:
        return ()

    from kedi.agent_adapter.tools.conversion import (
        kedi_procedure_tool_spec,
        python_callable_tool_spec,
    )
    from kedi.agent_adapter.tools.execution import with_approval
    from kedi.agent_adapter.tools.specs import SupportsToolRegistration, tool_approval_policy
    from kedi.agent_profile import AgentToolOverrideUnsupported
    from kedi.agent_profile.capabilities import capability_tool_policy
    from kedi.agent_profile.use_resolution import procedure_def_for_tool, python_callable_for_tool

    if not isinstance(adapter, SupportsToolRegistration):
        raise AgentToolOverrideUnsupported(
            f"Active adapter does not support tool registration (requested tools={tool_names!r})"
        )

    specs: list[Any] = []
    bound_tools = getattr(profile, "tool_bindings", {})
    for tool_name in tool_names:
        bound_spec = bound_tools.get(tool_name)
        if bound_spec is not None:
            specs.append(bound_spec)
            continue
        proc = procedure_def_for_tool(self, tool_name)
        if proc is not None:
            specs.append(kedi_procedure_tool_spec(self, proc))
            continue
        tool_fn = python_callable_for_tool(self, tool_name)
        if tool_fn is None:
            raise KediExecutionError(f"Unknown tool procedure: {tool_name!r}")
        specs.append(python_callable_tool_spec(tool_name, tool_fn))
    specs = [
        replace(spec, capability_policy=policy)
        if (policy := capability_tool_policy(profile, name, tool=spec)) is not None
        else spec
        for name, spec in zip(tool_names, specs, strict=True)
    ]
    if self._adapter_uses_native_approvals(adapter):
        return tuple(specs)
    return tuple(
        with_approval(
            spec,
            policy=tool_approval_policy(spec, profile.approval),
            adapter_shortname=getattr(adapter, "shortname", None),
        )
        for spec in specs
    )


def _resolve_agent_tool_specs(
    self: KediRuntime,
    profile: Any,
    adapter: Any,
    *,
    artifact_manager: ArtifactManager | None = None,
) -> tuple[Any, ...]:
    """Resolve explicit Kedi tools plus the opt-in project-local skill tools."""
    from kedi.agent_adapter.telemetry import adapter_shortname
    from kedi.core.capabilities import resolve_deferred_tool_specs, subagent_catalog_profile

    adapter_name = adapter_shortname(adapter)
    artifacts = self.artifacts if artifact_manager is None else artifact_manager
    specs = list(self._resolve_tool_specs(profile, adapter))
    specs.extend(resolve_deferred_tool_specs(self, profile, adapter))

    skills = self._skills_settings(profile)
    if skills is not None and skills.enabled:
        from kedi.agent_adapter.tools.specs import SupportsToolRegistration
        from kedi.agent_profile import AgentToolOverrideUnsupported

        if not isinstance(adapter, SupportsToolRegistration):
            raise AgentToolOverrideUnsupported(
                "Active adapter does not support skill tool registration"
            )
        from kedi.agent_adapter.skills import resolve_skill_store, skill_tool_specs
        from kedi.agent_adapter.tools.execution import with_approval

        skill_specs = skill_tool_specs(
            store=resolve_skill_store(skills, base_dir=self._skill_base_dir()),
            stream_results=profile.artifacts is not None,
        )
        if not self._adapter_uses_native_approvals(adapter):
            skill_specs = tuple(
                with_approval(
                    spec,
                    policy=profile.approval,
                    adapter_shortname=adapter_name,
                )
                for spec in skill_specs
            )
        specs.extend(skill_specs)

    catalog_profile = subagent_catalog_profile(profile)
    if catalog_profile.subagents:
        from kedi.agent_adapter.meta import AdapterCapabilities
        from kedi.agent_adapter.subagents import subagent_tool_specs
        from kedi.agent_adapter.subagents.workflow import dynamic_workflow_tool_spec
        from kedi.agent_profile import AgentSubagentUnsupported

        capabilities = getattr(adapter, "capabilities", None)
        if not isinstance(capabilities, AdapterCapabilities) or not capabilities.supports(
            "subagents"
        ):
            raise AgentSubagentUnsupported(
                "Active adapter does not support Kedi subagent delegation"
            )
        if profile.workflow_mode == "dynamic":
            specs.append(
                dynamic_workflow_tool_spec(self, catalog_profile, activation_profile=profile)
            )
        else:
            specs.extend(
                subagent_tool_specs(
                    self,
                    catalog_profile,
                    supports_background=capabilities.supports("background_subagents"),
                    activation_profile=profile,
                )
            )
    from kedi.agent_adapter.subagents.control import current_live_input
    from kedi.agent_adapter.subagents.messaging import input_tool_specs

    capabilities = getattr(adapter, "capabilities", None)
    live_control = current_live_input()
    if capabilities is not None and capabilities.supports("live_subagent_input"):
        specs.extend(
            input_tool_specs(
                self,
                children=bool(catalog_profile.subagents),
                parent=live_control is not None and live_control.parent is not None,
            )
        )
    from kedi.agent_adapter.subagents.ownership import with_subagent_ownership

    specs = list(with_subagent_ownership(specs, self, catalog_profile, activation_profile=profile))
    artifact_policy = getattr(profile, "artifacts", None)
    artifact_contract_needed = self._artifact_contract_needed(
        profile,
        adapter,
        has_tools=bool(specs),
    )
    if artifact_contract_needed:
        from kedi.agent_adapter.meta import AdapterCapabilities
        from kedi.agent_adapter.tools.execution import with_approval, with_artifact_result
        from kedi.agent_adapter.tools.specs import SupportsToolRegistration
        from kedi.artifacts import ArtifactCapabilityError, artifact_tool_specs

        capabilities = getattr(adapter, "capabilities", None)
        if (
            not isinstance(adapter, SupportsToolRegistration)
            or not isinstance(capabilities, AdapterCapabilities)
            or not capabilities.supports("artifacts")
        ):
            raise ArtifactCapabilityError(
                "Active adapter does not support Kedi artifact references and tools"
            )
        management_specs = artifact_tool_specs(
            artifacts,
            artifact_policy,
            query_artifact=getattr(adapter, "query_artifact", False),
        )
        if not self._adapter_uses_native_approvals(adapter):
            management_specs = tuple(
                with_approval(
                    spec,
                    policy=profile.approval,
                    adapter_shortname=adapter_name,
                )
                for spec in management_specs
            )
        specs.extend(management_specs)
    if specs and artifact_policy is not None:
        from kedi.agent_adapter.tools.execution import with_artifact_result

        specs = [
            with_artifact_result(
                spec,
                manager=artifacts,
                policy=artifact_policy,
            )
            for spec in specs
        ]
    _validate_unique_tool_names(specs)
    from kedi.agent_adapter.tools.execution import with_required_before_output, with_tool_telemetry

    specs = [with_required_before_output(spec) for spec in specs]
    from kedi.agent_adapter.subagents.tool_activity import with_live_tool_tracking

    specs = [
        with_tool_telemetry(with_live_tool_tracking(spec), adapter_shortname=adapter_name)
        for spec in specs
    ]
    return tuple(specs)


def _validate_unique_tool_names(specs: list[Any]) -> None:
    owners: dict[str, str] = {}
    for spec in specs:
        name = spec.name
        proc = getattr(spec, "proc", None)
        source_name = getattr(proc, "name", name)
        previous = owners.get(name)
        if previous is not None:
            raise KediExecutionError(
                f"Tool name collision for exposed name {name!r}: {previous!r} and {source_name!r}"
            )
        owners[name] = source_name


def _adapter_uses_native_approvals(adapter: Any) -> bool:
    from kedi.agent_adapter.meta import AdapterCapabilities

    capabilities = getattr(adapter, "capabilities", None)
    return isinstance(capabilities, AdapterCapabilities) and capabilities.supports(
        "native_approvals"
    )


def _skill_base_dir(self: KediRuntime) -> Path:
    """Resolve the project base beside the executed Kedi source."""
    program_loc = getattr(self._program, "loc", None)
    source_path = getattr(program_loc, "source_path", None)
    if isinstance(source_path, str) and not source_path.startswith("<"):
        return Path(source_path).resolve().parent
    return Path.cwd()


def _skills_settings(profile: Any) -> Any:
    from kedi.agent_profile.skills import skills_settings

    return skills_settings(getattr(profile, "skills", None))


def _agent_instruction_parts(self: KediRuntime, profile: Any, adapter: Any) -> tuple[Any, ...]:
    """Resolve cache-stable instruction fragments with explicit provenance."""
    from kedi.agent_adapter.request import InstructionPart

    parts: list[InstructionPart] = []
    system = getattr(profile, "system", None)
    if system:
        parts.append(
            InstructionPart(
                identity="profile.system",
                source="profile",
                content=system,
            )
        )
    for contribution in getattr(profile, "capability_instructions", ()):
        if contribution.content:
            parts.append(
                InstructionPart(
                    identity=f"capability:{contribution.identity}",
                    source="capability",
                    content=contribution.content,
                )
            )
    skills = self._skills_settings(profile)
    if (
        getattr(profile, "tools", ())
        or getattr(profile, "mcp", ())
        or (skills is not None and skills.enabled)
    ):
        from kedi.agent_adapter.request.prompts import GROUNDING_INSTRUCTION

        parts.append(
            InstructionPart(
                identity="grounding.instructions",
                source="runtime",
                content=f"<ToolCallInstructions>\n{GROUNDING_INSTRUCTION}\n</ToolCallInstructions>",
            )
        )
    if skills is not None and skills.enabled:
        from kedi.agent_adapter.skills import resolve_skill_store, skill_instructions

        parts.append(
            InstructionPart(
                identity="skills.instructions",
                source="skills",
                content=skill_instructions(
                    store=resolve_skill_store(skills, base_dir=self._skill_base_dir())
                ),
            )
        )
    if self._artifact_contract_needed(
        profile,
        adapter,
        has_tools=bool(
            getattr(profile, "tools", ())
            or (skills is not None and skills.enabled)
            or getattr(profile, "subagents", ())
        ),
    ):
        from kedi.artifacts import artifact_instructions

        parts.append(
            InstructionPart(
                identity="artifacts.instructions",
                source="artifacts",
                content=artifact_instructions(
                    query_artifact=profile.artifacts.queries_enabled(
                        default=getattr(adapter, "query_artifact", False),
                    ),
                ),
            )
        )
    return tuple(parts)


def _agent_instructions(self: KediRuntime, profile: Any, adapter: Any) -> str | None:
    """Append enabled runtime contracts without replacing user instructions."""

    rendered = "\n\n".join(part.content for part in self._agent_instruction_parts(profile, adapter))
    return rendered or None


def _artifact_contract_needed(
    self: KediRuntime,
    profile: Any,
    adapter: Any,
    *,
    has_tools: bool,
) -> bool:
    """Choose a cache-stable artifact contract from static call capabilities."""

    policy = getattr(profile, "artifacts", None)
    if policy is None or not policy.resolved().enabled:
        return False
    supports_contract = getattr(adapter, "supports_artifact_contract", None)
    if callable(supports_contract) and not supports_contract(profile):
        if policy.is_explicit:
            from kedi.artifacts import ArtifactCapabilityError

            raise ArtifactCapabilityError(
                "Active model does not support artifact management tools; "
                "disable artifacts for this profile"
            )
        return False
    from kedi.agent_adapter.meta import AdapterCapabilities

    capabilities = getattr(adapter, "capabilities", None)
    if isinstance(capabilities, AdapterCapabilities) and capabilities.supports("artifacts"):
        return True
    return policy.is_explicit or has_tools
