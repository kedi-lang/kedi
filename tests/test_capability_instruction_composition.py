from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from kedi.agent_profile.capabilities import AppliedCapability, deferred_capabilities
from kedi.agent_profile.definitions import ProfileDefinition, ProfileSystemMember
from kedi.agent_profile.instructions import CapabilityInstruction
from kedi.agent_profile.mcp import McpServerSpec
from kedi.agent_profile.profile import AgentProfile
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import TemplateLine, TextSegment
from tests.mock_adapter import MockAdapter


def contribution(name: str, content: str) -> AgentProfile:
    return AgentProfile(
        capability_instructions=(CapabilityInstruction(identity=name, content=content),)
    )


def test_capabilities_append_in_stable_order_without_replacing_base():
    base = AgentProfile(system="Base.")
    first = base.merge(contribution("a", "A."))
    result = first.merge(contribution("b", "B.")).merge(contribution("a", "A."))
    assert result.system == "Base."
    assert [(part.identity, part.content) for part in result.capability_instructions] == [
        ("a", "A."),
        ("b", "B."),
    ]
    assert base.capability_instructions == ()
    assert len(first.capability_instructions) == 1


@pytest.mark.parametrize("replacement", ["Base.", "New base.", ""])
def test_explicit_instruction_reset_clears_contributions_and_preserves_tools(replacement):
    original = AgentProfile(system="Base.", tools=("read_document",), history=True).merge(
        contribution("citations", "Cite sources.")
    )
    reset = original.merge(AgentProfile(system=replacement))
    assert reset.system == replacement
    assert reset.capability_instructions == ()
    assert reset.tools == ("read_document",)
    assert reset.history is True
    assert original.capability_instructions[0].identity == "citations"


def test_model_only_override_retains_instructions():
    original = AgentProfile(system="Base.").merge(contribution("a", "A."))
    changed = original.merge(AgentProfile(model="fixture-model"))
    assert changed.system == original.system
    assert changed.capability_instructions == original.capability_instructions


def test_profile_reset_then_append_can_be_evaluated_before_application():
    outer = AgentProfile(system="Outer.").merge(contribution("old", "Old."))
    profile = AgentProfile.empty().merge(AgentProfile(system="New."))
    profile = profile.merge(contribution("new", "New contribution."))
    combined = outer.merge(profile)
    assert combined.system == "New."
    assert combined.capability_instructions == profile.capability_instructions


def test_explicit_reuse_after_reset_restores_guidance_once():
    capability = contribution("a", "A.")
    original = AgentProfile(system="Base.").merge(capability)
    reset = original.merge(AgentProfile(system="Base."))
    restored = reset.merge(capability).merge(capability)
    assert restored == original


def test_same_text_from_distinct_capabilities_retains_ownership():
    result = contribution("a", "Shared text.").merge(contribution("b", "Shared text."))
    assert [part.identity for part in result.capability_instructions] == ["a", "b"]


def test_instruction_contribution_is_immutable():
    part = CapabilityInstruction(identity="a", content="A.")
    with pytest.raises(FrozenInstanceError):
        part.content = "Changed."


def test_definition_detaches_member_sequence_and_private_binding_maps():
    members = [ProfileSystemMember(plain_value="Read exact revisions.")]
    bindings = {}
    definition = ProfileDefinition(
        name="reader", kind="capability", members=members, profile_bindings=bindings
    )
    members.clear()
    bindings["other"] = ProfileDefinition(name="other")
    assert len(definition.members) == 1
    assert definition.profile_bindings == {}
    with pytest.raises(TypeError):
        definition.profile_bindings["other"] = bindings["other"]
    copied = deepcopy(definition)
    assert copied == definition and copied is not definition
    with pytest.raises(TypeError):
        copied.profile_bindings["other"] = bindings["other"]


def test_definition_detaches_mutable_instruction_ast():
    text = TextSegment(text="Original guidance.")
    line = TemplateLine(segments=[text])
    member = ProfileSystemMember(template_lines=(line,))
    definition = ProfileDefinition(name="reader", kind="capability", members=[member])
    text.text = "Changed guidance."
    line.segments.clear()
    captured = definition.members[0].template_lines[0].segments
    assert len(captured) == 1 and captured[0].text == "Original guidance."


def test_capability_snapshot_detaches_settings_and_mcp_credentials():
    class SDKObject:
        def __deepcopy__(self, memo):
            pytest.fail("SDK objects must retain their runtime identity")

    sdk = SDKObject()
    settings = {"reasoning": {"effort": "low"}, "stop": ["done"], "sdk": sdk}
    headers = {"Authorization": "test-token"}
    environment = {"MODE": "read"}
    profile = AgentProfile(
        settings=settings,
        mcp_servers=(
            McpServerSpec(command="reader", env=environment),
            McpServerSpec(transport="http", url="https://example.invalid/mcp", headers=headers),
        ),
    )
    capability = AppliedCapability(
        identity="reader", name="reader", description=None, profile=profile
    )
    settings["reasoning"]["effort"] = "high"
    settings["stop"].append("changed")
    headers["Authorization"] = "changed-token"
    environment["MODE"] = "write"
    assert capability.profile.settings == {
        "reasoning": {"effort": "low"},
        "stop": ["done"],
        "sdk": sdk,
    }
    assert capability.profile.mcp_servers[0].env == {"MODE": "read"}
    assert capability.profile.mcp_servers[1].headers == {"Authorization": "test-token"}


@pytest.mark.parametrize("deferred", [False, True])
def test_reuse_cannot_change_activation_mode_of_already_registered_capability(deferred):
    capability = AppliedCapability(
        identity="module::reader",
        name="reader",
        description=None,
        profile=AgentProfile(tools=("read_document",)),
        deferred=deferred,
    )
    original = AgentProfile(capabilities=(capability,))
    changed = AgentProfile(capabilities=(replace(capability, deferred=not deferred),))
    with pytest.raises(KediExecutionError, match="different defer mode"):
        original.merge(changed)
    assert original.capabilities == (capability,)


def test_concurrent_local_resets_do_not_mutate_captured_profile():
    outer = AgentProfile(system="Outer.").merge(contribution("shared", "Shared."))

    def local(index: int) -> AgentProfile:
        return outer.merge(AgentProfile(system=f"Local {index}.")).merge(
            contribution(str(index), f"Contribution {index}.")
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(local, range(64)))
    for index, result in enumerate(results):
        assert result.system == f"Local {index}."
        assert result.capability_instructions == (
            CapabilityInstruction(identity=str(index), content=f"Contribution {index}."),
        )
    assert outer.system == "Outer."
    assert outer.capability_instructions == (
        CapabilityInstruction(identity="shared", content="Shared."),
    )


def test_request_parts_preserve_capability_provenance_and_runtime_contracts():
    runtime = compile_program(parse_program("= done\n"), adapter=MockAdapter())
    try:
        profile = AgentProfile(system="Base.").merge(contribution("citations", "Cite sources."))
        parts = runtime._agent_instruction_parts(profile, runtime.adapter)
        assert [(part.source, part.identity, part.content) for part in parts[:2]] == [
            ("profile", "profile.system", "Base."),
            ("capability", "capability:citations", "Cite sources."),
        ]
        reset_parts = runtime._agent_instruction_parts(
            profile.merge(AgentProfile(system="Local.")), runtime.adapter
        )
        assert not any(part.source == "capability" for part in reset_parts)
        assert [part for part in parts if part.source not in {"profile", "capability"}] == [
            part for part in reset_parts if part.source != "profile"
        ]
    finally:
        runtime.artifacts.close()


def test_reset_retains_deferred_registration_but_reuse_restores_guidance():
    descriptor = AppliedCapability(
        identity="reader",
        name="reader",
        description="Read sources",
        profile=AgentProfile(system="Cite sources.", tools=("read_document",)),
        deferred=True,
    )
    contribution = AgentProfile(capabilities=(descriptor,))
    original = AgentProfile(system="Base.").merge(contribution)
    reset = original.merge(AgentProfile(system="Base."))
    assert deferred_capabilities(reset)[0].instructions_enabled is False
    assert deferred_capabilities(reset)[0].profile.tools == ("read_document",)
    assert descriptor.instructions_enabled is True
    restored = reset.merge(contribution).merge(contribution)
    assert len(restored.capabilities) == 1
    assert restored.capabilities[0].instructions_enabled is True
    assert restored.capabilities[0].instructions_restored is True
    assert restored.capabilities[0].profile == descriptor.profile


def test_eager_parent_cannot_reintroduce_suppressed_deferred_child():
    child = AppliedCapability(
        identity="child",
        name="child",
        description=None,
        profile=AgentProfile(system="Child guidance."),
        deferred=True,
    )
    parent = AppliedCapability(
        identity="parent",
        name="parent",
        description=None,
        profile=AgentProfile(capabilities=(child,)),
    )
    reset = AgentProfile(capabilities=(child, parent)).merge(AgentProfile(system="New base."))
    assert len(deferred_capabilities(reset)) == 1
    assert deferred_capabilities(reset)[0].instructions_enabled is False


def test_reuse_keeps_tool_and_subagent_catalog_order_stable():
    descriptor = AppliedCapability(
        identity="reader",
        name="reader",
        description=None,
        profile=AgentProfile(tools=("read_document",), subagents=("reviewer",)),
    )
    capability = AgentProfile(
        capabilities=(descriptor,), tools=("read_document",), subagents=("reviewer",)
    )
    profile = (
        AgentProfile.empty()
        .merge(capability)
        .merge(AgentProfile(tools=("lookup_region",), subagents=("planner",)))
    )
    repeated = profile.merge(capability)
    assert repeated.tools == profile.tools == ("read_document", "lookup_region")
    assert repeated.subagents == profile.subagents == ("reviewer", "planner")
