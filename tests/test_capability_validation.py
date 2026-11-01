from copy import deepcopy
from dataclasses import replace

import pytest

from kedi import Capability
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, McpServerSpec
from kedi.agent_profile.capabilities import (
    AppliedCapability,
    capability_subagent_policy,
    capability_tool_policy,
    deferred_boundaries,
)
from kedi.agent_profile.capture import CapturedAgentProfile
from kedi.agent_profile.definitions import (
    ProfileDeferMember,
    ProfileDefinition,
    ProfileHooksMember,
    ProfileModelMember,
    ProfileSubagentMember,
    ProfileSystemMember,
    ProfileToolMember,
    ProfileUseMember,
)
from kedi.agent_profile.evaluation import _bound_capability_tool, eval_profile_definition
from kedi.agent_profile.hooks import HookSettings
from kedi.agent_profile.use_resolution import use_needs_lazy_capture
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import UseDirectiveStmt
from tests.mock_adapter import MockAdapter


@pytest.fixture
def runtime():
    rt = compile_program(parse_program(""), adapter=MockAdapter())
    try:
        yield rt
    finally:
        rt.artifacts.close()


@pytest.mark.parametrize("operation", ["resolve", "apply"])
def test_recursive_capture_fails_without_poisoning_future_evaluation(operation):
    recursive = True

    def evaluate(env):
        if recursive:
            if operation == "resolve":
                capture.resolve()
            else:
                capture.apply(AgentProfile(), env)
        return AgentProfile(system="Recovered.")

    capture = CapturedAgentProfile(AgentProfile(), evaluate, {})
    assert deepcopy(capture) is capture
    with pytest.raises(KediExecutionError, match="Recursive agent profile capture"):
        capture.resolve()
    recursive = False
    assert capture.resolve().system == "Recovered."


def test_capability_subagent_alias_must_match_definition():
    with pytest.raises(ValueError, match="name"):
        Capability(name="bundle", subagents={"alias": ProfileDefinition(name="worker")})


@pytest.mark.parametrize("deferred", [False, True])
def test_reused_capability_identity_cannot_replace_subagent(deferred):
    worker = ProfileDefinition(name="worker")
    first = Capability(
        name="bundle", identity="bundle", defer=deferred, subagents={"worker": worker}
    )
    profile = first.as_profile()
    assert profile.merge(first.as_profile()).capabilities
    changed = replace(worker, members=(ProfileSystemMember(plain_value="Different."),))
    second = Capability(
        name="bundle", identity="bundle", defer=deferred, subagents={"worker": changed}
    )
    with pytest.raises(KediExecutionError, match="Conflicting capability subagent binding"):
        profile.merge(second.as_profile())


def test_non_capability_subagent_bindings_still_override():
    worker = ProfileDefinition(name="worker")
    changed = replace(worker, members=(ProfileSystemMember(plain_value="New."),))
    merged = AgentProfile(subagent_bindings={"worker": worker}).merge(
        AgentProfile(subagent_bindings={"worker": changed})
    )
    assert merged.subagent_bindings["worker"] == changed


def test_capability_mcp_composition_deduplicates_only_identical_servers():
    first = McpServerSpec(command="first")
    second = McpServerSpec(command="second")
    capability = Capability(name="sources", mcp_servers=(first, second)).as_profile()
    merged = AgentProfile(mcp_servers=(first,)).merge(capability).merge(capability)
    assert merged.mcp_servers == (first, second)


def test_profile_snapshot_detaches_mutable_values_inside_tuples():
    original = AgentProfile(settings={"nested": (["initial"],)})
    snapshot = original.frozen_copy()
    original.settings["nested"][0].append("changed")
    assert snapshot.settings["nested"] == (["initial"],)


@pytest.mark.parametrize(
    ("kind", "member", "message"),
    [
        ("profile", ProfileDeferMember(), "defer is only valid"),
        ("capability", ProfileModelMember(plain_value="test"), "not allowed"),
        ("capability", ProfileSubagentMember(name="missing"), "Unknown subagent"),
        ("capability", ProfileHooksMember(enabled=False), "owned tool hooks"),
    ],
)
def test_programmatic_definitions_obey_language_scope_rules(runtime, kind, member, message):
    definition = ProfileDefinition(name="invalid", kind=kind, members=(member,))
    with pytest.raises(KediExecutionError, match=message):
        eval_profile_definition(runtime, definition, {})


@pytest.mark.parametrize("member_type", [ProfileUseMember, ProfileToolMember])
def test_captured_python_capability_is_discovered_and_evaluated(runtime, member_type):
    capability = Capability(name="reader", instructions="Read sources.")
    member = (
        member_type(name="reader")
        if member_type is ProfileUseMember
        else member_type(tool_names=("reader",))
    )
    owner = ProfileDefinition(
        name="owner", capability_bindings={"reader": capability}, members=(member,)
    )
    assert use_needs_lazy_capture(
        runtime, UseDirectiveStmt(target="owner"), {"owner": owner}, {}, available_procedures=set()
    )
    profile = eval_profile_definition(runtime, owner, {})
    assert [part.content for part in profile.capability_instructions] == ["Read sources."]


def test_python_capability_in_runtime_namespace_can_be_used_from_profile(runtime):
    runtime._globals_env["reader"] = Capability(name="reader", instructions="Read sources.")
    owner = ProfileDefinition(name="owner", members=(ProfileUseMember(name="reader"),))
    assert eval_profile_definition(runtime, owner, {}).capabilities


def test_capability_tool_resolves_runtime_callable_and_rejects_missing_binding(runtime):
    def read():
        return "source"

    runtime._globals_env["read"] = read
    owner = ProfileDefinition(
        name="owner", kind="capability", members=(ProfileUseMember(name="read"),)
    )
    assert not use_needs_lazy_capture(
        runtime, UseDirectiveStmt(target="read"), {}, {}, available_procedures=set()
    )
    profile = eval_profile_definition(runtime, owner, {})
    assert profile.tool_bindings["read"].fn() == "source"
    with pytest.raises(KediExecutionError, match="Unknown tool"):
        _bound_capability_tool(runtime, owner, "missing", {})


def test_deferred_binding_validation_waits_for_activation():
    worker = ProfileDefinition(name="worker")
    read = ToolSpec(name="read", fn=lambda: "source")
    profile = Capability(
        name="reader", defer=True, tools=(read,), subagents={"worker": worker}
    ).as_profile()
    other_tool = replace(read, fn=lambda: "different")
    other_worker = replace(worker, members=(ProfileSystemMember(plain_value="Different."),))
    assert capability_tool_policy(profile, "read", tool=other_tool) is None
    assert capability_subagent_policy(profile, "worker", definition=other_worker) is None
    with pytest.raises(KediExecutionError, match="different callable"):
        capability_tool_policy(profile, "read", tool=other_tool, include_deferred=True)
    with pytest.raises(KediExecutionError, match="different profile"):
        capability_subagent_policy(
            profile, "worker", definition=other_worker, include_deferred=True
        )


def test_catalog_rejects_deferred_subagent_overwriting_existing_binding():
    from kedi.core.capabilities import subagent_catalog_profile

    worker = ProfileDefinition(name="worker")
    profile = Capability(name="reader", defer=True, subagents={"worker": worker}).as_profile()
    conflicting = replace(worker, members=(ProfileSystemMember(plain_value="Other."),))
    profile = replace(profile, subagent_bindings={"worker": conflicting})
    with pytest.raises(KediExecutionError, match="Conflicting deferred subagent binding"):
        subagent_catalog_profile(profile)


def test_repeated_ancestor_identity_is_rejected_as_cyclic():
    child = AppliedCapability(
        identity="cycle", name="child", description=None, profile=AgentProfile()
    )
    parent = replace(child, name="parent", profile=AgentProfile(capabilities=(child,)))
    with pytest.raises(KediExecutionError, match="Cyclic capability graph"):
        deferred_boundaries(AgentProfile(capabilities=(parent,)))


def test_conflicting_hooks_for_same_owner_report_only_related_capabilities():
    first = AppliedCapability(
        identity="shared",
        name="first",
        description=None,
        profile=AgentProfile(tools=("read",), hooks=HookSettings(enabled=True)),
    )
    second = replace(
        first,
        name="second",
        profile=AgentProfile(tools=("read",), hooks=HookSettings(enabled=False)),
    )
    unrelated = replace(first, identity="other", name="other", profile=AgentProfile())
    with pytest.raises(KediExecutionError, match="Conflicting capability hooks") as error:
        capability_tool_policy(AgentProfile(capabilities=(first, unrelated, second)), "read")
    assert "capability other" not in str(error.value)


def test_non_owning_capability_does_not_prevent_normal_subagent_override():
    first = ProfileDefinition(name="worker")
    second = replace(first, members=(ProfileSystemMember(plain_value="Changed."),))
    profile = AgentProfile(subagent_bindings={"worker": first}).merge(
        Capability(name="unrelated").as_profile()
    )
    assert (
        profile.merge(AgentProfile(subagent_bindings={"worker": second})).subagent_bindings[
            "worker"
        ]
        == second
    )


def test_legacy_profile_ast_use_targets_and_tools_keep_their_meaning(runtime):
    from kedi.agent_profile.compile import profile_stmt_to_definition
    from kedi.agent_profile.directives import _member_to_patch
    from kedi.lang.ast import ProfileDefStmt

    definition = profile_stmt_to_definition(ProfileDefStmt(name="reader", use_targets=["source"]))
    assert definition.members == (ProfileUseMember(name="source"),)
    patch = _member_to_patch(runtime, ProfileToolMember(tool_names=("read",)), {})
    assert patch.tools == ("read",)


def test_capability_cannot_be_subagent_in_programmatic_graph():
    from kedi.agent_profile.compile import validate_subagent_graph

    registry = {
        "reader": ProfileDefinition(name="reader", kind="capability"),
        "host": ProfileDefinition(name="host", members=(ProfileSubagentMember(name="reader"),)),
    }
    with pytest.raises(KediExecutionError, match="must name a profile"):
        validate_subagent_graph(registry)


def test_capture_without_external_capture_map_still_retains_capability(runtime):
    from kedi.agent_profile.compile import build_profile_registry, capture_procedure_profiles

    program = parse_program(
        "> capability: reader:\n  > instructions: Read carefully.\n> use: reader\n@read() -> str:\n  = done\n"
    )
    runtime._profile_registry = build_profile_registry(program)
    capture_procedure_profiles(runtime, program)
    proc = program.procedures[0]
    assert isinstance(proc.captured_agent_profile, CapturedAgentProfile)
    assert proc.captured_agent_profile.resolve().capabilities


@pytest.mark.parametrize("with_location", [False, True])
def test_telemetry_handles_definitions_without_source_path(monkeypatch, with_location):
    from kedi.agent_profile import telemetry
    from kedi.lang.ast import Span

    monkeypatch.setattr(telemetry, "capture_enabled", lambda name: True)
    capability = AppliedCapability(
        identity="reader",
        name="reader",
        description=None,
        profile=AgentProfile(),
        loc=Span(start_byte=0, end_byte=1, start_line=1, start_col=1, end_line=1, end_col=2)
        if with_location
        else None,
    )
    attributes = telemetry.capability_attributes(capability)
    assert attributes["kedi.capability.identity"] == "reader"
    assert "kedi.source.path" not in attributes
    assert ("kedi.source.line" in attributes) == with_location


def test_empty_capability_instruction_does_not_emit_an_empty_fragment(runtime):
    from kedi.agent_profile.instructions import CapabilityInstruction

    profile = AgentProfile(
        capability_instructions=(CapabilityInstruction(identity="empty", content=""),)
    )
    parts = runtime._agent_instruction_parts(profile, runtime.adapter)
    assert not any(part.identity == "capability:empty" for part in parts)


def test_captured_budget_directive_updates_active_execution_ledger():
    from kedi.run_budget import current_run_budget
    from tests.test_capability_capture import CapabilityAdapter

    observed = []

    def inspect_budget():
        observed.append(current_run_budget().budget.request_limit)

    rt = compile_program(
        parse_program(
            "> capability: reader:\n  > instructions: Read carefully.\n> use: reader\n"
            "> budget:\n  request_limit: 7\n@read() -> str:\n  = done\n"
            "`inspect_budget()`\n= done\n"
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"inspect_budget": inspect_budget},
    )
    try:
        assert rt.run_main() == "done"
        assert observed == [7]
    finally:
        rt.artifacts.close()
