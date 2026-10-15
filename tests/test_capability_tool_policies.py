from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy, ToolApprovalDenied
from kedi.agent_adapter.codemode import CodeModeLimits
from kedi.agent_adapter.codemode.tool_specs import CodeModeToolSpecBridge, approval_policies
from kedi.agent_adapter.hooks import (
    HookDispatcher,
    HookDispatcherSnapshot,
    PreToolUseDecision,
    bind_hook_dispatcher,
    dispatch_post_tool_use_sync,
    dispatch_pre_tool_use,
    dispatch_pre_tool_use_sync,
)
from kedi.agent_adapter.tools.specs import ToolSpec, tool_approval_policy
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.agent_profile.capabilities import (
    AppliedCapability,
    CapabilityToolPolicy,
    capability_patch,
    capability_tool_policy,
)
from kedi.agent_profile.definitions import ProfileDefinition
from kedi.errors import KediExecutionError


def capability(name, *, tools=("read_document",), approval=None, hooks=None, nested=()):
    return AppliedCapability(
        identity=f"fixture::{name}",
        name=name,
        description=None,
        profile=AgentProfile(tools=tools, approval=approval, hooks=hooks, capabilities=nested),
    )


def test_policy_does_not_change_caller_default_or_unrelated_tools():
    owned = capability("reader", approval=ApprovalPolicy.allow())
    profile = AgentProfile(approval=ApprovalPolicy.deny(), capabilities=(owned,))
    policy = capability_tool_policy(profile, "read_document")
    assert policy is not None and policy.approval == ApprovalPolicy.allow()
    assert capability_tool_policy(profile, "write_document") is None
    assert profile.approval == ApprovalPolicy.deny()


def test_explicit_tool_policy_wins_over_capability_policy():
    policy = CapabilityToolPolicy(owners=("reader",), approval=ApprovalPolicy.allow())
    tool = ToolSpec(
        fn=lambda: "ok",
        name="read_document",
        approval=ApprovalPolicy.deny(),
        capability_policy=policy,
    )
    assert tool_approval_policy(tool, ApprovalPolicy.allow()) == ApprovalPolicy.deny()
    assert (
        tool_approval_policy(replace(tool, approval=None), ApprovalPolicy.deny())
        == ApprovalPolicy.allow()
    )
    assert (
        tool_approval_policy(
            replace(tool, approval=None, capability_policy=None), ApprovalPolicy.deny()
        )
        == ApprovalPolicy.deny()
    )


def test_nested_policy_inherits_parent_only_when_child_does_not_declare_one():
    for child_policy, expected in [
        (None, ApprovalPolicy.deny()),
        (ApprovalPolicy.allow(), ApprovalPolicy.allow()),
    ]:
        child = capability("child", approval=child_policy)
        parent = capability("parent", approval=ApprovalPolicy.deny(), nested=(child,))
        policy = capability_tool_policy(AgentProfile(capabilities=(child, parent)), "read_document")
        assert policy is not None and policy.approval == expected
        assert policy.owners == ("fixture::parent", "fixture::child")


def test_conflicting_sibling_ownership_is_not_last_wins():
    profile = AgentProfile(
        capabilities=(
            capability("reader", approval=ApprovalPolicy.allow()),
            capability("restricted", approval=ApprovalPolicy.deny()),
        )
    )
    with pytest.raises(KediExecutionError, match="Conflicting capability approval.*read_document"):
        capability_tool_policy(profile, "read_document")


def test_diamond_hook_contributions_are_deduplicated_by_owner():
    def observed(event):
        return None

    shared = capability("shared", hooks=HookSettings(post_tool_use=(observed,)))
    left = capability("left", nested=(shared,))
    right = capability("right", nested=(shared,))
    profile = AgentProfile(capabilities=(shared, left, right))
    policy = capability_tool_policy(profile, "read_document")
    assert policy is not None
    assert policy.hook_contributions == (("fixture::shared", shared.profile.hooks),)
    assert policy.hooks.post_tool_use == (observed,)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_owned_hooks_append_without_leaking_to_unrelated_tools(asynchronous):
    events = []

    def base(event):
        events.append(("base", event.tool_name, event.sequence))
        return PreToolUseDecision.continue_()

    def owned(event):
        events.append(("owned", event.tool_name, event.sequence))
        return PreToolUseDecision.continue_()

    policy = CapabilityToolPolicy(
        owners=("reader",),
        hook_contributions=(("reader", HookSettings(pre_tool_use=(owned,))),),
    )
    tool = ToolSpec(fn=lambda: "ok", name="read_document", capability_policy=policy)
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(None, HookSettings(pre_tool_use=(base,)))
    )
    with bind_hook_dispatcher(dispatcher):
        for index, spec in enumerate(
            (tool, replace(tool, name="other", capability_policy=None), tool)
        ):
            kwargs = dict(adapter_shortname="test", tool_call_id=f"call_{index}")
            if asynchronous:
                asyncio.run(dispatch_pre_tool_use(spec, {}, **kwargs))
            else:
                dispatch_pre_tool_use_sync(spec, {}, **kwargs)
    assert [(kind, name) for kind, name, _ in events] == [
        ("base", "read_document"),
        ("owned", "read_document"),
        ("base", "other"),
        ("base", "read_document"),
        ("owned", "read_document"),
    ]
    assert events[0][2] == events[1][2] < events[2][2] < events[3][2] == events[4][2]
    assert dispatcher.handlers_for("pre_tool_use") == (base,)


def test_local_post_hook_runs_without_agent_wide_hooks():
    events = []
    policy = CapabilityToolPolicy(
        owners=("reader",),
        hook_contributions=(("reader", HookSettings(post_tool_use=(events.append,))),),
    )
    tool = ToolSpec(fn=lambda: "ok", name="read_document", capability_policy=policy)
    dispatch_post_tool_use_sync(tool, {}, "ok", adapter_shortname="test", tool_call_id="call")
    assert len(events) == 1 and events[0].result == "ok"


def test_parallel_tool_hook_chains_do_not_mutate_shared_dispatcher():
    barrier = Barrier(8)
    dispatcher = HookDispatcher(HookDispatcherSnapshot.compose(None, None))

    def run(index):
        seen = []

        def observe(event):
            barrier.wait(timeout=5)
            seen.append((event.tool_name, event.sequence))
            return PreToolUseDecision.continue_()

        name = f"tool_{index}"
        tool = ToolSpec(
            fn=lambda: "ok",
            name=name,
            capability_policy=CapabilityToolPolicy(
                owners=(name,),
                hook_contributions=((name, HookSettings(pre_tool_use=(observe,))),),
            ),
        )
        with bind_hook_dispatcher(dispatcher):
            dispatch_pre_tool_use_sync(tool, {}, adapter_shortname="test", tool_call_id=name)
        assert len(seen) == 1 and seen[0][0] == name
        return seen[0][1]

    with ThreadPoolExecutor(max_workers=8) as pool:
        sequences = list(pool.map(run, range(64)))
    assert sorted(sequences) == list(range(1, 65))
    assert dispatcher.handlers_for("pre_tool_use") == ()


def test_deferred_policy_is_not_installed_by_use():
    definition = ProfileDefinition(name="reader", kind="capability")
    patch = capability_patch(
        definition,
        AgentProfile(tools=("read_document",), approval=ApprovalPolicy.allow()),
        deferred=True,
    )
    assert capability_tool_policy(patch, "read_document") is None


@pytest.mark.parametrize("enforced_deny", [False, True])
def test_pydantic_native_approval_uses_owned_policy_but_retains_ceiling(enforced_deny):
    calls = []

    def read_document() -> str:
        calls.append("read")
        return "evidence"

    def write_document() -> str:
        calls.append("write")
        return "saved"

    policy = CapabilityToolPolicy(owners=("reader",), approval=ApprovalPolicy.allow())
    tools = (
        ToolSpec(fn=read_document, name="read_document", capability_policy=policy),
        ToolSpec(fn=write_document, name="write_document"),
    )
    adapter = PydanticAdapter(TestModel(call_tools=[tool.name for tool in tools]))
    with (
        adapter.tool_scope(tools),
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())),
        adapter.approval_ceiling((ApprovalPolicy.deny(),) if enforced_deny else ()),
    ):
        adapter.run_sync("Read and write the document.")
    assert calls == ([] if enforced_deny else ["read"])


@pytest.mark.parametrize("enforced_deny", [False, True])
def test_langchain_native_approval_uses_owned_policy_but_retains_ceiling(enforced_deny):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from kedi.agent_adapter.adapters import LangChainAdapter

    class ToolChatModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    calls = []

    def read_document() -> str:
        calls.append("read")
        return "evidence"

    def write_document() -> str:
        calls.append("write")
        return "saved"

    policy = CapabilityToolPolicy(owners=("reader",), approval=ApprovalPolicy.allow())
    tools = (
        ToolSpec(fn=read_document, name="read_document", capability_policy=policy),
        ToolSpec(fn=write_document, name="write_document"),
    )
    model = ToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool.name, "args": {}, "id": f"call_{index}"}
                    for index, tool in enumerate(tools)
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    adapter = LangChainAdapter(model)
    with (
        adapter.tool_scope(tools),
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())),
        adapter.approval_ceiling((ApprovalPolicy.deny(),) if enforced_deny else ()),
    ):
        asyncio.run(adapter.invoke(prompt="Read and write the document."))
    assert calls == ([] if enforced_deny else ["read"])


@pytest.mark.parametrize("enforced_deny", [False, True])
def test_codemode_resolves_underlying_tool_policy_without_losing_ceiling(enforced_deny):
    tool = ToolSpec(
        fn=lambda: "evidence",
        name="read_document",
        json_schema={"type": "object", "properties": {}},
        capability_policy=CapabilityToolPolicy(owners=("reader",), approval=ApprovalPolicy.allow()),
    )
    bridge = CodeModeToolSpecBridge(
        [tool],
        limits=CodeModeLimits(),
        adapter_shortname="test",
        policies=(),
        policy_resolver=lambda spec: approval_policies(
            tool_approval_policy(spec, ApprovalPolicy.deny()),
            (ApprovalPolicy.deny(),) if enforced_deny else (),
        ),
    )

    async def run():
        try:
            if enforced_deny:
                with pytest.raises(ToolApprovalDenied):
                    await bridge._invoke(tool.name, {})
            else:
                assert await bridge._invoke(tool.name, {}) == "evidence"
        finally:
            await bridge.close()

    asyncio.run(run())
