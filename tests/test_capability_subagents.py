"""Shared child-control tools retain per-child capability ownership."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.hooks import HookDeniedError, PreToolUseDecision
from kedi.agent_adapter.hooks.tool_wrapper import dispatch_post_tool_use, dispatch_pre_tool_use
from kedi.agent_adapter.subagents.ownership import with_subagent_ownership
from kedi.agent_adapter.subagents.runs import SubagentRunRegistry
from kedi.agent_adapter.tools.specs import ToolSpec, tool_approval_policy, tool_capability_policy
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.capabilities import AppliedCapability
from kedi.agent_profile.definitions import ProfileDefinition, ProfileSystemMember
from kedi.agent_profile.hooks import HookSettings
from kedi.lang import compile_program, parse_program


def owned_tools(*, before=None, after=None):
    definitions = {name: ProfileDefinition(name=name) for name in ("researcher", "writer")}
    capability = AppliedCapability(
        identity="reader-capability",
        name="reader",
        description=None,
        profile=AgentProfile(
            subagents=("researcher",),
            approval=ApprovalPolicy.allow(),
            hooks=HookSettings(
                pre_tool_use=(before,) if before else (),
                post_tool_use=(after,) if after else (),
            ),
        ),
    )
    profile = AgentProfile(
        name="host",
        subagents=("researcher", "writer"),
        capabilities=(capability,),
        approval=ApprovalPolicy.deny(),
    )
    runs = SubagentRunRegistry()
    for name, owner in (("researcher", "host"), ("writer", "host"), ("researcher", "other")):
        runs.create(
            run_id=f"{owner}-{name}",
            profile_name=name,
            owner_profile_name=owner,
            parent_run_id=None,
            root_run_id="root",
            depth=1,
            profile_definition=definitions[name],
        )
    rt = SimpleNamespace(
        _subagent_coordinator=SimpleNamespace(runs=runs), _profile_registry=definitions
    )
    tools = with_subagent_ownership(
        [
            ToolSpec(name=name, fn=lambda **kwargs: None, metadata={"kind": kind})
            for name, kind in (
                ("delegate_task", "subagent"),
                ("continue_subagent", "subagent"),
                ("send_subagent_input", "subagent_lifecycle"),
                ("send_parent_message", "subagent_lifecycle"),
            )
        ],
        rt,
        profile,
    )
    return {tool.name: tool for tool in tools}, profile


def test_subagent_policy_uses_target_not_shared_tool_name():
    tools, profile = owned_tools()
    for tool, arguments in (
        (tools["delegate_task"], {"subagent": "researcher"}),
        (tools["continue_subagent"], {"run_id": "host-researcher"}),
        (tools["send_subagent_input"], {"task_id": "host-researcher"}),
    ):
        assert tool_capability_policy(tool, arguments).owners == ("reader-capability",)
        assert tool_approval_policy(tool, profile.approval, arguments) == ApprovalPolicy.allow()
    for arguments in (
        {"subagent": "writer"},
        {"run_id": "other-researcher"},
        {"run_id": "missing"},
        {},
    ):
        assert tool_capability_policy(tools["delegate_task"], arguments) is None
    assert tools["send_parent_message"].capability_policy_resolver is None


@pytest.mark.parametrize("key", ["run_id", "task_id"])
@pytest.mark.parametrize("binding", ["missing", "different", "restored", "matching", "copied"])
def test_lifecycle_requires_concrete_profile_binding(key, binding):
    events = []

    def before(event):
        events.append(event.tool_name)
        return PreToolUseDecision.continue_()

    definition = ProfileDefinition(
        name="researcher", members=(ProfileSystemMember(plain_value="Read evidence."),)
    )
    capability = AppliedCapability(
        identity="research",
        name="research",
        description=None,
        profile=AgentProfile(
            subagents=("researcher",),
            subagent_bindings={"researcher": definition},
            approval=ApprovalPolicy.allow(),
            hooks=HookSettings(pre_tool_use=(before,)),
        ),
    )
    profile = AgentProfile(
        name="host",
        subagents=("researcher",),
        subagent_bindings={"researcher": definition},
        capabilities=(capability,),
        approval=ApprovalPolicy.deny(),
    )
    runs = SubagentRunRegistry()
    run = runs.create(
        run_id="unverified",
        profile_name="researcher",
        owner_profile_name="host",
        parent_run_id=None,
        root_run_id="unverified",
        depth=1,
        profile_definition=(
            None
            if binding == "missing"
            else replace(definition, members=(ProfileSystemMember(plain_value="Modify evidence."),))
            if binding == "different"
            else deepcopy(definition)
            if binding == "copied"
            else definition
        ),
    )
    if binding == "restored":
        runs = SubagentRunRegistry()
        runs.restore(run)
    assert "profile_definition" not in asdict(run)
    runtime = SimpleNamespace(
        _subagent_coordinator=SimpleNamespace(runs=runs), _profile_registry={}
    )
    (tool,) = with_subagent_ownership(
        (
            ToolSpec(
                name="send_subagent_input",
                fn=lambda **kwargs: None,
                metadata={"kind": "subagent_lifecycle"},
            ),
        ),
        runtime,
        profile,
    )
    if binding in {"matching", "copied"}:
        assert tool_capability_policy(tool, {key: "unverified"}).owners == ("research",)
        assert (
            tool_approval_policy(tool, profile.approval, {key: "unverified"})
            == ApprovalPolicy.allow()
        )
    else:
        assert tool_capability_policy(tool, {key: "unverified"}) is None
        assert (
            tool_approval_policy(tool, profile.approval, {key: "unverified"})
            == ApprovalPolicy.deny()
        )
    asyncio.run(
        dispatch_pre_tool_use(
            tool, {key: "unverified"}, adapter_shortname="fixture", tool_call_id="control"
        )
    )
    assert events == (["send_subagent_input"] if binding in {"matching", "copied"} else [])
    runs.discard(run.run_id)
    assert not runs.matches_profile(run.run_id, definition)


@pytest.mark.asyncio
async def test_owned_hooks_do_not_run_for_other_children_or_allow_owner_switch():
    events = []

    def before(event):
        events.append(("pre", event.arguments["subagent"]))
        return PreToolUseDecision.continue_()

    def after(event):
        events.append(("post", event.arguments["subagent"]))

    tools, _profile = owned_tools(before=before, after=after)
    for name in ("researcher", "writer"):
        arguments = {"subagent": name, "task": "Review evidence."}
        await dispatch_pre_tool_use(
            tools["delegate_task"], arguments, adapter_shortname="fixture", tool_call_id=name
        )
        await dispatch_post_tool_use(
            tools["delegate_task"],
            arguments,
            "done",
            adapter_shortname="fixture",
            tool_call_id=name,
        )
    assert events == [("pre", "researcher"), ("post", "researcher")]

    tools, _profile = owned_tools(
        before=lambda event: PreToolUseDecision.edit({**event.arguments, "subagent": "writer"})
    )
    with pytest.raises(HookDeniedError, match="cannot change capability ownership"):
        await dispatch_pre_tool_use(
            tools["delegate_task"],
            {"subagent": "researcher"},
            adapter_shortname="fixture",
            tool_call_id="redirect",
        )


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("workflow", ["delegate", "dynamic"])
def test_dsl_delegation_hooks_are_owned_and_do_not_leak_into_children(kind, workflow):
    events = []
    completed_children = []
    tool_name = "delegate_task" if workflow == "delegate" else "run_workflow"
    calls = (
        [
            {"name": tool_name, "args": {"subagent": name, "task": "Review evidence."}, "id": name}
            for name in ("researcher", "writer")
        ]
        if workflow == "delegate"
        else [
            {
                "name": tool_name,
                "args": {
                    "code": (
                        "import asyncio\n"
                        "results = await asyncio.gather(\n"
                        "    researcher(task='Review evidence.'),\n"
                        "    writer(task='Review evidence.'))\n"
                        "repeat = await researcher(task='Review evidence.')\n"
                        "results"
                    )
                },
                "id": "workflow",
            }
        ]
    )

    def before(event):
        events.append(("pre", event.arguments.get("subagent", event.tool_name)))
        return PreToolUseDecision.continue_()

    def after(event):
        events.append(("post", event.arguments.get("subagent", event.tool_name)))

    def respond(messages, info):
        if tool_name not in {tool.name for tool in info.function_tools}:
            completed_children.append("child")
            return ModelResponse([TextPart("Child review completed.")])
        if not any(
            isinstance(part, ToolReturnPart) for message in messages for part in message.parts
        ):
            return ModelResponse(
                [ToolCallPart(call["name"], call["args"], call["id"]) for call in calls]
            )
        return ModelResponse([TextPart("done")])

    class Pydantic(PydanticAdapter):
        capabilities = replace(PydanticAdapter.capabilities, supports_capabilities=True)

    class LangChain(LangChainAdapter):
        capabilities = replace(LangChainAdapter.capabilities, supports_capabilities=True)

    class ChatModel(FakeMessagesListChatModel):
        parent_tools: bool = False

        def bind_tools(self, tools, **kwargs):
            return self.model_copy(
                update={"parent_tools": any(tool.name == tool_name for tool in tools)}
            )

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if not self.parent_tools:
                completed_children.append("child")
                message = AIMessage(content="Child review completed.")
            elif not any(isinstance(message, ToolMessage) for message in messages):
                message = AIMessage(
                    content="",
                    tool_calls=calls,
                )
            else:
                message = AIMessage(content="done")
            return ChatResult(generations=[ChatGeneration(message=message)])

    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n  > instructions: Read the evidence.\n"
            "> profile: writer:\n  > instructions: Draft a summary.\n"
            "> capability: research:\n"
            "  > subagent: researcher\n"
            "  > hooks:\n    pre_tool_use: `before`\n    post_tool_use: `after`\n"
            "> profile: host:\n  > use: research\n  > subagent: writer\n"
            f"  > workflow: {workflow}\n"
            "> use: host\n[reply] << Review the evidence with both specialists.\n= <reply>\n"
        ),
        adapter=Pydantic(FunctionModel(respond))
        if kind == "pydantic"
        else LangChain(ChatModel(responses=[])),
        runtime_globals={"before": before, "after": after},
    )
    try:
        assert runtime.run_main() == "done"
        assert completed_children == ["child", "child"]
        assert events == [("pre", "researcher"), ("post", "researcher")]
        runs = runtime._subagent_coordinator.runs
        for run in runs.snapshot():
            assert runs.matches_profile(run.run_id, runtime._profile_registry[run.profile_name])
    finally:
        asyncio.run(runtime.aclose())
