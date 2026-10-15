from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from kedi import Capability
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters.pydantic.capabilities import _KediArtifactCapability
from kedi.agent_adapter.adapters.pydantic.user_capabilities import _DeferredCapabilityGraph
from kedi.agent_adapter.tools.errors import CapabilityActivationError, correctable_tool_error
from kedi.agent_adapter.tools.specs import ToolSpec


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{", "not json", "[]"])
async def test_malformed_tool_arguments_stay_with_native_validator(arguments):
    profile = Capability(name="reader", defer=True).as_profile()
    graph = _DeferredCapabilityGraph(profile, ())
    seen = []

    async def validate(args):
        seen.append(args)
        raise ValueError("Invalid JSON")

    with pytest.raises(ValueError, match="Invalid JSON"):
        await graph.wrap_tool_validate(
            SimpleNamespace(active_capability_ids=set()),
            call=ToolCallPart("read", arguments),
            tool_def=ToolDefinition(name="read"),
            args=arguments,
            handler=validate,
        )
    assert seen == [arguments]


@pytest.mark.asyncio
async def test_native_tool_cannot_bypass_its_inactive_capability():
    graph = _DeferredCapabilityGraph(Capability(name="reader", defer=True).as_profile(), ())
    identity = next(iter(graph.boundaries))
    executed = []

    async def validate(args):
        executed.append(args)
        return args

    with pytest.raises(ModelRetry, match="Load this tool's capability"):
        await graph.wrap_tool_validate(
            SimpleNamespace(active_capability_ids=set()),
            call=ToolCallPart("read", {}),
            tool_def=ToolDefinition(name="read", capability_id=identity),
            args={},
            handler=validate,
        )
    assert executed == []


@pytest.mark.asyncio
async def test_artifact_capability_without_admission_preserves_result():
    capability = _KediArtifactCapability(PydanticAdapter(TestModel()))
    result = object()
    assert await capability.after_run(SimpleNamespace(), result=result) is result


@pytest.mark.asyncio
async def test_deferred_run_rejects_invalid_native_capability_collection():
    adapter = PydanticAdapter(TestModel())
    with adapter.profile_override(Capability(name="reader", defer=True).as_profile()):
        with pytest.raises(TypeError, match="capabilities must be a list or tuple"):
            await adapter.invoke(prompt="Read the source.", capabilities={"invalid": True})


def test_activation_errors_are_correctable_but_arbitrary_codemode_errors_are_not():
    tool = ToolSpec(
        name="execute", fn=lambda: None, metadata={"kedi_tool_kind": "codemode_control"}
    )
    assert (
        correctable_tool_error(tool, CapabilityActivationError("Load reader first"))
        == "Load reader first"
    )
    assert correctable_tool_error(tool, ValueError("Unrelated error")) is None


@pytest.mark.asyncio
async def test_codemode_resolves_argument_sensitive_ownership_before_execution():
    from kedi.agent_adapter.approvals.policy import ApprovalPolicy
    from kedi.agent_adapter.codemode import CodeModeLimits
    from kedi.agent_adapter.codemode.tool_specs import CodeModeToolSpecBridge
    from kedi.agent_profile.capabilities import CapabilityToolPolicy

    observed = []

    def policy(arguments):
        observed.append(dict(arguments))
        return CapabilityToolPolicy(owners=("reader",), approval=ApprovalPolicy.allow())

    tool = ToolSpec(name="read", fn=lambda path: path, capability_policy_resolver=policy)
    bridge = CodeModeToolSpecBridge(
        (tool,), limits=CodeModeLimits(), policies=(), adapter_shortname="pydantic"
    )
    try:
        assert await bridge._invoke("read", {"path": "document"}) == "document"
        assert observed and all(args == {"path": "document"} for args in observed)
    finally:
        await bridge.close()


def test_hook_argument_edit_preserves_ownership_for_same_capability():
    from kedi.agent_adapter.hooks import PreToolUseDecision
    from kedi.agent_adapter.hooks.tool_wrapper import _resolve_pre_decision
    from kedi.agent_profile.capabilities import CapabilityToolPolicy

    tool = ToolSpec(
        name="read",
        fn=lambda path: path,
        json_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        capability_policy_resolver=lambda args: CapabilityToolPolicy(owners=("reader",)),
    )
    decision = PreToolUseDecision.edit({"path": "new"})
    assert _resolve_pre_decision(tool, decision, {"path": "old"}) == {"path": "new"}


@pytest.mark.asyncio
async def test_owned_dynamic_child_validates_and_returns_structured_result():
    from kedi.agent_adapter.approvals.policy import ApprovalPolicy
    from kedi.agent_profile.compile import build_capture_env
    from kedi.agent_profile.evaluation import eval_profile_definition
    from tests.test_subagents import _dynamic_workflow_tool

    adapter, runtime, _ = _dynamic_workflow_tool()
    try:
        profile = eval_profile_definition(
            runtime, runtime._profile_registry["manager"], build_capture_env(runtime)
        )
        profile = profile.merge(
            Capability(
                name="workers",
                subagents={"worker": runtime._profile_registry["worker"]},
                approval=ApprovalPolicy.allow(),
            ).as_profile()
        )
        tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}
        result = await tools["run_workflow"].fn(
            code=(
                "result = await worker(task='structured', final_schema={'type': 'object', "
                "'properties': {'value': {'type': 'integer'}}, 'required': ['value']})\n"
                "result['final_result']['value']"
            )
        )
        assert result == 7
    finally:
        await runtime.aclose()
