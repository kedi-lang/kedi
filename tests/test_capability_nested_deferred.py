from __future__ import annotations

import re
import asyncio
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest
from pydantic_ai.messages import (
    CompactionPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    post_compaction_window,
)
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.lang import compile_program, parse_program
from kedi.agent_adapter.hooks import PreToolUseDecision
from kedi import Capability
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.activation import capability_activation_scope, require_subagent_activation
from kedi.agent_adapter.tools.errors import CapabilityActivationError
from kedi.agent_adapter.tools.specs import ToolSpec


def native_id(instructions: str, name: str) -> str:
    return re.search(rf"kedi_{name}_[a-f0-9]{{12}}", instructions).group()


@pytest.mark.parametrize("codemode", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_nested_deferred_catalog_and_execution_follow_parent_activation(codemode, reset):
    requests = []
    executions = []

    def generate_image() -> str:
        executions.append("image")
        return "product.png"

    def respond(messages, info):
        requests.append(info)
        names = {tool.name for tool in info.function_tools}
        if len(requests) == 1:
            assert "kedi_images_" not in info.instructions
            assert "generate_image" not in names
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "social")}, "social"
                    )
                ]
            )
        if len(requests) == 2:
            assert "generate_image" not in names
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "images")}, "images"
                    )
                ]
            )
        if len(requests) == 3:
            if not codemode:
                assert "generate_image" in names
            assert ("Preserve the product appearance." in str(messages)) is not reset
            if codemode:
                return ModelResponse(
                    [ToolCallPart("get_tool_schema", {"tool_names": ["generate_image"]}, "schema")]
                )
            return ModelResponse([ToolCallPart("generate_image", {}, "image")])
        if codemode and len(requests) == 4:
            return ModelResponse(
                [ToolCallPart("execute_code", {"code": "await generate_image()"}, "image")]
            )
        assert executions == ["image"]
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> capability: images:\n  > defer: true\n"
            "  > instructions: Preserve the product appearance.\n"
            "  > approval: allow\n  > use: generate_image\n"
            "> capability: social:\n  > defer: true\n  > use: images\n"
            "> use: social\n"
            + ("> instructions: Replacement rules.\n" if reset else "")
            + "[reply] << Prepare the product image.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond), codemode=codemode),
        runtime_globals={"generate_image": generate_image},
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert len(requests) == 4 + int(codemode)


@pytest.mark.parametrize("guess_first", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_deferred_subagent_is_hidden_and_gated_until_loading(guess_first, nested):
    requests = []
    children = []

    def respond(messages, info):
        if "Research the product." in (info.instructions or ""):
            children.append(messages)
            return ModelResponse([TextPart("Product evidence verified.")])
        requests.append(info)
        names = {tool.name for tool in info.function_tools}
        step = len(requests) - 1
        if guess_first and step == 0:
            assert "delegate_task" not in names
            return ModelResponse(
                [
                    ToolCallPart(
                        "delegate_task",
                        {"subagent": "researcher", "task": "Verify evidence."},
                        "guess",
                    )
                ]
            )
        if step == int(guess_first):
            assert children == []
            assert "delegate_task" not in names
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability",
                        {"id": native_id(info.instructions, "social" if nested else "research")},
                        "load",
                    )
                ]
            )
        if nested and step == int(guess_first) + 1:
            assert "delegate_task" not in names
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability",
                        {"id": native_id(info.instructions, "research")},
                        "research",
                    )
                ]
            )
        if step == int(guess_first) + 1 + int(nested):
            delegate = next(tool for tool in info.function_tools if tool.name == "delegate_task")
            assert delegate.parameters_json_schema["properties"]["subagent"]["enum"] == [
                "researcher"
            ]
            return ModelResponse(
                [
                    ToolCallPart(
                        "delegate_task",
                        {"subagent": "researcher", "task": "Verify evidence."},
                        "delegate",
                    )
                ]
            )
        assert len(children) == 1
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert any(part.tool_name == "delegate_task" for part in returns)
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n  > instructions: Research the product.\n"
            "> capability: research:\n  > defer: true\n  > subagent: researcher\n"
            + (
                "> capability: social:\n  > defer: true\n  > use: research\n> use: social\n"
                if nested
                else "> use: research\n"
            )
            + "[reply] << Verify the product evidence.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond), retries=3),
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert len(requests) == 3 + int(guess_first) + int(nested)


@pytest.mark.parametrize("same_response", [False, True])
def test_guessed_nested_load_cannot_skip_parent_confirmation(same_response):
    steps = []
    hidden = "kedi_images_" + sha256(b"<program>::images").hexdigest()[:12]

    def respond(messages, info):
        steps.append(info)
        if len(steps) == 1:
            assert "kedi_images_" not in info.instructions
        parent = native_id(info.instructions, "social")
        if len(steps) == 1:
            calls = [ToolCallPart("load_capability", {"id": hidden}, "guess")]
            if same_response:
                calls.insert(0, ToolCallPart("load_capability", {"id": parent}, "parent"))
            return ModelResponse(calls)
        if not same_response and len(steps) == 2:
            return ModelResponse([ToolCallPart("load_capability", {"id": parent}, "parent")])
        assert native_id(info.instructions, "images") == hidden
        assert "Hidden image guidance." not in str(messages)
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> capability: images:\n  > defer: true\n  > instructions: Hidden image guidance.\n"
            "> capability: social:\n  > defer: true\n  > use: images\n"
            "> use: social\n[reply] << Inspect the available capabilities.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond), retries=3),
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert len(steps) == 3 - int(same_response)


@pytest.mark.parametrize("workflow", ["delegate", "dynamic"])
@pytest.mark.parametrize("codemode", [False, True])
def test_mixed_deferred_subagents_keep_target_policies_and_hooks(workflow, codemode):
    steps = []
    children = []
    events = []
    shared_tool = "delegate_task" if workflow == "delegate" else "run_workflow"

    def before(event):
        events.append(("pre", event.arguments.get("subagent", event.tool_name)))
        return PreToolUseDecision.continue_()

    def after(event):
        events.append(("post", event.arguments.get("subagent", event.tool_name)))

    def respond(messages, info):
        if "Research the product." in (info.instructions or ""):
            children.append("researcher")
            return ModelResponse([TextPart("Verified evidence.")])
        if "Write the copy." in (info.instructions or ""):
            children.append("writer")
            return ModelResponse([TextPart("Copy drafted.")])
        steps.append(info)
        if len(steps) == 1:
            if codemode:
                return ModelResponse(
                    [
                        ToolCallPart(
                            "get_tool_schema", {"tool_names": [shared_tool]}, "initial-schema"
                        )
                    ]
                )
            definition = next(tool for tool in info.function_tools if tool.name == shared_tool)
            assert "researcher" not in definition.description
            if workflow == "delegate":
                assert definition.parameters_json_schema["properties"]["subagent"]["enum"] == [
                    "writer"
                ]
        if codemode and len(steps) == 2:
            assert "parameters_json_schema" in str(messages[-1])
            assert "researcher" not in str(messages[-1])
        offset = int(codemode)
        if len(steps) == 1 + offset:
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "research")}, "load"
                    )
                ]
            )
        if len(steps) == 2 + offset:
            if codemode:
                return ModelResponse(
                    [
                        ToolCallPart(
                            "get_tool_schema", {"tool_names": [shared_tool]}, "loaded-schema"
                        )
                    ]
                )
            definition = next(tool for tool in info.function_tools if tool.name == shared_tool)
            assert "researcher" in definition.description
        if len(steps) == 2 + 2 * offset:
            if workflow == "delegate":
                arguments = {"subagent": "researcher", "task": "Verify evidence."}
                if codemode:
                    arguments = {
                        "code": "await delegate_task(subagent='researcher', task='Verify evidence.')"
                    }
            else:
                arguments = {"code": "await researcher(task='Verify evidence.')"}
                if codemode:
                    arguments = {
                        "code": "await run_workflow(code=\"await researcher(task='Verify evidence.')\")"
                    }
            return ModelResponse(
                [ToolCallPart("execute_code" if codemode else shared_tool, arguments, "work")]
            )
        assert children == ["researcher"], str(messages[-1])
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n  > instructions: Research the product.\n"
            "> profile: writer:\n  > instructions: Write the copy.\n"
            "> capability: research:\n  > defer: true\n  > subagent: researcher\n"
            "  > approval: allow\n  > hooks:\n    pre_tool_use: `before`\n    post_tool_use: `after`\n"
            "> profile: host:\n  > subagent: writer\n  > use: research\n"
            f"  > workflow: {workflow}\n> use: host\n"
            "[reply] << Verify the evidence.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond), codemode=codemode),
        runtime_globals={"before": before, "after": after},
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert events == [("pre", "researcher"), ("post", "researcher")]


def nested_profile():
    images = Capability(
        name="images", identity="fixture.images", instructions="Image guidance.", defer=True
    )
    social = Capability(
        name="social",
        identity="fixture.social",
        instructions="Social guidance.",
        capabilities=(images,),
        defer=True,
    )
    return social.as_profile()


def loaded_ids(messages):
    parts = [part for message in post_compaction_window(messages) for part in message.parts]
    completed = {part.tool_call_id for part in parts if isinstance(part, LoadCapabilityReturnPart)}
    return {
        part.capability_id
        for part in parts
        if isinstance(part, LoadCapabilityCallPart) and part.tool_call_id in completed
    }


def test_parallel_nested_runs_on_shared_adapter_start_with_independent_activation():
    starts = []
    lock = Lock()

    def respond(messages, info):
        loaded = loaded_ids(messages)
        if not loaded:
            with lock:
                starts.append(info.instructions)
            assert "kedi_images_" not in info.instructions
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "social")}, "parent"
                    )
                ]
            )
        if len(loaded) == 1:
            assert "Image guidance." not in str(messages)
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "images")}, "child"
                    )
                ]
            )
        assert len(loaded) == 2
        return ModelResponse([TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond))
    profile = nested_profile()

    def run(index):
        with adapter.profile_override(profile):
            return adapter.run_sync(f"Prepare campaign {index}.").output

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, range(12))) == ["done"] * 12
    assert len(starts) == 12
    assert len(set(starts)) == 1
    assert adapter.active_profile.capabilities == ()


@pytest.mark.parametrize("compacted", [False, True])
def test_nested_resume_reset_and_reuse_preserve_native_activation_and_guidance(compacted):
    phase = "initial"
    requests = []

    def respond(messages, info):
        loaded = loaded_ids(messages)
        requests.append((phase, loaded, info.instructions))
        if phase == "reset":
            assert "Image guidance." not in info.instructions
            assert "Social guidance." not in info.instructions
        if not loaded:
            assert "kedi_images_" not in info.instructions
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "social")}, "parent"
                    )
                ]
            )
        if len(loaded) == 1:
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "images")}, "child"
                    )
                ]
            )
        if phase == "reuse":
            assert info.instructions.count("Image guidance.") == 1
            assert info.instructions.count("Social guidance.") == 1
        return ModelResponse([TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond))
    profile = nested_profile()
    with adapter.profile_override(profile):
        initial = adapter.run_sync("Prepare the campaign.")
    history = initial.all_messages()
    if compacted:
        history = [*history, ModelResponse([CompactionPart(content="Campaign prepared.")])]
    reset = profile.merge(AgentProfile(system="Replacement guidance."))
    phase = "reset"
    with adapter.profile_override(reset):
        cleared = adapter.run_sync("Continue.", message_history=history)
    phase = "reuse"
    restored = reset.merge(profile).merge(profile)
    with adapter.profile_override(restored):
        assert (
            adapter.run_sync("Continue.", message_history=cleared.all_messages()).output == "done"
        )
    assert [phase for phase, _, _ in requests].count("reset") == (3 if compacted else 1)
    assert [phase for phase, _, _ in requests].count("reuse") == 1


@pytest.mark.parametrize("direct_use", [False, True])
def test_shared_nested_child_is_reachable_through_any_loaded_parent_or_direct_use(direct_use):
    images = Capability(name="images", identity="fixture.images", defer=True)
    left = Capability(name="left", identity="fixture.left", capabilities=(images,), defer=True)
    right = Capability(name="right", identity="fixture.right", capabilities=(images,), defer=True)
    profile = left.as_profile().merge(right.as_profile())
    if direct_use:
        profile = profile.merge(images.as_profile())

    def respond(messages, info):
        loaded = loaded_ids(messages)
        if not loaded:
            assert ("kedi_images_" in info.instructions) is direct_use
            name = "images" if direct_use else "right"
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, name)}, "first"
                    )
                ]
            )
        assert info.instructions.count("- kedi_images_") == 1
        if not direct_use and len(loaded) == 1:
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "images")}, "child"
                    )
                ]
            )
        return ModelResponse([TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond))
    with adapter.profile_override(profile):
        assert adapter.run_sync("Prepare a campaign.").output == "done"


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("restored_parent_first", [False, True])
def test_shared_child_guidance_is_restored_only_through_an_active_restored_parent(
    reverse_order, restored_parent_first
):
    images = Capability(name="images", instructions="Preserve product details.", defer=True)
    left = Capability(name="left", capabilities=(images,), defer=True)
    right = Capability(name="right", capabilities=(images,), defer=True)
    first, second = (right, left) if reverse_order else (left, right)
    profile = first.as_profile().merge(second.as_profile())
    profile = profile.merge(AgentProfile(system="New campaign rules.")).merge(left.as_profile())
    steps = []
    loads = ("left", "images") if restored_parent_first else ("right", "images", "left")

    def respond(messages, info):
        step = len(steps)
        steps.append(info)
        if step < len(loads):
            assert "Preserve product details." not in info.instructions
            assert "Preserve product details." not in str(messages)
            name = loads[step]
            return ModelResponse(
                [ToolCallPart("load_capability", {"id": native_id(info.instructions, name)}, name)]
            )
        assert info.instructions.count("Preserve product details.") == 1
        return ModelResponse([TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond))
    with adapter.profile_override(profile):
        assert adapter.run_sync("Prepare a campaign.").output == "done"
    assert len(steps) == len(loads) + 1


@pytest.mark.parametrize("background", [False, True])
def test_deferred_subagent_background_wait_and_continuation_keep_owned_binding(background):
    steps = []
    children = []
    original_run = None

    def respond(messages, info):
        nonlocal original_run
        if "Research the product." in (info.instructions or ""):
            children.append(messages)
            return ModelResponse([TextPart("Evidence checked.")])
        steps.append(info)
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if len(steps) == 1:
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "research")}, "load"
                    )
                ]
            )
        if len(steps) == 2:
            return ModelResponse(
                [
                    ToolCallPart(
                        "delegate_task",
                        {
                            "subagent": "researcher",
                            "task": "Check evidence.",
                            "background": background,
                        },
                        "delegate",
                    )
                ]
            )
        if original_run is None:
            original_run = next(
                part.content["run_id"] for part in returns if part.tool_name == "delegate_task"
            )
        if background and len(steps) == 3:
            return ModelResponse([ToolCallPart("wait_subagent", {"run_id": original_run}, "wait")])
        if len(steps) == 3 + int(background):
            assert len(children) == 1
            return ModelResponse(
                [
                    ToolCallPart(
                        "continue_subagent",
                        {"run_id": original_run, "task": "Check once more."},
                        "continue",
                    )
                ]
            )
        assert len(children) == 2
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n  > instructions: Research the product.\n"
            "> capability: research:\n  > defer: true\n  > subagent: researcher\n"
            "> use: research\n[reply] << Verify evidence, then double check it.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    try:
        assert runtime.run_main() == "done"
        assert len(runtime._subagent_coordinator.runs.snapshot()) == 2
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("workflow", ["delegate", "dynamic"])
def test_guessed_inactive_target_cannot_use_an_eager_siblings_shared_control(workflow):
    children = []
    events = []
    steps = []
    name = "delegate_task" if workflow == "delegate" else "run_workflow"

    def before(event):
        events.append(event.tool_name)
        return PreToolUseDecision.continue_()

    def respond(messages, info):
        if "Research the product." in (info.instructions or ""):
            children.append("researcher")
            return ModelResponse([TextPart("Checked.")])
        if "Write the copy." in (info.instructions or ""):
            children.append("writer")
            return ModelResponse([TextPart("Drafted.")])
        steps.append(info)
        if len(steps) == 1:
            definition = next(tool for tool in info.function_tools if tool.name == name)
            assert "researcher" not in definition.description
            arguments = (
                {"subagent": "researcher", "task": "Check evidence."}
                if workflow == "delegate"
                else {"code": "await researcher(task='Check evidence.')"}
            )
            return ModelResponse([ToolCallPart(name, arguments, "guess")])
        if len(steps) == 2:
            assert children == [] and events == []
            assert "not active" in str(messages)
            if workflow == "dynamic":
                return ModelResponse(
                    [
                        ToolCallPart(
                            "run_workflow",
                            {"code": "await writer(task='Draft copy.')"},
                            "correction",
                        )
                    ]
                )
        assert "researcher" not in children and events == []
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n  > instructions: Research the product.\n"
            "> profile: writer:\n  > instructions: Write the copy.\n"
            "> capability: research:\n  > defer: true\n  > subagent: researcher\n"
            "  > hooks:\n    pre_tool_use: `before`\n"
            "> profile: host:\n  > subagent: writer\n  > use: research\n"
            f"  > workflow: {workflow}\n> use: host\n[reply] << Inspect the specialists.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond), retries=3),
        runtime_globals={"before": before},
    )
    try:
        assert runtime.run_main() == "done"
        assert all(
            run.profile_name == "writer" for run in runtime._subagent_coordinator.runs.snapshot()
        )
    finally:
        runtime.artifacts.close()


@pytest.mark.asyncio
async def test_cancelling_an_activated_scope_does_not_activate_a_sibling_task():
    from kedi.agent_profile.definitions import ProfileDefinition

    bundle = Capability(
        name="research",
        identity="research",
        defer=True,
        subagents={"researcher": ProfileDefinition(name="researcher")},
    ).as_profile()
    entered = asyncio.Event()

    async def activated():
        with capability_activation_scope(frozenset({"research"})):
            require_subagent_activation(bundle, "researcher")
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(activated())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(CapabilityActivationError):
            require_subagent_activation(bundle, "researcher")
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    with pytest.raises(CapabilityActivationError):
        require_subagent_activation(bundle, "researcher")


@pytest.mark.asyncio
async def test_cancelling_native_nested_tool_execution_cleans_activation_for_the_next_run():
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def retrieve() -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return "unreachable"

    child = Capability(name="images", identity="fixture.images", tools=(retrieve,), defer=True)
    parent = Capability(name="social", identity="fixture.social", capabilities=(child,), defer=True)
    profile = parent.as_profile()

    async def respond(messages, info):
        loaded = loaded_ids(messages)
        if len(loaded) < 2:
            name = "social" if not loaded else "images"
            return ModelResponse(
                [ToolCallPart("load_capability", {"id": native_id(info.instructions, name)}, name)]
            )
        return ModelResponse([ToolCallPart("retrieve", {}, "retrieve")])

    adapter = PydanticAdapter(FunctionModel(respond))
    tool = ToolSpec(
        name="retrieve", fn=retrieve, deferred_capability="fixture.images", risk="read_only"
    )

    async def run():
        with adapter.profile_override(profile), adapter.tool_scope((tool,)):
            await adapter.run("Retrieve product details.")
        assert adapter.active_profile.capabilities == ()

    for _ in range(2):
        entered.clear()
        cleaned.clear()
        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(entered.wait(), 3)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        assert cleaned.is_set()
    assert adapter.active_profile.capabilities == ()
    assert adapter.scoped_tool_specs == ()


def test_three_deferred_levels_reveal_only_the_next_reachable_catalog_entry():
    images = Capability(name="images", identity="fixture.images", defer=True)
    campaign = Capability(
        name="campaign", identity="fixture.campaign", capabilities=(images,), defer=True
    )
    social = Capability(
        name="social", identity="fixture.social", capabilities=(campaign,), defer=True
    )
    names = ("social", "campaign", "images")

    def respond(messages, info):
        completed = len(loaded_ids(messages))
        for index, name in enumerate(names):
            assert (f"kedi_{name}_" in info.instructions) is (index <= completed)
        if completed == len(names):
            return ModelResponse([TextPart("done")])
        return ModelResponse(
            [
                ToolCallPart(
                    "load_capability",
                    {"id": native_id(info.instructions, names[completed])},
                    f"load-{completed}",
                )
            ]
        )

    adapter = PydanticAdapter(FunctionModel(respond))
    with adapter.profile_override(social.as_profile()):
        assert adapter.run_sync("Prepare a campaign.").output == "done"


def test_deferred_parent_activates_its_eager_child_tools_and_guidance_together():
    calls = []
    executed = []

    def generate_image() -> str:
        executed.append("image")
        return "product.png"

    def respond(messages, info):
        calls.append(info)
        if len(calls) == 1:
            assert "generate_image" not in {tool.name for tool in info.function_tools}
            assert "Preserve product details." not in info.instructions
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "social")}, "load"
                    )
                ]
            )
        assert "kedi_images_" not in info.instructions
        assert "Preserve product details." in str(messages)
        if len(calls) == 2:
            return ModelResponse([ToolCallPart("generate_image", {}, "image")])
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> capability: images:\n  > instructions: Preserve product details.\n"
            "  > approval: allow\n  > use: generate_image\n"
            "> capability: social:\n  > defer: true\n  > use: images\n"
            "> use: social\n[reply] << Prepare a product image.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond)),
        runtime_globals={"generate_image": generate_image},
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert executed == ["image"]


@pytest.mark.parametrize("deferred", [False, True])
def test_dynamic_workflow_accepts_subagents_contributed_only_by_a_capability(deferred):
    steps = []
    children = []

    def respond(messages, info):
        if "Research the product." in (info.instructions or ""):
            children.append("researcher")
            return ModelResponse([TextPart("Verified.")])
        steps.append(info)
        names = {tool.name for tool in info.function_tools}
        if deferred and len(steps) == 1:
            assert "run_workflow" not in names
            return ModelResponse(
                [
                    ToolCallPart(
                        "load_capability", {"id": native_id(info.instructions, "research")}, "load"
                    )
                ]
            )
        if len(steps) == 1 + int(deferred):
            assert "run_workflow" in names
            return ModelResponse(
                [
                    ToolCallPart(
                        "run_workflow",
                        {"code": "await researcher(task='Verify evidence.')"},
                        "workflow",
                    )
                ]
            )
        assert children == ["researcher"]
        return ModelResponse([TextPart("done")])

    source = (
        "> profile: researcher:\n  > instructions: Research the product.\n"
        "> capability: research:\n"
        + ("  > defer: true\n" if deferred else "")
        + "  > subagent: researcher\n"
        "> profile: manager:\n  > use: research\n  > workflow: dynamic\n"
        "> use: manager\n[reply] << Verify evidence through the specialist.\n= <reply>\n"
    )
    runtime = compile_program(
        parse_program(source), adapter=PydanticAdapter(FunctionModel(respond))
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()


def test_dynamic_workflow_rejects_a_tool_only_use_after_resolution_before_inference():
    def respond(messages, info):
        pytest.fail("Invalid workflow reached the model")

    source = (
        "> profile: manager:\n  > use: read_document\n  > workflow: dynamic\n"
        "> use: manager\n[reply] << Audit the document.\n= <reply>\n"
    )
    from kedi.errors import KediExecutionError

    with pytest.raises(KediExecutionError, match="dynamic workflow requires"):
        compile_program(
            parse_program(source),
            adapter=PydanticAdapter(FunctionModel(respond)),
            runtime_globals={"read_document": lambda: "document"},
        ).run_main()
