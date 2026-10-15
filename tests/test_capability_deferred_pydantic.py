from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from pydantic_ai.messages import (
    CompactionPart,
    LoadCapabilityReturnPart,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    post_compaction_window,
)
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.hooks import PreToolUseDecision
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.capabilities import AppliedCapability, CapabilityToolPolicy
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


class DeferredPydanticAdapter(PydanticAdapter):
    capabilities = replace(
        PydanticAdapter.capabilities,
        supports_capabilities=True,
        supports_deferred_tools=True,
    )


def compile_reader(adapter, *, reset=False, tool=True, **globals):
    return compile_program(
        parse_program(
            "> approval: deny\n"
            "> capability: reader:\n"
            "  ###\n  Retrieve document evidence.\n  ###\n"
            "  > defer: true\n"
            "  > instructions: Use the document's exact revision.\n"
            "  > approval: allow\n"
            "  > hooks:\n"
            "    pre_tool_use: `observe_pre`\n"
            "    post_tool_use: `observe_post`\n"
            + ("  > use: read_document\n" if tool else "")
            + "> instructions: Host instructions.\n"
            "> use: reader\n> use: reader\n"
            + ("> instructions: Replacement instructions.\n" if reset else "")
            + "[reply] << Review the document.\n= <reply>\n"
        ),
        adapter=adapter,
        runtime_globals=globals,
    )


@pytest.mark.parametrize("guess_first", [False, True])
@pytest.mark.parametrize("duplicate_load", [False, True])
def test_native_loader_gates_execution_and_preserves_owned_policy(guess_first, duplicate_load):
    executed = []
    hooks = []
    requests = []

    def read_document() -> str:
        executed.append("read")
        return "Revision 7"

    def observe_pre(event):
        hooks.append(("pre", event.tool_name))
        return PreToolUseDecision.continue_()

    def observe_post(event):
        hooks.append(("post", event.tool_name))

    def respond(messages, info):
        names = {tool.name for tool in info.function_tools}
        requests.append((names, info.instructions or ""))
        step = len(requests) - 1
        if guess_first and step == 0:
            assert "read_document" not in names
            return ModelResponse([ToolCallPart("read_document", {}, "guess")])
        if step == int(guess_first):
            assert executed == [] and hooks == []
            assert "read_document" not in names
            assert "load_capability" in names
            assert "exact revision" not in (info.instructions or "")
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            calls = [ToolCallPart("load_capability", {"id": identity}, "load")]
            if duplicate_load:
                calls.append(ToolCallPart("load_capability", {"id": identity}, "duplicate"))
            return ModelResponse(calls)
        if step == int(guess_first) + 1:
            assert "read_document" in names
            assert executed == [] and hooks == []
            assert "exact revision" in str(messages)
            return ModelResponse([ToolCallPart("read_document", {}, "read")])
        assert executed == ["read"]
        return ModelResponse([TextPart("done")])

    adapter = DeferredPydanticAdapter(FunctionModel(respond), retries=3)
    runtime = compile_reader(
        adapter,
        read_document=read_document,
        observe_pre=observe_pre,
        observe_post=observe_post,
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert hooks == [("pre", "read_document"), ("post", "read_document")]
    assert adapter.scoped_tool_specs == ()
    assert adapter.scoped_tools == ()
    assert len(requests) == 3 + int(guess_first)


@pytest.mark.parametrize("tool", [False, True])
def test_unused_deferred_capability_does_not_call_tools_or_deliver_guidance(tool):
    def unexpected(*args):
        pytest.fail("Unused capability executed a tool or hook")

    def respond(messages, info):
        names = {tool.name for tool in info.function_tools}
        assert names & {"load_capability", "read_document"} == {"load_capability"}
        assert "Retrieve document evidence" in info.instructions
        assert "exact revision" not in info.instructions
        return ModelResponse([TextPart("no document needed")])

    runtime = compile_reader(
        DeferredPydanticAdapter(FunctionModel(respond)),
        tool=tool,
        read_document=unexpected,
        observe_pre=unexpected,
        observe_post=unexpected,
    )
    try:
        assert runtime.run_main() == "no document needed"
    finally:
        runtime.artifacts.close()


def test_defer_expression_is_not_repeated_by_native_loader_retry_or_tool_steps():
    evaluations = []
    requests = []
    executed = []

    def choose_mode():
        evaluations.append(True)
        return True

    def read_document() -> str:
        executed.append("read")
        return "Revision 7"

    def respond(messages, info):
        requests.append(messages)
        assert evaluations == [True]
        if len(requests) <= 2:
            identity = (
                "nonexistent_capability"
                if len(requests) == 1
                else re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            )
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        if len(requests) == 3:
            assert "read_document" in {tool.name for tool in info.function_tools}
            return ModelResponse([ToolCallPart("read_document", {}, "read")])
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> capability: reader:\n  > defer: `choose_mode()`\n"
            "  > approval: allow\n  > use: read_document\n"
            "> use: reader\n"
            "@review() -> str:\n  [reply] << Review the document.\n  = <reply>\n"
            "= <review()>\n"
        ),
        adapter=DeferredPydanticAdapter(FunctionModel(respond), retries=3),
        runtime_globals={"choose_mode": choose_mode, "read_document": read_document},
    )
    try:
        assert evaluations == []
        assert runtime.run_main() == "done"
        assert len(requests) == 4
        assert evaluations == [True]
        assert executed == ["read"]
    finally:
        runtime.artifacts.close()


def test_parallel_native_runs_do_not_share_activation_on_one_adapter():
    def respond(messages, info):
        names = {tool.name for tool in info.function_tools}
        if len(messages) == 1:
            assert "read_document" not in names
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        assert "read_document" in names
        return ModelResponse([TextPart("loaded")])

    adapter = DeferredPydanticAdapter(FunctionModel(respond))

    def run(index):
        def read_document() -> str:
            return str(index)

        runtime = compile_reader(
            adapter,
            read_document=read_document,
            observe_pre=lambda event: PreToolUseDecision.continue_(),
            observe_post=lambda event: None,
        )
        try:
            return runtime.run_main()
        finally:
            runtime.artifacts.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, range(12))) == ["loaded"] * 12
    assert adapter.scoped_tool_specs == ()


def test_reset_suppresses_loaded_guidance_without_removing_owned_tools():
    calls = []
    requests = []

    def read_document() -> str:
        calls.append("read")
        return "Revision 7"

    def respond(messages, info):
        requests.append(info)
        assert "Replacement instructions." in info.instructions
        assert "exact revision" not in info.instructions
        assert "exact revision" not in str(messages)
        if len(requests) == 1:
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        if len(requests) == 2:
            assert "read_document" in {tool.name for tool in info.function_tools}
            return ModelResponse([ToolCallPart("read_document", {}, "read")])
        return ModelResponse([TextPart("done")])

    runtime = compile_reader(
        DeferredPydanticAdapter(FunctionModel(respond)),
        reset=True,
        read_document=read_document,
        observe_pre=lambda event: PreToolUseDecision.continue_(),
        observe_post=lambda event: None,
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert calls == ["read"]


@pytest.mark.parametrize("trim_history", [False, True])
def test_native_history_resume_reset_and_explicit_guidance_restoration(trim_history):
    capability = AppliedCapability(
        identity="fixture::reader",
        name="reader",
        description="Retrieve evidence",
        profile=AgentProfile(system="Use the exact document revision."),
        deferred=True,
    )
    applied = AgentProfile(capabilities=(capability,))
    initial = AgentProfile(system="Base.").merge(applied)
    reset = initial.merge(AgentProfile(system="Replacement."))
    restored = reset.merge(applied).merge(applied)
    reset_again = restored.merge(AgentProfile(system="Final instructions."))
    phases = iter(["initial", "loaded", "reset", "restored", "final"])
    observed = []

    def respond(messages, info):
        phase = next(phases)
        observed.append(phase)
        names = {tool.name for tool in info.function_tools}
        text = info.instructions or ""
        if phase == "initial":
            assert "read_document" not in names
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", text).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        assert "read_document" in names
        assert "exact document revision" in str(messages)
        if phase == "restored":
            assert text.count("Use the exact document revision.") == 1
        else:
            assert "exact document revision" not in text
        return ModelResponse([TextPart(phase)])

    adapter = DeferredPydanticAdapter(
        FunctionModel(respond),
        history_processor=(lambda ctx: ctx.keep_recent_groups(1)) if trim_history else None,
    )
    tool = ToolSpec(
        name="read_document",
        fn=lambda: "evidence",
        deferred_capability=capability.identity,
        capability_policy=CapabilityToolPolicy(
            owners=(capability.identity,), approval=ApprovalPolicy.allow()
        ),
    )
    history = []
    for profile in (initial, reset, restored, reset_again):
        with adapter.profile_override(profile), adapter.tool_scope((tool,)):
            result = adapter.run_sync("Continue.", message_history=history)
        history = result.all_messages()
    assert observed == ["initial", "loaded", "reset", "restored", "final"]


def test_restored_deferred_guidance_still_waits_for_native_load():
    capability = AppliedCapability(
        identity="fixture::reader",
        name="reader",
        description="Retrieve evidence",
        profile=AgentProfile(system="Use exact revisions."),
        deferred=True,
    )
    applied = AgentProfile(capabilities=(capability,))
    profile = applied.merge(AgentProfile(system="Base.")).merge(applied)
    requests = []

    def respond(messages, info):
        requests.append(info)
        if len(requests) == 1:
            assert "Use exact revisions." not in info.instructions
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        assert info.instructions.count("Use exact revisions.") == 1
        returned = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert "Use exact revisions." not in str(returned)
        return ModelResponse([TextPart("done")])

    adapter = DeferredPydanticAdapter(FunctionModel(respond))
    with adapter.profile_override(profile):
        assert adapter.run_sync("Continue.").output == "done"


@pytest.mark.parametrize("restore", [False, True])
def test_native_compaction_requires_reload_and_respects_instruction_reset(restore):
    capability = AppliedCapability(
        identity="fixture::reader",
        name="reader",
        description="Retrieve evidence",
        profile=AgentProfile(system="Use exact revisions."),
        deferred=True,
    )
    applied = AgentProfile(capabilities=(capability,))
    reset = applied.merge(AgentProfile(system="Replacement instructions."))
    profile = reset.merge(applied) if restore else reset
    phases = iter(["initial", "loaded", "compacted", "reloaded"])
    observed = []

    def respond(messages, info):
        phase = next(phases)
        observed.append(phase)
        names = {tool.name for tool in info.function_tools}
        instructions = info.instructions or ""
        if phase in {"initial", "compacted"}:
            assert "read_document" not in names
            assert "Use exact revisions." not in instructions
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, phase)])
        assert "read_document" in names
        if phase == "reloaded":
            assert "Replacement instructions." in instructions
            assert instructions.count("Use exact revisions.") == int(restore)
            current_returns = [
                part
                for message in post_compaction_window(messages)
                for part in message.parts
                if isinstance(part, LoadCapabilityReturnPart)
            ]
            assert len(current_returns) == 1
            assert "Use exact revisions." not in str(current_returns)
        return ModelResponse([TextPart(phase)])

    adapter = DeferredPydanticAdapter(FunctionModel(respond))
    tool = ToolSpec(
        name="read_document",
        fn=lambda: "evidence",
        deferred_capability=capability.identity,
    )
    with adapter.profile_override(applied), adapter.tool_scope((tool,)):
        result = adapter.run_sync("Read the document.")
    history = [*result.all_messages(), ModelResponse([CompactionPart(content="Document read.")])]
    with adapter.profile_override(profile), adapter.tool_scope((tool,)):
        assert adapter.run_sync("Continue.", message_history=history).output == "reloaded"
    assert observed == ["initial", "loaded", "compacted", "reloaded"]


def test_orphan_deferred_tool_is_rejected_before_inference():
    def unexpected(messages, info):
        pytest.fail("Invalid tool registration reached model inference")

    adapter = DeferredPydanticAdapter(FunctionModel(unexpected))
    tool = ToolSpec(name="hidden", fn=lambda: "done", deferred_capability="missing")
    with adapter.tool_scope((tool,)), pytest.raises(KediExecutionError, match="unregistered"):
        adapter.run_sync("Continue.")


@pytest.mark.asyncio
async def test_cancellation_unwinds_loaded_tools_and_does_not_activate_the_next_run():
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    capability = AppliedCapability(
        identity="fixture::reader",
        name="reader",
        description="Retrieve evidence",
        profile=AgentProfile(),
        deferred=True,
    )
    profile = AgentProfile(capabilities=(capability,))

    async def read_document() -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return "unreachable"

    async def respond(messages, info):
        names = {tool.name for tool in info.function_tools}
        if len(messages) == 1:
            assert "read_document" not in names
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        assert "read_document" in names
        return ModelResponse([ToolCallPart("read_document", {}, "read")])

    adapter = DeferredPydanticAdapter(FunctionModel(respond))
    tool = ToolSpec(
        name="read_document",
        fn=read_document,
        deferred_capability=capability.identity,
        capability_policy=CapabilityToolPolicy(
            owners=(capability.identity,), approval=ApprovalPolicy.allow()
        ),
    )

    async def run():
        try:
            with adapter.profile_override(profile), adapter.tool_scope((tool,)):
                await adapter.run("Continue.")
        finally:
            assert adapter.scoped_tool_specs == ()
            assert adapter.active_profile.capabilities == ()

    for _ in range(2):
        entered.clear()
        cleaned.clear()
        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        assert cleaned.is_set()
