from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi import (
    AgentSurface,
    AgentRunStateEvent,
    ConversationState,
    KediRuntime,
    SubagentUsageLimits,
    current_subagent_execution,
    observe_agent_events,
)
from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.tool_override import ToolSpec
from kedi.agent_profile import (
    AgentProfile,
    AgentProfileOverrideUnsupported,
    AgentToolOverrideUnsupported,
    ProfileDefinition,
    ProfileSystemMember,
    SkillsSettings,
)
from kedi.artifacts import ArtifactPolicy


class _ProfileOnlyAdapter:
    kind = "agent-framework"
    shortname = "profile-only"
    capabilities = AdapterCapabilities(supports_profile_override=True)

    def default_model(self) -> str | None:
        return None

    @contextmanager
    def profile_override(self, _profile: AgentProfile) -> Iterator[None]:
        yield


def test_subagent_usage_limits_are_part_of_the_public_embedding_surface() -> None:
    limits = SubagentUsageLimits()

    assert limits.request_limit == 8
    assert limits.tool_calls_limit == 16


def test_bound_surface_replays_native_history_and_applies_profile() -> None:
    observed: list[list[ModelMessage]] = []
    instructions: list[str | None] = []

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        observed.append(list(messages))
        request = next(message for message in messages if isinstance(message, ModelRequest))
        instructions.append(request.instructions)
        return ModelResponse(parts=[TextPart(f"response-{len(observed)}")])

    adapter = PydanticAdapter(FunctionModel(respond))
    conversation = ConversationState()
    runtime = KediRuntime(adapter=adapter, conversation_state=conversation)
    surface = runtime.bind_agent(
        profile=AgentProfile(
            history=True,
            system="Keep the answer exact.",
            artifacts=ArtifactPolicy(enabled=False),
        )
    )

    with surface.run_scope():
        assert adapter.run_sync("first prompt").output == "response-1"
    with surface.run_scope():
        assert adapter.run_sync("second prompt").output == "response-2"

    assert instructions == ["Keep the answer exact.", "Keep the answer exact."]
    assert len(observed[1]) > len(observed[0])
    assert conversation.adapter_state_snapshot(adapter.shortname).state["pydantic_messages"]
    surface.close()
    conversation.close()


def test_bound_surface_uses_canonical_tool_and_instruction_assembly(tmp_path: Path) -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    def lookup() -> str:
        return "found"

    adapter = PydanticAdapter(FunctionModel(respond))
    profile = AgentProfile.empty().merge(
        AgentProfile(
            system="Base instructions.",
            tools=("lookup",),
            tool_bindings={
                "lookup": ToolSpec(fn=lookup, name="lookup", risk="read_only"),
            },
            skills=SkillsSettings(
                enabled=True,
                cwd=tmp_path,
                include_registry=False,
            ),
        )
    )
    surface = KediRuntime(adapter=adapter).bind_agent(profile=profile)
    names = {tool.name for tool in surface.tool_specs}

    assert {"lookup", "list_skills", "read_skill"} <= names
    assert {"read_artifact", "search_artifacts", "run_artifact_code"} <= names
    assert surface.profile.system is not None
    assert surface.profile.system.startswith("Base instructions.")
    assert "list_skills" in surface.profile.system
    assert "artifact" in surface.profile.system.lower()
    with surface.run_scope():
        assert {tool.name for tool in adapter.scoped_tool_specs} == names
        assert adapter.active_profile == surface.profile
    assert adapter.scoped_tool_specs == ()
    surface.close()


def test_bound_surface_lifecycle_is_atomic_and_explicitly_owned() -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    adapter = PydanticAdapter(FunctionModel(respond))
    conversation = ConversationState()
    surface = KediRuntime(
        adapter=adapter,
        conversation_state=conversation,
    ).bind_agent(
        profile=AgentProfile(
            history=True,
            artifacts=ArtifactPolicy(enabled=False),
        ),
        close_conversation_state=True,
    )

    with surface.run_scope():
        with pytest.raises(RuntimeError, match="already running"):
            with surface.run_scope():
                pass
        with pytest.raises(RuntimeError, match="while it is running"):
            surface.close()

    surface.close()
    surface.close()
    with pytest.raises(RuntimeError, match="closed"):
        with surface.run_scope():
            pass
    with pytest.raises(RuntimeError, match="closed"):
        conversation.adapter_state_snapshot(adapter.shortname)


def test_agent_surface_is_public() -> None:
    assert AgentSurface.__module__ == "kedi.agent_surface"


def test_agent_surface_rejects_missing_binding_capabilities() -> None:
    profile = AgentProfile.empty()
    with pytest.raises(AgentProfileOverrideUnsupported, match="profile override"):
        AgentSurface(
            agent=cast(Any, object()),
            profile=profile,
            tool_specs=(),
            conversation_state=None,
        )

    profile_only = cast(Any, _ProfileOnlyAdapter())
    with pytest.raises(AgentToolOverrideUnsupported, match="tool registration"):
        AgentSurface(
            agent=profile_only,
            profile=profile,
            tool_specs=(ToolSpec(lambda: None, "tool"),),
            conversation_state=None,
        )

    conversation = ConversationState()
    with pytest.raises(AgentProfileOverrideUnsupported, match="conversation state"):
        AgentSurface(
            agent=profile_only,
            profile=profile,
            tool_specs=(),
            conversation_state=conversation,
        )
    conversation.close()


def test_agent_surface_sync_and_async_context_lifecycle_without_optional_bindings() -> None:
    profile = AgentProfile.empty()
    sync_surface = AgentSurface(
        agent=cast(Any, _ProfileOnlyAdapter()),
        profile=profile,
        tool_specs=(),
        conversation_state=None,
    )
    with sync_surface as entered:
        assert entered is sync_surface
        with sync_surface.run_scope():
            pass
    with pytest.raises(RuntimeError, match="closed"):
        sync_surface.__enter__()

    async def exercise() -> None:
        async_surface = AgentSurface(
            agent=cast(Any, _ProfileOnlyAdapter()),
            profile=profile,
            tool_specs=(),
            conversation_state=None,
        )
        async with async_surface as entered:
            assert entered is async_surface
        await async_surface.aclose()

    asyncio.run(exercise())


def test_runtime_aclose_cancels_background_subagents() -> None:
    async def exercise() -> None:
        child_started = asyncio.Event()

        async def wait_forever(
            _messages: list[ModelMessage],
            _info: AgentInfo,
        ) -> ModelResponse:
            child_started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled child must not return")

        adapter = PydanticAdapter(FunctionModel(wait_forever))
        worker = ProfileDefinition(
            name="worker",
            members=[ProfileSystemMember(plain_value="Inspect the delegated task.")],
        )
        profile = AgentProfile(
            name="main",
            subagents=("worker",),
            subagent_bindings={"worker": worker},
            artifacts=ArtifactPolicy(enabled=False),
        )
        runtime = KediRuntime(adapter=adapter)
        surface = runtime.bind_agent(profile=profile)
        delegate = next(tool for tool in surface.tool_specs if tool.name == "delegate_task")

        handle = await delegate.fn(
            subagent="worker",
            task="Inspect the repository.",
            background=True,
        )
        await child_started.wait()
        await runtime.aclose()

        assert runtime._subagent_status(handle["run_id"]) == "cancelled"
        assert runtime._subagent_coordinator.active_runs == 0
        with pytest.raises(RuntimeError, match="closed"):
            runtime.bind_agent(profile=profile)
        surface.close()

    asyncio.run(exercise())


def test_runtime_aclose_releases_implicit_and_borrowed_conversation_paths() -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    async def exercise() -> None:
        implicit = KediRuntime(adapter=PydanticAdapter(FunctionModel(respond)))
        implicit.bind_agent(
            profile=AgentProfile(history=True, artifacts=ArtifactPolicy(enabled=False))
        )
        assert implicit._implicit_conversation_state is not None
        await implicit.aclose()

        borrowed = ConversationState()
        explicit = KediRuntime(
            adapter=PydanticAdapter(FunctionModel(respond)),
            conversation_state=borrowed,
        )
        await explicit.aclose()
        borrowed.close()

    asyncio.run(exercise())


def test_subagent_execution_context_is_public_and_task_scoped() -> None:
    observed_profiles: list[str | None] = []
    parent_calls = 0

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal parent_calls
        context = current_subagent_execution()
        observed_profiles.append(None if context is None else context.profile.name)
        if context is not None:
            assert context.run_id
            assert context.conversation_id
            return ModelResponse(parts=[TextPart("child complete")])
        parent_calls += 1
        if parent_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {"subagent": "worker", "task": "Inspect the repository."},
                        tool_call_id="delegate-context-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("parent complete")])

    adapter = PydanticAdapter(FunctionModel(respond))
    profile = AgentProfile(
        name="main",
        subagents=("worker",),
        subagent_bindings={
            "worker": ProfileDefinition(
                name="worker",
                members=[ProfileSystemMember(plain_value="Inspect the delegated task.")],
            )
        },
        artifacts=ArtifactPolicy(enabled=False),
    )
    surface = KediRuntime(adapter=adapter).bind_agent(profile=profile)

    with surface.run_scope():
        assert adapter.run_sync("Delegate once.").output == "parent complete"

    assert observed_profiles == [None, "worker", None]
    assert current_subagent_execution() is None
    surface.close()


def test_native_parent_run_links_child_semantic_events() -> None:
    events: list[object] = []
    parent_calls = 0

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal parent_calls
        if current_subagent_execution() is not None:
            return ModelResponse(parts=[TextPart("child complete")])
        parent_calls += 1
        if parent_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {"subagent": "worker", "task": "Inspect the repository."},
                        tool_call_id="delegate-event-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("parent complete")])

    adapter = PydanticAdapter(FunctionModel(respond))
    profile = AgentProfile(
        name="main",
        subagents=("worker",),
        subagent_bindings={
            "worker": ProfileDefinition(
                name="worker",
                members=[ProfileSystemMember(plain_value="Inspect the delegated task.")],
            )
        },
        artifacts=ArtifactPolicy(enabled=False),
    )
    surface = KediRuntime(adapter=adapter).bind_agent(profile=profile)

    with observe_agent_events(events.append), surface.run_scope():
        assert adapter.run_sync("Delegate once.").output == "parent complete"

    run_events = [event for event in events if isinstance(event, AgentRunStateEvent)]
    parent_started = next(
        event for event in run_events if event.parent_run_id is None and event.state == "started"
    )
    child_started = next(
        event for event in run_events if event.agent_name == "worker" and event.state == "started"
    )
    assert child_started.parent_run_id == parent_started.run_id
    surface.close()
