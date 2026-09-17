from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from pydantic_ai import ModelRetry, UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.adapters.langchain import _langchain_tool_from_spec
from kedi.agent_adapter.adapters.pydantic import _model_retry_for_tool_error
from kedi.agent_adapter.compaction import HistoryArchiveSettings, LangChainArtifactHistoryProcessor
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.invocation import observe_invocations
from kedi.agent_adapter.tool_override import ToolSpec, artifact_admission_scope
from kedi.artifacts import ArtifactManager, ArtifactPolicy
from kedi.agent_profile import AgentProfile


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    settings_seen: list[dict[str, Any]] = Field(default_factory=list)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "synthetic-parity"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        self.settings_seen.append(kwargs)
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        self.settings_seen.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def response(index: int, *, tool: bool = True) -> AIMessage:
    return AIMessage(
        content="" if tool else "done",
        tool_calls=[{"name": "terminal", "args": {"step": index}, "id": f"call_{index}"}]
        if tool
        else [],
        usage_metadata={
            "input_tokens": 10_000 + index * 1200,
            "output_tokens": 10,
            "total_tokens": 10_010 + index * 1200,
            "input_token_details": {"cache_read": 9000},
        },
    )


def tool_spec(fn: Any, *, kind: str = "terminal") -> ToolSpec:
    return ToolSpec(
        fn=fn,
        name="terminal",
        description="Synthetic terminal",
        json_schema={
            "type": "object",
            "properties": {"step": {"type": "integer"}},
            "required": ["step"],
        },
        risk="read_only",
        metadata={"kedi_tool_kind": kind},
        artifact_result="never",
    )


@pytest.mark.parametrize(
    "kind,error",
    [
        ("filesystem", FileNotFoundError("missing")),
        ("filesystem", ValueError("bad patch")),
        ("terminal", ValueError("invalid process ID")),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_correctable_tool_errors_have_identical_messages(
    kind: str, error: Exception, asynchronous: bool
) -> None:
    def fail(step: int) -> str:
        raise error

    async def afail(step: int) -> str:
        raise error

    spec = tool_spec(afail if asynchronous else fail, kind=kind)
    native = _langchain_tool_from_spec(spec)
    expected = _model_retry_for_tool_error(spec, error)
    assert isinstance(expected, ModelRetry)
    result = asyncio.run(
        native.ainvoke({"type": "tool_call", "name": spec.name, "args": {"step": 1}, "id": "x"})
    )
    assert result.status == "error"
    assert result.content == str(expected)


def test_unknown_tool_errors_remain_fatal() -> None:
    def fail(step: int) -> str:
        raise RuntimeError("bug")

    spec = tool_spec(fail, kind="application")
    assert _model_retry_for_tool_error(spec, RuntimeError("bug")) is None
    with pytest.raises(RuntimeError, match="bug"):
        asyncio.run(_langchain_tool_from_spec(spec).ainvoke({"step": 1}))


def test_native_loop_corrects_tool_error_and_bounds_repeated_failures() -> None:
    calls: list[int] = []

    def terminal(step: int) -> str:
        calls.append(step)
        if step == 0:
            raise ValueError("invalid process ID")
        return "ok"

    model = ScriptedModel(responses=[response(0), response(1), response(2, tool=False)])
    adapter = LangChainAdapter(model)
    with adapter.tool_scope([tool_spec(terminal)]):
        assert adapter.invoke_sync(prompt="perform work") == "done"
    assert calls == [0, 1]
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in model.seen[1])

    model = ScriptedModel(responses=[response(0) for _ in range(4)])
    adapter = LangChainAdapter(model, tool_retries=1)
    with (
        adapter.tool_scope([tool_spec(terminal)]),
        pytest.raises(UnexpectedModelBehavior, match="max retries"),
    ):
        adapter.invoke_sync(prompt="perform work")
    assert len(model.seen) == 2


def test_request_limits_override_default_and_are_not_graph_input() -> None:
    model = ScriptedModel(responses=[response(0), response(1, tool=False)])
    adapter = LangChainAdapter(model, default_usage_limits=UsageLimits(request_limit=1))
    usages = []
    with adapter.tool_scope([tool_spec(lambda step: "ok")]), observe_invocations(usages.append):
        with pytest.raises(UsageLimitExceeded, match="request_limit"):
            adapter.invoke_sync(prompt="perform work")
    assert len(model.seen) == 1
    assert usages[0].input_tokens == 10_000
    assert usages[0].cache_read_tokens == 9000
    assert adapter.invoke_sync(prompt="finish", usage_limits=UsageLimits(request_limit=2)) == "done"


@pytest.mark.parametrize(
    "limits",
    [
        UsageLimits(input_tokens_limit=1),
        UsageLimits(output_tokens_limit=1),
        UsageLimits(total_tokens_limit=1),
        UsageLimits(per_request_input_tokens_limit=1),
    ],
)
def test_native_token_limits(limits: UsageLimits) -> None:
    adapter = LangChainAdapter(
        ScriptedModel(responses=[response(0, tool=False)]), default_usage_limits=limits
    )
    with pytest.raises(UsageLimitExceeded):
        adapter.invoke_sync(prompt="work")


def test_parallel_tool_limit_reserves_before_any_excess_side_effect() -> None:
    called = []

    async def terminal(step: int) -> str:
        called.append(step)
        await asyncio.sleep(0.01)
        return "ok"

    first = response(0)
    first.tool_calls += response(1).tool_calls
    model = ScriptedModel(responses=[first])
    adapter = LangChainAdapter(model, default_usage_limits=UsageLimits(tool_calls_limit=1))
    with (
        adapter.tool_scope([tool_spec(terminal)]),
        pytest.raises(UsageLimitExceeded, match="tool_calls_limit"),
    ):
        adapter.invoke_sync(prompt="work")
    assert len(called) <= 1


def test_archive_persists_native_history_and_does_not_lose_usage(tmp_path: Path) -> None:
    responses = [response(i) for i in range(6)] + [response(6, tool=False)]
    expected_input = sum(m.usage_metadata["input_tokens"] for m in responses if m.usage_metadata)
    model = ScriptedModel(responses=responses, model_kwargs={"prompt_cache_key": "stable"})
    settings = HistoryArchiveSettings(
        threshold_tokens=2000, preserve_recent_tokens=500, minimum_reduction_tokens=500
    )
    adapter = LangChainAdapter(model, history_archive=settings)
    manager = ArtifactManager(session_id="lc_archive")
    policy = ArtifactPolicy.patch(
        {"enabled": True, "store": "file", "path": tmp_path, "threshold_bytes": 100_000}
    )
    conversation = AdapterConversationContext()
    usages = []
    with (
        adapter.tool_scope([tool_spec(lambda step: "x" * 4000)]),
        artifact_admission_scope(manager=manager, policy=policy),
        adapter.conversation_scope(conversation),
        observe_invocations(usages.append),
    ):
        assert adapter.invoke_sync(prompt="finish the task") == "done"
    checkpoints = [
        m for m in model.seen[-1] if m.additional_kwargs.get("kedi_compaction_checkpoint")
    ]
    assert checkpoints
    assert usages[0].requests == 7
    assert usages[0].input_tokens == expected_input
    assert usages[0].cache_read_tokens == 63_000
    assert any("_h" in s.get("prompt_cache_key", "") for s in model.settings_seen)
    assert len(model.seen[-1]) < 13
    assert model.seen[-1][0].content == "finish the task"
    assert any(
        m["data"]["additional_kwargs"].get("kedi_compaction_checkpoint")
        for m in conversation.result_state["langchain_messages"]
    )


def test_archive_no_storage_and_unresolved_lifecycle_are_noops() -> None:
    settings = HistoryArchiveSettings(
        threshold_tokens=100, preserve_recent_tokens=10, minimum_reduction_tokens=10
    )
    processor = LangChainArtifactHistoryProcessor(settings)
    messages: list[BaseMessage] = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "terminal", "args": {}, "id": "open"}]),
        ToolMessage(content="x" * 1000, tool_call_id="other"),
    ]
    assert processor(messages) is messages


def test_archive_settings_are_shared_with_pydantic() -> None:
    from pydantic_ai.models.test import TestModel
    from kedi.agent_adapter.compaction import PydanticHistoryArchiveSettings

    assert PydanticHistoryArchiveSettings is HistoryArchiveSettings
    PydanticAdapter(TestModel(), history_archive=HistoryArchiveSettings())


def test_archive_exact_payload_retained_prefix_tail_and_quota_failure(tmp_path: Path) -> None:
    import json
    from langchain_core.messages import messages_to_dict

    settings = HistoryArchiveSettings(
        threshold_tokens=2000, preserve_recent_tokens=500, minimum_reduction_tokens=500
    )
    processor = LangChainArtifactHistoryProcessor(settings)
    messages: list[BaseMessage] = [HumanMessage(content="Keep the user's rules")]
    for i in range(6):
        messages.extend(
            [
                response(i),
                ToolMessage(content=f"evidence-{i}:" + "x" * 4000, tool_call_id=f"call_{i}"),
            ]
        )
    before = messages_to_dict(messages)
    manager = ArtifactManager(session_id="exact-lc")
    policy = ArtifactPolicy.patch({"enabled": True, "store": "file", "path": tmp_path})
    with artifact_admission_scope(manager=manager, policy=policy):
        updated = processor(messages)
    assert updated[0] is messages[0]
    assert updated[-1] is messages[-1]
    assert updated[-2] is messages[-2]
    checkpoint = json.loads(str(updated[1].content).splitlines()[1])
    archive = manager.get_value(checkpoint["artifact_refs"][0])
    assert archive["schema"] == "kedi.langchain-history-archive.v1"
    removed = len(messages) - len(updated) + 1
    assert archive["messages"] == before[1 : removed + 1]
    assert messages_to_dict(messages) == before
    assert updated[1].additional_kwargs["kedi_history_provider_input_baseline_tokens"] == 10_000
    sealed = messages_to_dict([updated[1]])
    with artifact_admission_scope(manager=manager, policy=policy):
        again = processor(updated)
    assert messages_to_dict([again[1]]) == sealed
    assert again[1] is updated[1]

    tiny = policy.overlay(ArtifactPolicy.patch({"session_quota_bytes": 1}))
    with artifact_admission_scope(manager=ArtifactManager(session_id="full"), policy=tiny):
        assert processor(messages) is messages


def test_profile_instruction_fallback_and_explicit_empty_override() -> None:
    model = ScriptedModel(responses=[response(0, tool=False), response(1, tool=False)])
    adapter = LangChainAdapter(model)
    with adapter.profile_override(AgentProfile(system="Profile instructions")):
        adapter.invoke_sync(prompt="first")
        adapter.invoke_sync(prompt="second", instructions="")
    assert model.seen[0][0].content == "Profile instructions"
    assert all(m.content != "Profile instructions" for m in model.seen[1])


def test_explicit_none_does_not_inherit_constructor_limit() -> None:
    model = ScriptedModel(responses=[response(0), response(1, tool=False)])
    adapter = LangChainAdapter(model, default_usage_limits=UsageLimits(request_limit=1))
    with adapter.tool_scope([tool_spec(lambda step: "ok")]):
        assert adapter.invoke_sync(prompt="work", usage_limits=None) == "done"
    assert len(model.seen) == 2


def test_cancellation_closes_run_scope_without_committing_partial_history() -> None:
    from kedi.agent_adapter.adapters.langchain_runtime import _RUN

    async def run() -> None:
        entered = asyncio.Event()

        class WaitingModel(ScriptedModel):
            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop: Any = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                entered.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        conversation = AdapterConversationContext()
        adapter = LangChainAdapter(WaitingModel(responses=[]))
        with adapter.conversation_scope(conversation):
            task = asyncio.create_task(adapter.invoke(prompt="wait"))
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        assert _RUN.get() is None
        assert conversation.result_state == {}

    asyncio.run(run())


@pytest.mark.parametrize(
    "limits", [UsageLimits(count_tokens_before_request=True), UsageLimits(cost_limit=Decimal("1"))]
)
def test_unsupported_native_limit_modes_fail_before_model(limits: UsageLimits) -> None:
    model = ScriptedModel(responses=[response(0, tool=False)])
    adapter = LangChainAdapter(model, default_usage_limits=limits)
    with pytest.raises(ValueError, match="LangChain usage limits"):
        adapter.invoke_sync(prompt="work")
    assert model.seen == []
