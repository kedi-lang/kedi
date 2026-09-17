from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from kedi.agent_adapter.compaction import (
    PydanticArtifactHistoryCapability,
    PydanticArtifactHistoryProcessor,
    PydanticHistoryArchiveSettings,
)
from kedi.agent_adapter.tools.scope import artifact_admission_scope
from kedi.artifacts import ArtifactManager, ArtifactPolicy


def _history(*, cycles: int, payload_chars: int = 900) -> list[ModelMessage]:
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("finish the task")])]
    for index in range(cycles):
        call_id = f"call_{index}"
        messages.extend(
            (
                ModelResponse(
                    parts=[
                        ThinkingPart("r" * payload_chars, id=f"reasoning_{index}"),
                        ToolCallPart("terminal", {"step": index}, call_id),
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "terminal",
                            {"stdout": "x" * payload_chars},
                            call_id,
                        )
                    ]
                ),
            )
        )
    return messages


def _policy(path: Path, *, quota: int = 1_000_000) -> ArtifactPolicy:
    return ArtifactPolicy.patch(
        {
            "enabled": True,
            "store": "file",
            "path": path,
            "threshold_bytes": 100_000,
            "session_quota_bytes": quota,
        }
    )


def _processor() -> PydanticArtifactHistoryProcessor:
    return PydanticArtifactHistoryProcessor(
        PydanticHistoryArchiveSettings(
            threshold_tokens=2_000,
            preserve_recent_tokens=500,
            minimum_reduction_tokens=500,
        )
    )


def _checkpoint_payload(message: ModelMessage) -> dict[str, Any]:
    assert isinstance(message, ModelRequest)
    part = message.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, str)
    _, encoded, _ = part.content.splitlines()
    return cast("dict[str, Any]", json.loads(encoded))


def test_history_archive_settings_validate_bounds() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PydanticHistoryArchiveSettings(threshold_tokens=0)
    with pytest.raises(ValueError, match="lower than threshold"):
        PydanticHistoryArchiveSettings(
            threshold_tokens=100,
            preserve_recent_tokens=100,
        )


def test_history_archive_is_identity_noop_below_threshold(tmp_path: Path) -> None:
    messages = _history(cycles=1, payload_chars=10)
    manager = ArtifactManager(session_id="below_threshold")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        updated = _processor()(messages)

    assert updated is messages
    assert manager.stats()["active_artifacts"] == 0
    manager.close()


def test_history_archive_uses_provider_input_growth_when_local_estimate_is_small(
    tmp_path: Path,
) -> None:
    messages = _history(cycles=6, payload_chars=20)
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    for response, input_tokens in zip(
        responses,
        (10_000, 11_000, 12_000, 13_000, 14_000, 16_000),
        strict=True,
    ):
        response.usage = RequestUsage(input_tokens=input_tokens, output_tokens=10)
        response.provider_name = "openai"

    processor = PydanticArtifactHistoryProcessor(
        PydanticHistoryArchiveSettings(
            threshold_tokens=5_000,
            preserve_recent_tokens=100,
            minimum_reduction_tokens=100,
        )
    )
    assert processor.codec.inspect({processor.codec.state_key: messages}).total_tokens < 5_000

    manager = ArtifactManager(session_id="provider_pressure")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        updated = processor(messages)

    assert updated is not messages
    checkpoint = next(
        message
        for message in updated
        if message.metadata and message.metadata.get("kedi_compaction_checkpoint")
    )
    assert checkpoint.metadata is not None
    assert checkpoint.metadata["kedi_history_provider_input_baseline_tokens"] == 10_000
    manager.close()


def test_history_archive_preserves_provider_baseline_across_epochs(tmp_path: Path) -> None:
    processor = PydanticArtifactHistoryProcessor(
        PydanticHistoryArchiveSettings(
            threshold_tokens=5_000,
            preserve_recent_tokens=100,
            minimum_reduction_tokens=100,
        )
    )
    messages = _history(cycles=6, payload_chars=20)
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    for response, input_tokens in zip(
        responses,
        (10_000, 11_000, 12_000, 13_000, 14_000, 16_000),
        strict=True,
    ):
        response.usage = RequestUsage(input_tokens=input_tokens, output_tokens=10)
        response.provider_name = "openai"

    manager = ArtifactManager(session_id="provider_epochs")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        first = processor(messages)
        first_checkpoint = next(
            message
            for message in first
            if message.metadata and message.metadata.get("kedi_compaction_checkpoint")
        )
        extended = [*first]
        for index, input_tokens in enumerate((11_000, 12_000, 14_000, 16_000), start=6):
            call_id = f"call_{index}"
            extended.extend(
                (
                    ModelResponse(
                        parts=[ToolCallPart("terminal", {"step": index}, call_id)],
                        usage=RequestUsage(input_tokens=input_tokens, output_tokens=10),
                        provider_name="openai",
                    ),
                    ModelRequest(parts=[ToolReturnPart("terminal", {"stdout": "ok"}, call_id)]),
                )
            )
        second = processor(extended)

    checkpoints = [
        message
        for message in second
        if message.metadata and message.metadata.get("kedi_compaction_checkpoint")
    ]
    assert len(checkpoints) == 2
    assert checkpoints[0] is first_checkpoint
    assert {
        message.metadata["kedi_history_provider_input_baseline_tokens"]
        for message in checkpoints
        if message.metadata is not None
    } == {10_000}
    manager.close()


def test_history_archive_skips_retry_barrier_and_archives_later_closed_cycles(
    tmp_path: Path,
) -> None:
    retry_call = ModelResponse(parts=[ToolCallPart("terminal", {"invalid": True}, "retry_call")])
    retry_result = ModelRequest(
        parts=[
            RetryPromptPart(
                "invalid arguments",
                tool_name="terminal",
                tool_call_id="retry_call",
            )
        ]
    )
    later_cycles = _history(cycles=6, payload_chars=300)[1:]
    messages = [
        ModelRequest(parts=[UserPromptPart("finish the task")]),
        retry_call,
        retry_result,
        *later_cycles,
    ]
    manager = ArtifactManager(session_id="retry_barrier")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        updated = _processor()(messages)

    assert updated[0] is messages[0]
    assert updated[1] is retry_call
    assert updated[2] is retry_result
    assert any(
        message.metadata and message.metadata.get("kedi_compaction_checkpoint")
        for message in updated[3:]
    )
    manager.close()


def test_history_archive_preserves_prefix_tail_and_exact_source(tmp_path: Path) -> None:
    messages = _history(cycles=8)
    original = list(messages)
    manager = ArtifactManager(session_id="archive")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        updated = _processor()(messages)

    assert len(updated) < len(messages)
    assert updated[0] is original[0]
    assert updated[-1] is original[-1]
    assert updated[-2] is original[-2]
    assert updated[1].metadata is not None
    assert updated[1].metadata["kedi_compaction_checkpoint"].startswith("history_")

    checkpoint = _checkpoint_payload(updated[1])
    archive_ref = checkpoint["resources"][0].split(": ", 1)[1]
    archived = manager.get_value(archive_ref)
    restored = ModelMessagesTypeAdapter.validate_python(archived["messages"])
    assert restored == original[1 : 1 + len(restored)]
    assert archived["source_digest"].startswith(
        updated[1].metadata["kedi_compaction_checkpoint"][8:]
    )
    assert manager.metadata(archive_ref)["sensitive"] is True
    assert any(item.startswith("terminal: calls=") for item in checkpoint["completed_outcomes"])
    manager.close()


def test_history_archive_never_splits_an_open_tool_lifecycle(tmp_path: Path) -> None:
    messages = _history(cycles=6)
    dangling = ModelResponse(
        parts=[
            ThinkingPart("unfinished" * 100, id="open_reasoning"),
            ToolCallPart("terminal", {"step": "open"}, "open_call"),
        ]
    )
    trailing_request = ModelRequest(parts=[])
    messages.extend((dangling, trailing_request))
    manager = ArtifactManager(session_id="open_lifecycle")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        updated = _processor()(messages)

    assert dangling in updated
    assert trailing_request in updated
    assert any(
        isinstance(part, ToolCallPart) and part.tool_call_id == "open_call"
        for message in updated
        if isinstance(message, ModelResponse)
        for part in message.parts
    )
    manager.close()


def test_history_archive_keeps_committed_checkpoint_immutable_across_epochs(
    tmp_path: Path,
) -> None:
    messages = _history(cycles=8)
    manager = ArtifactManager(session_id="epochs")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        first = _processor()(messages)
        first_checkpoint = first[1]
        first_checkpoint_dump = ModelMessagesTypeAdapter.dump_python(
            [first_checkpoint], mode="json"
        )
        extended = [*first]
        for index in range(8, 16):
            call_id = f"call_{index}"
            extended.extend(
                (
                    ModelResponse(
                        parts=[
                            ThinkingPart("r" * 900, id=f"reasoning_{index}"),
                            ToolCallPart("terminal", {"step": index}, call_id),
                        ]
                    ),
                    ModelRequest(
                        parts=[ToolReturnPart("terminal", {"stdout": "x" * 900}, call_id)]
                    ),
                )
            )
        second = _processor()(extended)

    assert second[0] is messages[0]
    assert second[1] is first_checkpoint
    assert ModelMessagesTypeAdapter.dump_python([second[1]], mode="json") == first_checkpoint_dump
    assert (
        sum(
            bool(message.metadata and message.metadata.get("kedi_compaction_checkpoint"))
            for message in second
        )
        == 2
    )
    assert manager.stats()["active_artifacts"] == 2
    manager.close()


def test_history_archive_fails_open_without_storage_or_quota(tmp_path: Path) -> None:
    messages = _history(cycles=8)
    assert _processor()(messages) is messages

    manager = ArtifactManager(session_id="quota")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path, quota=100)):
        assert _processor()(messages) is messages
    assert manager.stats()["active_artifacts"] == 0
    manager.close()

    class UnavailableStore:
        def artifactize(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise OSError("disk unavailable")

    with artifact_admission_scope(
        manager=UnavailableStore(),
        policy=_policy(tmp_path),
    ):
        assert _processor()(messages) is messages


def test_history_archive_capability_rotates_cache_key_once_per_checkpoint(
    tmp_path: Path,
) -> None:
    messages = _history(cycles=8)
    manager = ArtifactManager(session_id="cache_epoch")
    capability = PydanticArtifactHistoryCapability(
        PydanticHistoryArchiveSettings(
            threshold_tokens=2_000,
            preserve_recent_tokens=500,
            minimum_reduction_tokens=500,
        )
    )
    request = ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=cast("ModelSettings", {"openai_prompt_cache_key": "stable"}),
        model_request_parameters=ModelRequestParameters(),
    )
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        first = asyncio.run(capability.before_model_request(cast("RunContext[Any]", None), request))
        assert first.model_settings is not None
        first_key = cast("dict[str, Any]", first.model_settings)["openai_prompt_cache_key"]
        second = asyncio.run(capability.before_model_request(cast("RunContext[Any]", None), first))

    assert isinstance(first_key, str)
    assert first_key.startswith("stable_h")
    assert second.model_settings is not None
    assert cast("dict[str, Any]", second.model_settings)["openai_prompt_cache_key"] == first_key
    manager.close()


def test_history_archive_runs_inside_a_real_pydantic_tool_loop(tmp_path: Path) -> None:
    step = 0
    seen_message_counts: list[int] = []
    seen_cache_keys: list[str] = []

    def evidence() -> dict[str, object]:
        nonlocal step
        step += 1
        return {"step": step, "evidence": "x" * 900}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_message_counts.append(len(messages))
        cache_key = (info.model_settings or {}).get("openai_prompt_cache_key")
        if isinstance(cache_key, str):
            seen_cache_keys.append(cache_key)
        if step < 12:
            return ModelResponse(
                parts=[
                    ThinkingPart("r" * 900, id=f"reasoning_{step}"),
                    ToolCallPart("evidence", {}, f"call_{step}"),
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    capability = PydanticArtifactHistoryCapability(
        PydanticHistoryArchiveSettings(
            threshold_tokens=3_000,
            preserve_recent_tokens=700,
            minimum_reduction_tokens=500,
        )
    )
    agent = Agent(
        FunctionModel(respond),
        tools=[Tool(evidence, sequential=True)],
        capabilities=[capability],
        model_settings=cast("ModelSettings", {"openai_prompt_cache_key": "stable"}),
    )
    manager = ArtifactManager(session_id="pydantic_loop")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        result = agent.run_sync("collect all evidence")

    assert result.output == "done"
    assert step == 12
    assert manager.stats()["active_artifacts"] >= 1
    assert any(key.startswith("stable_h") for key in seen_cache_keys)
    assert max(seen_message_counts) < 1 + (12 * 2)
    assert any(
        message.metadata and message.metadata.get("kedi_compaction_checkpoint")
        for message in result.all_messages()
    )
    manager.close()
