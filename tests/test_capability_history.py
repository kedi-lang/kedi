"""Deferred activation evidence must survive Kedi history processing."""

import asyncio

import pytest
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from kedi.agent_adapter import HistoryProcessingValidationError
from kedi.agent_adapter.adapters.pydantic.history_processing import PydanticHistoryMessageCodec
from kedi.agent_adapter.compaction.pydantic_codec import PydanticHistoryCodec
from kedi.agent_adapter.compaction import (
    PydanticArtifactHistoryProcessor,
    PydanticHistoryArchiveSettings,
)
from kedi.agent_adapter.history_processing import HistoryProcessorRunner
from kedi.agent_adapter.tools.scope import artifact_admission_scope
from kedi.artifacts import ArtifactManager, ArtifactPolicy


def loaded_history():
    return [
        ModelRequest(parts=[UserPromptPart("Find the current revision.")]),
        ModelResponse(parts=[LoadCapabilityCallPart(args={"id": "reader"}, tool_call_id="load")]),
        ModelRequest(
            parts=[
                LoadCapabilityReturnPart(
                    content={"instructions": "Use exact revisions."}, tool_call_id="load"
                )
            ]
        ),
        ModelResponse(parts=[TextPart("Revision found.")]),
        ModelRequest(parts=[UserPromptPart("Continue.")]),
    ]


@pytest.mark.parametrize("serialized", [False, True])
def test_codec_marks_native_load_evidence_protected(serialized):
    messages = loaded_history()
    if serialized:
        messages = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    snapshot = PydanticHistoryCodec().inspect({"pydantic_messages": messages})
    assert snapshot.entries[1].provider_required
    assert snapshot.entries[2].provider_required
    assert not snapshot.entries[0].provider_required


@pytest.mark.parametrize("operation", ["remove", "replace", "reorder"])
def test_history_processor_cannot_destroy_or_forge_loaded_state(operation):
    messages = loaded_history()

    def processor(ctx):
        if operation == "remove":
            return ctx.messages[-1:]
        if operation == "replace":
            edited = list(ctx.messages)
            edited[1].parts[0].args = {"id": "other_capability"}
            return edited
        return [ctx.messages[0], ctx.messages[2], ctx.messages[1], *ctx.messages[3:]]

    runner = HistoryProcessorRunner(processor, PydanticHistoryMessageCodec())

    async def run():
        with runner.run_scope(None):
            return await runner.process(messages, model_id="fixture")

    with pytest.raises(HistoryProcessingValidationError):
        asyncio.run(run())
    assert messages[1].parts[0].args == {"id": "reader"}


def test_user_tool_with_loader_name_is_not_native_activation_evidence():
    messages = [
        ModelResponse(parts=[ToolCallPart("load_capability", {"id": "reader"}, "call")]),
        ModelRequest(parts=[ToolReturnPart("load_capability", "loaded", "call")]),
    ]
    snapshot = PydanticHistoryCodec().inspect({"pydantic_messages": messages})
    assert not any(entry.provider_required for entry in snapshot.entries)


def test_artifact_archive_preserves_activation_evidence_while_compacting_other_groups(tmp_path):
    messages = loaded_history()
    for index in range(6):
        messages.extend(
            [
                ModelResponse(parts=[ToolCallPart("read", {"index": index}, f"read-{index}")]),
                ModelRequest(
                    parts=[ToolReturnPart("read", "Long tool result. " * 400, f"read-{index}")]
                ),
            ]
        )
    processor = PydanticArtifactHistoryProcessor(
        PydanticHistoryArchiveSettings(
            threshold_tokens=2_000,
            preserve_recent_tokens=100,
            minimum_reduction_tokens=100,
        )
    )
    manager = ArtifactManager(session_id="capability-history")
    policy = ArtifactPolicy.patch({"enabled": True, "store": "file", "path": tmp_path})
    try:
        with artifact_admission_scope(manager=manager, policy=policy):
            updated = processor(messages)
        assert len(updated) < len(messages)
        assert any(
            message.metadata and message.metadata.get("kedi_compaction_checkpoint")
            for message in updated
        )
        for part_type in (LoadCapabilityCallPart, LoadCapabilityReturnPart):
            before = [
                part
                for message in messages
                for part in message.parts
                if isinstance(part, part_type)
            ]
            after = [
                part for message in updated for part in message.parts if isinstance(part, part_type)
            ]
            assert before == after
    finally:
        manager.close()
