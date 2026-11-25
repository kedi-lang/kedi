"""History compaction stores archived media as artifacts of their own (#192)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.compaction.archive import HistoryArchiveSettings
from kedi.agent_adapter.compaction.langchain_archive import LangChainArtifactHistoryProcessor
from kedi.agent_adapter.tools.scope import artifact_admission_scope
from kedi.artifacts import ArtifactInputError, ArtifactManager, ArtifactPolicy
from kedi.artifacts.tools import artifact_tool_specs
from kedi.media import Image
from tests.test_media_values import png_bytes
from tests.test_pydantic_history_archive import _checkpoint_payload, _policy, _processor

PNG = png_bytes()
OTHER_PNG = PNG + b"\x00"


def _media_history(*, cycles: int = 8) -> list[ModelMessage]:
    """A tool loop whose results hold images: one repeated, one other, one by URL."""

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    ["look", BinaryContent(PNG, media_type="image/png", identifier="cat")]
                )
            ]
        )
    ]
    for index in range(cycles):
        call_id = f"call_{index}"
        image = BinaryContent(
            PNG if index % 2 else OTHER_PNG, media_type="image/png", identifier=f"shot_{index}"
        )
        messages.extend(
            (
                ModelResponse(parts=[TextPart("x" * 600), ToolCallPart("snap", {}, call_id)]),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "snap",
                            [
                                f"<image shot_{index}>",
                                image,
                                ImageUrl("https://example.com/cat.png"),
                                "y" * 600,
                            ],
                            call_id,
                        )
                    ]
                ),
            )
        )
    return messages


def _archive(manager: ArtifactManager, policy: ArtifactPolicy) -> tuple[list[Any], dict[str, Any]]:
    messages = _media_history()
    with artifact_admission_scope(manager=manager, policy=policy):
        updated = _processor()(messages)
    assert updated is not messages
    (message,) = [
        message
        for message in updated
        if isinstance(message, ModelRequest)
        and isinstance(message.parts[0], UserPromptPart)
        and isinstance(message.parts[0].content, str)
        and message.parts[0].content.startswith("<kedi_compaction_checkpoint>")
    ]
    checkpoint = _checkpoint_payload(message)
    return updated, checkpoint


def test_archived_media_become_artifacts_and_the_archive_names_them(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="archived_media")
    _, checkpoint = _archive(manager, _policy(tmp_path))
    transcript_ref = checkpoint["resources"][0].split(": ", 1)[1]
    archive = manager.get_value(transcript_ref)
    text = json.dumps(archive)
    # No base64 is left in the archive; each payload names its artifact instead.
    assert "iVBORw0KGgo" not in text
    references = sorted(set(_references(archive)))
    # The user's cat and the repeated shot are one image, so two artifacts in all.
    assert len(references) == 2
    assert "https://example.com/cat.png" in text
    media_lines = [line for line in checkpoint["resources"][1:]]
    assert len(media_lines) == 2
    assert all("read_artifact returns it as media" in line for line in media_lines)
    for ref in references:
        assert ref in checkpoint["artifact_refs"]
        metadata = manager.metadata(ref)
        assert metadata["sensitive"] is True
        assert metadata["provenance"]["media_type"] == "image/png"
    manager.close()


def test_read_artifact_returns_archived_media_as_media(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="read_media")
    _, checkpoint = _archive(manager, _policy(tmp_path))
    ref = checkpoint["resources"][1].split(": ", 1)[1].split(" ", 1)[0]
    read_artifact = next(
        spec for spec in artifact_tool_specs(manager) if spec.name == "read_artifact"
    )
    image = read_artifact.fn(ref_id=ref)
    assert isinstance(image, Image) and image.data in (PNG, OTHER_PNG)
    with pytest.raises(ArtifactInputError, match="is archived media; read it whole"):
        read_artifact.fn(ref_id=ref, pattern="x")
    manager.close()


def test_a_model_reads_archived_media_natively(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="model_reads_media")
    _, checkpoint = _archive(manager, _policy(tmp_path))
    ref = checkpoint["resources"][1].split(": ", 1)[1].split(" ", 1)[0]
    seen: list[Any] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if returns:
            seen.append(returns[0].content)
            return ModelResponse(parts=[TextPart("seen")])
        return ModelResponse(parts=[ToolCallPart("read_artifact", {"ref_id": ref}, "c1")])

    adapter = PydanticAdapter(model=FunctionModel(respond))
    for spec in artifact_tool_specs(manager):
        if spec.name == "read_artifact":
            adapter.register_tool(spec)
    assert adapter.run_sync("Look again.").output == "seen"
    ((text, image),) = seen
    assert text.startswith("<image ") and isinstance(image, BinaryContent)
    manager.close()


def test_a_full_quota_rejects_the_compaction_and_keeps_no_media(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="quota")
    messages = _media_history()
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path, quota=80)):
        assert _processor()(messages) is messages
    assert manager.stats()["active_artifacts"] == 0
    manager.close()


def test_langchain_archives_store_media_the_same_way(tmp_path: Path) -> None:
    import base64

    encoded = base64.b64encode(PNG).decode()
    messages: list[Any] = [HumanMessage("look")]
    for index in range(8):
        call_id = f"call_{index}"
        messages.extend(
            (
                AIMessage(
                    content="x" * 600,
                    tool_calls=[{"name": "snap", "args": {}, "id": call_id}],
                ),
                ToolMessage(
                    content=[
                        {"type": "text", "text": f"<image shot_{index}>" + "y" * 600},
                        {
                            "type": "image",
                            "base64": encoded,
                            "mime_type": "image/png",
                            "id": "shot",
                        },
                    ],
                    tool_call_id=call_id,
                ),
            )
        )
    processor = LangChainArtifactHistoryProcessor(
        HistoryArchiveSettings(
            threshold_tokens=2_000, preserve_recent_tokens=500, minimum_reduction_tokens=500
        )
    )
    manager = ArtifactManager(session_id="langchain_media")
    with artifact_admission_scope(manager=manager, policy=_policy(tmp_path)):
        updated = processor(messages)
    assert len(updated) < len(messages)
    refs = [ref for ref in manager.lifecycle_snapshot()[0]]
    sources = {manager.metadata(ref)["source"] for ref in refs}
    assert sources == {"history:compaction", "history:media"}
    archive_ref = next(
        ref for ref in refs if manager.metadata(ref)["source"] == "history:compaction"
    )
    assert encoded not in json.dumps(manager.get_value(archive_ref))
    manager.close()


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.removeprefix("kedi-artifact:")] if value.startswith("kedi-artifact:") else []
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _references(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in _references(item)]
    return []


def test_a_manager_that_stores_no_media_artifact_is_an_error() -> None:
    from types import SimpleNamespace

    manager = SimpleNamespace(artifactize=lambda *args, **kwargs: "not a handle")
    with (
        artifact_admission_scope(manager=manager, policy=ArtifactPolicy.patch({"enabled": True})),
        pytest.raises(RuntimeError, match="forced media archival did not create an artifact"),
    ):
        _processor()(_media_history())
