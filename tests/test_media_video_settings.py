"""Gemini video settings: start and end offsets, frames per second, media resolution (#191)."""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from kedi._interactive.persistence import SessionValueDecoder, SessionValueEncoder
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters.langchain import media as langchain_media
from kedi.agent_adapter.adapters.pydantic import media as pydantic_media
from kedi.agent_adapter.request.media import MediaAttachment, MediaNamer
from kedi.errors import KediExecutionError
from kedi.media import BRIDGE_KEY, MediaError, Video, from_bridge, to_bridge
from tests.test_media_provider_bodies import MP4, Bodies, _run, pydantic_model

CLIP = Video.from_bytes(MP4, "video/mp4", name="clip")
SETTINGS: dict[str, Any] = {"start": 5, "end": 10.5, "fps": 2, "resolution": "low"}
GEMINI_PART = {
    "video_metadata": {"start_offset": "5s", "end_offset": "10.5s", "fps": 2.0},
    "media_resolution": {"level": "MEDIA_RESOLUTION_LOW"},
}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "clip.mp4").write_bytes(MP4)
    return tmp_path


def _program(binding: str, prompt: str = "Describe <<clip>>.") -> str:
    return f"> import: media\n{binding}\n[reply] << {prompt}\n= <reply>\n"


# Values


def test_settings_are_validated_kept_and_cleared() -> None:
    clip = CLIP.with_settings(**SETTINGS)
    assert (clip.start, clip.end, clip.fps, clip.resolution) == (5.0, 10.5, 2.0, "low")
    assert clip.label == "<video clip · video/mp4 · 28 B · 5s–10.5s · 2 fps · low resolution>"
    assert clip.with_settings(start=1).end == 10.5
    assert clip.with_settings(fps=None).fps is None
    assert clip.with_name("other").start == 5.0
    assert not CLIP.has_settings and clip.has_settings
    invalid_settings: tuple[tuple[dict[str, Any], str], ...] = (
        ({"start": -1}, "must not be negative"),
        ({"start": 5, "end": 5}, "must come before its end"),
        ({"fps": 0}, "above 0 and at most 24"),
        ({"fps": 24.5}, "above 0 and at most 24"),
        ({"resolution": "ultra"}, "low, medium or high"),
        ({"start": True}, "must be a number"),
        ({"end": float("inf")}, "must be finite"),
    )
    for invalid, message in invalid_settings:
        with pytest.raises(MediaError, match=message):
            CLIP.with_settings(**invalid)


def test_settings_are_part_of_the_content_identity() -> None:
    clip = CLIP.with_settings(start=5, end=10)
    assert clip.content_key == (*CLIP.content_key, "start=5", "end=10")
    assert clip != CLIP and hash(clip) != hash(CLIP)
    assert clip == CLIP.with_settings(start=5.0, end=10.0)
    # Two clips of one file are two attachments, not one repeated value.
    namer = MediaNamer()
    first = namer.name_for(clip.with_name("intro"), None)
    second = namer.name_for(CLIP.with_settings(start=20, end=30).with_name("ending"), None)
    assert (first, second) == ("intro", "ending")
    assert namer.name_for(clip.with_name("again"), None) == "intro"


def test_settings_survive_pickle_bridge_and_session_snapshots() -> None:
    clip = CLIP.with_settings(**SETTINGS)
    assert pickle.loads(pickle.dumps(clip)) == clip
    assert from_bridge(to_bridge(clip)[BRIDGE_KEY]) == clip
    bridged = to_bridge(clip)[BRIDGE_KEY]
    with pytest.raises(MediaError, match="fps must be a number"):
        from_bridge({**bridged, "fps": "fast"})

    encoded = SessionValueEncoder(kedi_models={}).encode(clip)
    decoded = SessionValueDecoder(kedi_models={}).decode(json.loads(json.dumps(encoded)))
    assert decoded == clip
    # A snapshot written before video settings existed still loads.
    del encoded["settings"]
    assert SessionValueDecoder(kedi_models={}).decode(encoded) == CLIP


def test_the_video_loader_takes_settings(workspace: Path) -> None:
    from kedi.lang import compile_program, parse_program

    source = (
        "> import: media\n"
        '[clip: Video] = `video("clip.mp4", start=5, end=10.5, fps=2, resolution="low")`\n'
        "= `clip`\n"
    )
    clip = compile_program(parse_program(source)).run_main()
    assert (clip.start, clip.end, clip.fps, clip.resolution) == (5.0, 10.5, 2.0, "low")


# Providers


def test_gemini_receives_the_settings_on_both_adapters(workspace: Path) -> None:
    source = _program(f'[clip: Video] = `video("clip.mp4", {_kwargs(SETTINGS)})`')
    bodies = Bodies()
    _run(source, pydantic_model("google", bodies, "gemini-3.5-flash"), PydanticAdapter)
    part = bodies.last_user()["parts"][1]
    assert part["videoMetadata"] == GEMINI_PART["video_metadata"]
    assert part["mediaResolution"] == GEMINI_PART["media_resolution"]

    langchain_part = _langchain_gemini_part(CLIP.with_settings(**SETTINGS))
    assert langchain_part["video_metadata"] == GEMINI_PART["video_metadata"]
    assert langchain_part["media_resolution"] == GEMINI_PART["media_resolution"]


def test_youtube_clips_go_to_gemini_as_urls_with_their_offsets() -> None:
    talk = Video.from_url("https://youtu.be/abc123").with_settings(start=60, end=90)
    part = _langchain_gemini_part(talk)
    assert part["file_data"]["file_uri"] == "https://youtu.be/abc123"
    assert part["video_metadata"] == {"start_offset": "60s", "end_offset": "90s"}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "bedrock"])
def test_other_providers_refuse_a_video_with_settings(workspace: Path, provider: str) -> None:
    source = _program('[clip: Video] = `video("clip.mp4", start=5)`')
    bodies = Bodies()
    with pytest.raises(KediExecutionError, match="video settings .* are read only by Gemini"):
        _run(source, pydantic_model(provider, bodies), PydanticAdapter)
    assert bodies == []


def test_an_unknown_provider_refuses_settings_instead_of_dropping_them(workspace: Path) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    calls: list[Any] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        return ModelResponse(parts=[TextPart("ok")])

    source = _program('[clip: Video] = `video("clip.mp4", fps=2)`')
    with pytest.raises(KediExecutionError, match="read only by Gemini"):
        _run(source, FunctionModel(respond), PydanticAdapter)
    assert calls == []
    assert (
        _run(
            _program('[clip: Video] = `video("clip.mp4")`'), FunctionModel(respond), PydanticAdapter
        )
        == "ok"
    )


def test_resolution_needs_gemini_3(workspace: Path) -> None:
    source = _program('[clip: Video] = `video("clip.mp4", resolution="high")`')
    with pytest.raises(KediExecutionError, match="resolution needs Gemini 3 or later"):
        _run(source, pydantic_model("google", Bodies(), "gemini-2.5-flash"), PydanticAdapter)
    offsets = _program('[clip: Video] = `video("clip.mp4", start=1, end=2)`')
    bodies = Bodies()
    _run(offsets, pydantic_model("google", bodies, "gemini-2.5-flash"), PydanticAdapter)
    assert bodies.last_user()["parts"][1]["videoMetadata"] == {
        "start_offset": "1s",
        "end_offset": "2s",
    }


def test_two_clips_of_one_file_are_two_attachments(workspace: Path) -> None:
    source = (
        "> import: media\n"
        '[intro: Video] = `video("clip.mp4", start=0, end=10)`\n'
        '[ending: Video] = `video("clip.mp4", start=20, end=30)`\n'
        "[reply] << Compare <<intro>> with <<ending>>.\n= <reply>\n"
    )
    bodies = Bodies()
    _run(source, pydantic_model("google", bodies, "gemini-3.5-flash"), PydanticAdapter)
    metadata = [
        part["videoMetadata"] for part in bodies.last_user()["parts"] if "videoMetadata" in part
    ]
    assert metadata == [
        {"start_offset": "0s", "end_offset": "10s"},
        {"start_offset": "20s", "end_offset": "30s"},
    ]


# History reuse


def test_history_recognizes_the_same_clip_and_resends_another() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    clip = MediaAttachment("clip", CLIP.with_settings(**SETTINGS))
    other = MediaAttachment("clip", CLIP.with_settings(start=20))
    parts = asyncio.run(_pydantic_prompt(clip))
    pydantic_history = [ModelRequest(parts=[UserPromptPart(content=parts)])]
    assert pydantic_media.sent_attachments(pydantic_history) == {"clip": clip.identity}

    langchain_history = [asyncio.run(_langchain_message(clip))]
    assert langchain_media.sent_attachments(langchain_history) == {"clip": clip.identity}
    assert clip.identity != other.identity


def _kwargs(settings: dict[str, Any]) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in settings.items())


async def _pydantic_prompt(attachment: MediaAttachment) -> Any:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    model = GoogleModel("gemini-3.5-flash", provider=GoogleProvider(api_key="test"))
    return await pydantic_media.user_prompt("Describe it.", [attachment], history=None, model=model)


async def _langchain_message(attachment: MediaAttachment) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="test")
    return await langchain_media.user_message(
        "Describe it.", [attachment], history=None, model=model
    )


def _langchain_gemini_part(video: Video) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key="test")
    message = asyncio.run(_langchain_message(MediaAttachment("clip", video)))
    (content,) = model._prepare_request([HumanMessage(content=message.content)])["contents"]
    return content.parts[1].model_dump(exclude_none=True, mode="json")


def test_a_different_clip_changes_the_cached_prompt_identity() -> None:
    from kedi.core.model_calls import _snapshot_prompt_input

    def fingerprint(video: Video) -> str:
        return _snapshot_prompt_input("clip", video, by_reference=True).identity.fingerprint

    first = CLIP.with_settings(start=0, end=10)
    assert fingerprint(first) == fingerprint(CLIP.with_settings(start=0, end=10))
    assert fingerprint(first) != fingerprint(CLIP.with_settings(start=20, end=30))


def test_native_settings_that_are_malformed_carry_no_identity() -> None:
    from kedi.agent_adapter.request.media import gemini_settings_key

    assert gemini_settings_key(
        {"start_offset": "5s", "fps": 2}, {"level": "MEDIA_RESOLUTION_HIGH"}
    ) == (
        "start=5",
        "fps=2.0",
        "resolution=high",
    )
    assert gemini_settings_key({"start_offset": "fives", "end_offset": 9, "fps": True}, 3) == ()
    assert gemini_settings_key(None, "MEDIA_RESOLUTION_UNKNOWN") == ()

    identity = langchain_media._block_identity
    assert identity({"type": "media", "data": "@@not base64@@"}) is None
    assert identity({"type": "media", "mime_type": "video/mp4"}) is None
    assert identity({"type": "media", "file_uri": "gs://bucket/clip"}) == (
        "video",
        "url",
        "gs://bucket/clip",
    )


def test_partial_settings_label_and_lower_alone() -> None:
    assert CLIP.with_settings(end=10).label.endswith("· 0s–10s>")
    assert CLIP.with_settings(start=4).label.endswith("· 4s–end>")
    assert CLIP.with_settings(fps=0.5).label.endswith("· 0.5 fps>")
    assert CLIP.with_settings(resolution="high").label.endswith("· high resolution>")
    part = _langchain_gemini_part(CLIP.with_settings(resolution="medium"))
    assert part["media_resolution"] == {"level": "MEDIA_RESOLUTION_MEDIUM"}
    assert "video_metadata" not in part
