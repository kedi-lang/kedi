"""Speech output: ``[speech: Audio] << Read this aloud.`` returns an Audio (#190)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.media import Audio, MediaError
from tests.lsp_helpers import diagnostics_for
from tests.test_media_values import png_bytes

SPEAK = "> import: media\n[speech: Audio] << Say good morning.\n= `speech`\n"


def wav(frames: int = 240, *, rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


WAV = wav()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


# Audio values


def test_speech_keeps_its_transcript_as_a_label() -> None:
    audio = Audio.from_bytes(WAV, "audio/wav")
    spoken = audio.with_transcript("Good morning, " * 10)
    assert spoken == audio and hash(spoken) == hash(audio)
    assert spoken.with_transcript(None).transcript is None
    assert "transcript='Good morning, Good morning," in repr(spoken)
    assert repr(spoken).endswith("...')")
    with pytest.raises(MediaError, match="transcript must be text, not int"):
        Audio(media_type="audio/wav", data=WAV, transcript=3)  # type: ignore[arg-type]


# Pydantic AI: a one-shot realtime session


class _Session:
    def __init__(self, record: dict[str, Any], parts: list[Any], error: Exception | None) -> None:
        self.record = record
        self.parts = parts
        self.error = error

    async def send(self, prompt: str) -> None:
        self.record["prompt"] = prompt

    async def wait_for_reply(self) -> None:
        if self.error is not None:
            raise self.error
        await asyncio.sleep(self.record.get("delay", 0))

    def new_messages(self) -> list[Any]:
        from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

        return [ModelRequest(parts=[UserPromptPart("x")]), ModelResponse(parts=self.parts)]


def _fake_realtime(
    monkeypatch: pytest.MonkeyPatch,
    parts: list[Any] | None = None,
    *,
    error: Exception | None = None,
) -> dict[str, Any]:
    import pydantic_ai
    from pydantic_ai.messages import BinaryContent, SpeechPart, TextPart

    if parts is None:
        parts = [
            TextPart("ignored"),
            SpeechPart(speaker="assistant", transcript="Good morning!"),
            SpeechPart(
                speaker="assistant",
                transcript="Good morning!",
                audio=BinaryContent(WAV, media_type="audio/wav"),
            ),
        ]
    record: dict[str, Any] = {}

    class Agent:
        def __init__(self, *, instructions: str | None) -> None:
            record["instructions"] = instructions

        def realtime(self, model: Any, *, model_settings: Any, usage: Any) -> Any:
            record["model"] = model
            record["settings"] = model_settings
            usage.requests = 1
            usage.output_tokens = 7

            @asynccontextmanager
            async def session(*, audio_retention: str) -> AsyncIterator[_Session]:
                record["retention"] = audio_retention
                yield _Session(record, parts, error)

            return type("Bound", (), {"session": staticmethod(session)})()

    monkeypatch.setattr(pydantic_ai, "Agent", Agent)
    return record


def _live_model(name: str = "gemini-3.1-flash-live-preview") -> Any:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    return GoogleModel(name, provider=GoogleProvider(api_key="test"))


def test_a_realtime_model_speaks_the_reply(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai.realtime.google import GoogleRealtimeModel

    from kedi.agent_adapter.invocation import observe_invocations

    record = _fake_realtime(monkeypatch)
    usages: list[Any] = []
    source = (
        "> import: media\n"
        "> instructions: Speak warmly.\n"
        "> settings:\n    voice: Kore\n    timeout: 30\n"
        "[speech: Audio] << Say good morning.\n= `speech`\n"
    )
    with observe_invocations(usages.append):
        speech = _run(source, PydanticAdapter(model=_live_model()))
    assert isinstance(speech, Audio)
    assert (speech.data, speech.media_type, speech.transcript) == (
        WAV,
        "audio/wav",
        "Good morning!",
    )
    assert record["prompt"] == "Say good morning."
    assert record["instructions"] == "Speak warmly."
    assert record["settings"] == {"google_voice": "Kore"}
    assert record["retention"] == "output_audio"
    # The selected model's provider, and so its credentials, carry over to the session.
    assert isinstance(record["model"], GoogleRealtimeModel)
    assert [(usage.requests, usage.output_tokens) for usage in usages] == [(1, 7)]


@pytest.mark.parametrize(
    ("error", "parts", "delay", "message"),
    [
        (None, [], 0, "returned no speech"),
        (None, None, 5, "gave no speech within the timeout"),
        ("realtime", None, 0, "cannot produce speech: closed"),
    ],
)
def test_realtime_failures_name_the_model(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str | None,
    parts: list[Any] | None,
    delay: float,
    message: str,
) -> None:
    from pydantic_ai.realtime import RealtimeError

    record = _fake_realtime(
        monkeypatch, parts, error=RealtimeError("live", "closed") if error else None
    )
    record["delay"] = delay
    source = SPEAK.replace("[speech", "> settings:\n    timeout: 0.05\n[speech")
    with pytest.raises(KediExecutionError, match=f"gemini-3.1-flash-live-preview {message}"):
        _run(source, PydanticAdapter(model=_live_model()))


def test_speech_in_parts_is_joined_into_one_wav() -> None:
    from kedi.agent_adapter.adapters.pydantic.speech import join_wav

    with wave.open(io.BytesIO(join_wav([WAV, wav(100)]))) as reader:
        assert reader.getnframes() == 340
    with pytest.raises(MediaError, match="WAV parts of different formats"):
        join_wav([WAV, wav(rate=16_000)])


def test_realtime_models_come_from_the_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel
    from pydantic_ai.realtime.google import GoogleRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel
    from pydantic_ai.realtime.xai import XaiRealtimeModel

    from kedi.agent_adapter.adapters.pydantic.speech import _voice_settings, realtime_model

    provider = OpenAIProvider(api_key="test")
    openai = realtime_model(OpenAIChatModel("gpt-realtime", provider=provider))
    assert isinstance(openai, OpenAIRealtimeModel) and openai.model_name == "gpt-realtime"
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    gemini = realtime_model("google-gla:gemini-3.1-flash-live-preview")
    assert isinstance(gemini, GoogleRealtimeModel)
    with pytest.raises(MediaError, match="Model test:test cannot produce speech"):
        realtime_model(TestModel())
    assert _voice_settings(openai, None) is None
    assert _voice_settings(openai, "alloy") == {"openai_voice": "alloy"}
    # The voice key follows the model class; building these needs no connection.
    xai = XaiRealtimeModel.__new__(XaiRealtimeModel)
    assert _voice_settings(xai, "Ara") == {"xai_voice": "Ara"}
    eleven = ElevenLabsRealtimeModel.__new__(ElevenLabsRealtimeModel)
    assert _voice_settings(eleven, "Rachel") == {"elevenlabs_voice": "Rachel"}


# LangChain: audio chat models


class _Recorded:
    calls: list[list[Any]] = []

    @classmethod
    def last(cls) -> list[str]:
        return [type(message).__name__ for message in cls.calls[-1]]


def _fake_langchain(
    *messages: Any,
    profile: dict[str, Any] | None,
    name: str,
    module: str | None = None,
    replies: dict[str, Any] | None = None,
) -> Any:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Speaker(GenericFakeChatModel):
        model: str = name

        def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
            _Recorded.calls.append(list(messages))
            if replies is not None:
                reply = replies[str(messages[-1].content)]
                return ChatResult(generations=[ChatGeneration(message=reply)])
            return super()._generate(messages, *args, **kwargs)

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            # Text turns offer Kedi's tools; this fake answers without calling them.
            return self

    if module is not None:
        Speaker.__module__ = module
    _Recorded.calls = []
    model = Speaker(messages=iter(messages))
    model.profile = profile
    return model


def _wav_block_message() -> Any:
    from langchain_core.messages import AIMessage

    block = {"type": "audio", "base64": base64.b64encode(WAV).decode(), "mime_type": "audio/wav"}
    return AIMessage(content=[block])


def test_a_tts_model_reads_the_prompt_without_instructions(workspace: Path) -> None:
    source = SPEAK.replace("[speech", "> instructions: Speak warmly.\n[speech")
    # A newer Gemini TTS model without a LangChain profile yet speaks by its name.
    model = _fake_langchain(
        _wav_block_message(),
        profile=None,
        name="gemini-3.8-flash-tts",
        module="langchain_google_genai.chat_models",
    )
    speech = _run(source, LangChainAdapter(model=model))
    assert isinstance(speech, Audio) and speech.data == WAV
    assert _Recorded.last() == ["HumanMessage"]


def test_an_audio_model_keeps_the_instructions(workspace: Path) -> None:
    from langchain_core.messages import AIMessage

    source = SPEAK.replace("[speech", "> instructions: Speak warmly.\n[speech")
    message = AIMessage(content="", additional_kwargs={"audio": WAV})
    model = _fake_langchain(message, profile={"audio_outputs": True}, name="speaker")
    assert _run(source, LangChainAdapter(model=model)).data == WAV
    assert _Recorded.last() == ["SystemMessage", "HumanMessage"]


@pytest.mark.parametrize(
    ("profile", "name", "settings", "message"),
    [
        ({"audio_outputs": False}, "gemini-3.5-flash", "", "cannot produce speech"),
        ({"audio_outputs": True}, "gemini-3.8-flash-tts", "voice: Kore", "takes no voice"),
        ({"audio_outputs": True}, "speaker", "", "returned no speech"),
        # The name rule is the Gemini client's: another provider's -tts model needs a profile.
        (None, "gpt-4o-mini-tts", "", "cannot produce speech"),
    ],
)
def test_langchain_refuses_what_cannot_speak(
    workspace: Path, profile: dict[str, Any], name: str, settings: str, message: str
) -> None:
    from langchain_core.messages import AIMessage

    source = (
        SPEAK.replace("[speech", f"> settings:\n    {settings}\n[speech") if settings else SPEAK
    )
    model = _fake_langchain(AIMessage(content="no audio"), profile=profile, name=name)
    with pytest.raises(KediExecutionError, match=message):
        _run(source, LangChainAdapter(model=model))


def test_openai_audio_models_are_asked_for_wav_in_one_reply(workspace: Path) -> None:
    from langchain_openai import ChatOpenAI

    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content or b"{}"))
        audio = {
            "id": "audio_1",
            "data": base64.b64encode(WAV).decode(),
            "transcript": "Good morning, team.",
            "expires_at": 0,
        }
        choice = {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": None, "audio": audio},
        }
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-audio",
                "choices": [choice],
            },
        )

    model = ChatOpenAI(
        model="gpt-audio",
        api_key="test",  # type: ignore[arg-type]
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    source = SPEAK.replace("[speech", "> settings:\n    voice: verse\n[speech")
    speech = _run(source, LangChainAdapter(model=model))
    assert (speech.data, speech.transcript) == (WAV, "Good morning, team.")
    (body,) = bodies
    assert body["modalities"] == ["text", "audio"]
    assert body["audio"] == {"voice": "verse", "format": "wav"}
    assert not body.get("stream") and "tools" not in body


# Every adapter, and the editor


def test_adapters_without_speech_output_are_refused_before_the_call(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.agent_adapter.adapters.claude import ClaudeAdapter
    from tests.test_agent_adapters import _install_fake_claude_sdk

    events = _install_fake_claude_sdk(monkeypatch)
    with pytest.raises(KediExecutionError, match="Adapter 'claude' cannot produce speech"):
        _run(SPEAK, ClaudeAdapter(model="claude-sonnet-4"))
    assert events["prompts"] == []


def test_a_speech_request_sends_no_media(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _fake_realtime(monkeypatch)
    (workspace / "cat.png").write_bytes(png_bytes())
    source = (
        "> import: media\n"
        '[photo: Image] = `image("cat.png")`\n'
        "[speech: Audio] << Describe <<photo>> aloud.\n= `speech`\n"
    )
    with pytest.raises(KediExecutionError, match="photo: a speech request sends text only"):
        _run(source, PydanticAdapter(model=_live_model()))
    assert "prompt" not in record


def test_the_voice_setting_must_be_text(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_realtime(monkeypatch)
    source = SPEAK.replace("[speech", "> settings:\n    voice: `3`\n[speech")
    with pytest.raises(KediExecutionError, match="voice"):
        _run(source, PydanticAdapter(model=_live_model()))


def test_the_editor_reports_speech_output_on_adapters_without_it() -> None:
    def codes(head: str) -> list[tuple[str, int]]:
        return [
            (str(diagnostic.code), diagnostic.range.start.line)
            for diagnostic in diagnostics_for(head + SPEAK, "/tmp/speak.kedi")
            if str(diagnostic.code) == "K-ADAPTER-CAPABILITY"
        ]

    assert codes("> agent: claude\n") == [("K-ADAPTER-CAPABILITY", 2)]
    assert codes("> adapter: langchain\n") == []


def test_langchain_speech_stays_outside_the_conversation(workspace: Path) -> None:
    from langchain_core.messages import AIMessage

    source = (
        "> import: media\n"
        "> history: enabled\n"
        "[greeting] << Hello.\n"
        "[speech: Audio] << Say good morning.\n"
        "[answer] << What did I say first?\n"
        "= `speech`\n"
    )
    # Independent requests may run in any order, so the reply follows the prompt.
    model = _fake_langchain(
        profile={"audio_outputs": True},
        name="speaker",
        replies={
            "Hello.": AIMessage(content="Hi."),
            "Say good morning.": _wav_block_message(),
            "What did I say first?": AIMessage(content="Hello."),
        },
    )
    assert _run(source, LangChainAdapter(model=model)).data == WAV
    calls = {str(call[-1].content): call for call in _Recorded.calls}
    spoken, third = calls["Say good morning."], calls["What did I say first?"]
    # The audio model gets the prompt alone, and the next text turn sees no audio reply.
    assert [message.content for message in spoken] == ["Say good morning."]
    assert [
        (type(message).__name__, message.content)
        for message in third
        if type(message).__name__ != "SystemMessage"
    ] == [
        ("HumanMessage", "Hello."),
        ("AIMessage", "Hi."),
        ("HumanMessage", "What did I say first?"),
    ]


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
def test_speech_counts_against_the_request_budget(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, adapter_name: str
) -> None:
    record = _fake_realtime(monkeypatch)
    adapter = (
        PydanticAdapter(model=_live_model())
        if adapter_name == "pydantic"
        else LangChainAdapter(
            model=_fake_langchain(_wav_block_message(), profile={"audio_outputs": True}, name="s")
        )
    )
    source = SPEAK.replace("[speech", "> budget:\n    request_limit: 0\n[speech")
    with pytest.raises(KediExecutionError, match="request"):
        _run(source, adapter)
    assert "prompt" not in record
    if adapter_name == "langchain":
        assert _Recorded.calls == []


def test_a_speech_chat_model_with_a_voice_field_gets_the_voice(workspace: Path) -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    voices: list[str | None] = []

    class LocalSpeaker(GenericFakeChatModel):
        """A local TTS wrapped as a chat model, with voices of its own."""

        model: str = "local-tts"
        voice: str | None = None

        def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
            voices.append(self.voice)
            return ChatResult(generations=[ChatGeneration(message=_wav_block_message())])

    model = LocalSpeaker(messages=iter([]))
    model.profile = {"audio_outputs": True}
    source = SPEAK.replace("[speech", "> settings:\n    voice: zoey\n[speech")
    assert _run(source, LangChainAdapter(model=model)).data == WAV
    assert voices == ["zoey"]
    # The adapter's own model keeps its voice; only the speech request's copy changes.
    assert model.voice is None


def test_an_explicit_empty_instruction_sends_none() -> None:
    from kedi.agent_profile import AgentProfile

    profile = AgentProfile(system="Profile instructions")
    assert profile.call_instructions(None) == "Profile instructions"
    assert profile.call_instructions("") == ""
    assert profile.call_instructions("Call instructions") == "Call instructions"
