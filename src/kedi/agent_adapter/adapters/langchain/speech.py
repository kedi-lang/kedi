"""Speech output on LangChain: ``[speech: Audio] << Read this aloud.`` (#190).

Audio chat models speak their reply: Gemini TTS models (``audio_outputs`` in the model profile)
read the prompt aloud, and OpenAI audio models (``gpt-audio``, ``gpt-4o-audio-preview``) answer
with audio when asked for it.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from kedi.media import Audio, MediaError


def speech_model(model: Any, *, voice: str | None, label: str) -> Any:
    """A copy of ``model`` that answers with speech, or a clear error when it cannot."""

    # A streamed reply would come as audio chunks; speech is one value, so it is not streamed.
    updates: dict[str, Any] = {"disable_streaming": True}
    if _is_openai_audio_model(model, label):
        model_kwargs = dict(getattr(model, "model_kwargs", None) or {})
        model_kwargs["modalities"] = ["text", "audio"]
        model_kwargs["audio"] = {"voice": voice or "alloy", "format": "wav"}
        updates["model_kwargs"] = model_kwargs
    elif not _speaks(model, label):
        raise MediaError(
            f"Model {label} cannot produce speech; use an audio model such as "
            "google_genai:gemini-2.5-flash-preview-tts or openai:gpt-audio"
        )
    elif voice is not None:
        raise MediaError(
            f"Model {label} takes no voice setting through LangChain; remove the voice setting "
            "or use Pydantic AI with a realtime model"
        )
    return model.model_copy(update=updates)


def reads_aloud(label: str) -> bool:
    """Whether the model reads its prompt aloud, as Gemini TTS models do.

    Such a model takes no system instructions, so Kedi sends it the prompt alone.
    """

    return label.endswith("-tts") or "-tts-" in label


def speech_response(result: Any, label: str) -> Audio:
    """The speech in the final message, as an audio value."""

    messages = result.get("messages") if isinstance(result, Mapping) else None
    message = messages[-1] if isinstance(messages, (list, tuple)) and messages else result
    for block in reversed(getattr(message, "content_blocks", None) or []):
        if block.get("type") == "audio" and block.get("base64"):
            return Audio.from_bytes(
                base64.b64decode(block["base64"]), block.get("mime_type") or "audio/wav"
            )
    audio = (getattr(message, "additional_kwargs", None) or {}).get("audio")
    if isinstance(audio, bytes):
        return Audio.from_bytes(audio, "audio/wav")
    if isinstance(audio, Mapping) and isinstance(audio.get("data"), str):
        # OpenAI's audio reply: WAV, as speech_model asks, with the model's transcript.
        transcript = audio.get("transcript")
        return Audio.from_bytes(base64.b64decode(audio["data"]), "audio/wav").with_transcript(
            transcript if isinstance(transcript, str) and transcript else None
        )
    raise MediaError(f"Model {label} returned no speech")


def _speaks(model: Any, label: str) -> bool:
    audio_outputs = (getattr(model, "profile", None) or {}).get("audio_outputs")
    # Newer Gemini TTS models have no profile yet; LangChain's Gemini client asks them for
    # audio by the same name rule.
    return audio_outputs is True or (
        audio_outputs is None
        and type(model).__module__.startswith("langchain_google_genai")
        and reads_aloud(label)
    )


def _is_openai_audio_model(model: Any, label: str) -> bool:
    return type(model).__module__.startswith("langchain_openai") and "audio" in label
