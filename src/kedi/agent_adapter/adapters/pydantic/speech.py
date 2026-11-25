"""Speech output on Pydantic AI: ``[speech: Audio] << Read this aloud.`` (#190).

Pydantic AI's request models return no audio; its realtime models do. A speech request opens a
one-shot realtime session: it sends the prompt as one text turn, waits for the reply, keeps the
reply's audio and transcript, and closes the session. The session takes no tools.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import TYPE_CHECKING, Any, cast

from kedi.agent_adapter.invocation import InvocationUsage, report_invocation
from kedi.media import Audio, MediaError
from kedi.run_budget import admit_run_operation

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.realtime import RealtimeModel
    from pydantic_ai.realtime.settings import RealtimeModelSettings

# Pydantic AI request model systems and model id prefixes, as realtime providers.
_REALTIME_PROVIDERS = {
    "google": "google",
    "google-gla": "google",
    "google-vertex": "google-cloud",
    "google-cloud": "google-cloud",
    "openai": "openai",
    "azure": "azure",
    "xai": "xai",
    "elevenlabs": "elevenlabs",
}
_DEFAULT_TIMEOUT_SECONDS = 120.0


async def speak(
    prompt: str,
    *,
    model: Model | str,
    instructions: str | None,
    voice: str | None,
    timeout: float | None,
) -> Audio:
    """Ask a realtime model to speak its reply to ``prompt`` and return the speech."""

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, SpeechPart
    from pydantic_ai.realtime import RealtimeError
    from pydantic_ai.usage import RunUsage

    realtime = realtime_model(model)
    label = f"{realtime.system}:{realtime.model_name}"
    # The session's reply is one model request for the run budget.
    admit_run_operation("request")
    usage = RunUsage()
    bound = Agent(instructions=instructions).realtime(
        realtime,
        model_settings=_voice_settings(realtime, voice),
        usage=usage,
    )
    try:
        async with bound.session(audio_retention="output_audio") as session:
            await session.send(prompt)
            await asyncio.wait_for(session.wait_for_reply(), timeout or _DEFAULT_TIMEOUT_SECONDS)
            speech = [
                part
                for message in session.new_messages()
                if isinstance(message, ModelResponse)
                for part in message.parts
                if isinstance(part, SpeechPart) and part.audio is not None
            ]
    except RealtimeError as exc:
        raise MediaError(
            f"Model {label} cannot produce speech: {exc}. Speech comes from realtime models "
            "such as google:gemini-3.1-flash-live-preview or openai:gpt-realtime"
        ) from exc
    except TimeoutError as exc:
        raise MediaError(f"Model {label} gave no speech within the timeout") from exc
    finally:
        report_invocation(
            InvocationUsage(
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=usage.cost,
            )
        )
    if not speech:
        raise MediaError(f"Model {label} returned no speech")
    transcript = " ".join(part.transcript for part in speech if part.transcript) or None
    data = join_wav([part.audio.data for part in speech if part.audio is not None])
    return Audio.from_bytes(data, "audio/wav").with_transcript(transcript)


def realtime_model(model: Model | str) -> RealtimeModel:
    """The realtime model for the selected model, with the same credentials."""

    from pydantic_ai.realtime import infer_realtime_model

    if isinstance(model, str):
        prefix, separator, name = model.partition(":")
        return infer_realtime_model(f"{_REALTIME_PROVIDERS.get(prefix, prefix)}{separator}{name}")
    kind = _REALTIME_PROVIDERS.get(model.system)
    if kind is None:
        raise MediaError(
            f"Model {model.system}:{model.model_name} cannot produce speech; Pydantic AI speaks "
            "through Google, OpenAI, Azure, xAI and ElevenLabs realtime models"
        )
    provider = getattr(model, "provider", None)
    return infer_realtime_model(
        f"{kind}:{model.model_name}",
        provider_factory=None if provider is None else lambda _: provider,
    )


def _voice_settings(model: RealtimeModel, voice: str | None) -> RealtimeModelSettings | None:
    if voice is None:
        return None
    from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel
    from pydantic_ai.realtime.google import GoogleRealtimeModel
    from pydantic_ai.realtime.xai import XaiRealtimeModel

    # ElevenLabs' model is an OpenAI-protocol model, so it is matched first.
    for model_class, key in (
        (ElevenLabsRealtimeModel, "elevenlabs_voice"),
        (GoogleRealtimeModel, "google_voice"),
        (XaiRealtimeModel, "xai_voice"),
    ):
        if isinstance(model, model_class):
            return cast("RealtimeModelSettings", {key: voice})
    return cast("RealtimeModelSettings", {"openai_voice": voice})


def join_wav(chunks: list[bytes]) -> bytes:
    """One WAV file from WAV files of the same format, in order."""

    if len(chunks) == 1:
        return chunks[0]
    frames: list[bytes] = []
    params: Any = None
    for chunk in chunks:
        with wave.open(io.BytesIO(chunk)) as reader:
            current = reader.getparams()[:3]
            if params is not None and current != params:
                raise MediaError("The model's speech came in WAV parts of different formats")
            params = current
            frames.append(reader.readframes(reader.getnframes()))
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(params[0])
        writer.setsampwidth(params[1])
        writer.setframerate(params[2])
        writer.writeframes(b"".join(frames))
    return output.getvalue()
