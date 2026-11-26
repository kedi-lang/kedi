"""Local speech models as LangChain chat models, for Kedi's speech surfaces.

Speech output is ``[speech: Audio] << text`` (#190). On LangChain it works with any chat model
whose profile declares ``audio_outputs``: the model reads the user turn aloud and puts the WAV
bytes in its reply. Transcription is a prompt that references an audio clip, ``<<clip>>``,
sent to a model whose profile declares ``audio_inputs``. These classes run offline:

- ``EmaLightning``: Turkish speech, EMA Lightning (Apache 2.0, 48 kHz), ``pip install
  ema-lightning``;
- ``Phonon``: English speech, Phonon 31m (Apache 2.0, 24 kHz), ``pip install ptts``;
- ``Whistle``: transcription in English and six other languages, Cactus Whistle (Apache 2.0),
  ``pip install cactus-compute``. Turkish is not one of them.

Each loads its weights once per process. Provision them with ``provision.py`` while online;
later runs need no network.
"""

from __future__ import annotations

import base64
import io
import os
import time
import wave
from functools import cache
from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

PHONON_MODEL = "hf://gradium/phonon-31m-en@92741c5b07d085ff840e26fea8ce43110b87852f"

SYNTHESIS_SECONDS: list[float] = []
"""How long each synthesis took, for the measurements."""

# Cactus reports usage to its servers unless told not to; these demos never send anything.
os.environ["CACTUS_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"


class LocalSpeech(BaseChatModel):
    """Reads the last user turn aloud; ``seed`` makes the audio repeatable."""

    profile: Any = {"audio_outputs": True}
    voice: str | None = None
    seed: int | None = 7

    @property
    def _llm_type(self) -> str:
        return "local-speech"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **_: Any
    ) -> ChatResult:
        text = _last_user_text(messages)
        started = time.perf_counter()
        samples, rate = self.synthesize(text)
        SYNTHESIS_SECONDS.append(time.perf_counter() - started)
        message = AIMessage(content="", additional_kwargs={"audio": wav_bytes(samples, rate)})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class EmaLightning(LocalSpeech):
    """Turkish speech with EMA Lightning, 48 kHz."""

    @property
    def _llm_type(self) -> str:
        return "ema-lightning"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        speech = _ema().say(text, seed=self.seed)
        return speech.audio, speech.sample_rate


class Phonon(LocalSpeech):
    """English speech with Phonon 31m, 24 kHz."""

    @property
    def _llm_type(self) -> str:
        return "phonon-31m"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        tts = _phonon()
        return np.asarray(tts.synth(text, voice=self.voice, seed=self.seed)), tts.sample_rate


class Whistle(BaseChatModel):
    """Transcribes the audio in the last user turn; ``language`` skips detection."""

    profile: Any = {"audio_inputs": True}
    language: str | None = None

    @property
    def _llm_type(self) -> str:
        return "cactus-whistle"

    def bind_tools(self, tools: Any, **_: Any) -> Whistle:
        # A transcriber answers with what it hears and calls no tools, so Kedi's are unused.
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **_: Any
    ) -> ChatResult:
        message = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        clips = [
            part
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "audio"
        ]
        if len(clips) != 1:
            raise ValueError(f"Whistle transcribes one audio clip per prompt, got {len(clips)}")
        samples = _resample(*read_wav(base64.b64decode(clips[0]["base64"])), 16_000)
        text = _whistle().transcribe(samples, language=self.language)["text"].strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


@cache
def _whistle() -> Any:
    from cactus import Whistle as CactusWhistle

    return CactusWhistle()


@cache
def _ema() -> Any:
    from ema_lightning import EMA

    return EMA()


@cache
def _phonon() -> Any:
    import ptts

    return ptts.TTS(PHONON_MODEL, lang="en")


def load(model: str) -> float:
    """Load a model's weights now (``tr``, ``en`` or ``whistle``); returns the seconds."""

    started = time.perf_counter()
    {"tr": _ema, "en": _phonon, "whistle": _whistle}[model]()
    return time.perf_counter() - started


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    """Mono float samples in [-1, 1] as 16-bit PCM WAV."""

    pcm = (np.clip(samples.reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV bytes as mono float samples and their rate."""

    with wave.open(io.BytesIO(data)) as recording:
        if recording.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        rate, channels = recording.getframerate(), recording.getnchannels()
        pcm = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2")
    return pcm.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768, rate


def _resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return samples
    count = int(len(samples) * target / rate)
    positions = np.linspace(0, len(samples) - 1, count)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def _last_user_text(messages: list[BaseMessage]) -> str:
    message = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
    if isinstance(message.content, str):
        return message.content
    return " ".join(
        part.get("text", "") for part in message.content if isinstance(part, dict)
    ).strip()
