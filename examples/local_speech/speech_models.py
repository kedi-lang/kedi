"""Local speech models as LangChain chat models, for Kedi's speech surfaces.

Speech output is ``[speech: Audio] << text`` (#190). On LangChain it works with any chat model
whose profile declares ``audio_outputs``: the model reads the user turn aloud and puts the WAV
bytes in its reply. Transcription is a prompt that references an audio clip, ``<<clip>>``,
sent to a model whose profile declares ``audio_inputs``. These classes run offline:

- ``EmaLightning``: Turkish speech, EMA Lightning (Apache 2.0, 48 kHz), ``pip install
  ema-lightning``;
- ``Phonon``: English speech, Phonon 31m (Apache 2.0, 24 kHz), ``pip install ptts``;
- ``Whistle``: transcription in English and six other languages, Cactus Whistle (Apache 2.0),
  ``pip install cactus-compute``. Turkish is not one of them;
- ``Whisper``: transcription in Turkish (and Whisper's other languages), Whisper small through
  faster-whisper (MIT), ``pip install faster-whisper``.

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
WHISPER_MODEL = ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120")

LEADING_SILENCE = 8_000
"""Half a second at 16 kHz, prepended before Whisper transcribes."""

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
        if self.voice is not None:
            raise ValueError(
                f"EMA Lightning has one voice; remove the voice setting {self.voice!r}"
            )
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


class Transcriber(BaseChatModel):
    """Transcribes the audio clip in the last user turn; ``language`` skips detection."""

    # No tool calling: Kedi then offers no tools, and binds one only for a program tool or a
    # typed capture, which a transcriber refuses.
    profile: Any = {"audio_inputs": True, "tool_calling": False, "structured_output": False}
    language: str | None = None

    def bind_tools(self, tools: Any, **_: Any) -> Transcriber:
        raise NotImplementedError(
            f"{self._llm_type} answers with what it hears: it takes no tools and no typed "
            "output, so capture untyped text, for example [text] << Transcribe <<clip>>."
        )

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
            raise ValueError(f"{self._llm_type} transcribes one audio clip, got {len(clips)}")
        samples = _resample(*read_wav(base64.b64decode(clips[0]["base64"])), 16_000)
        text = self.transcribe(samples).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def transcribe(self, samples: np.ndarray) -> str:
        raise NotImplementedError


class Whistle(Transcriber):
    """English (and de, fr, es, it, nl, pl) transcription with Cactus Whistle."""

    @property
    def _llm_type(self) -> str:
        return "cactus-whistle"

    def transcribe(self, samples: np.ndarray) -> str:
        return _whistle().transcribe(samples, language=self.language)["text"]


class Whisper(Transcriber):
    """Turkish transcription with Whisper small (faster-whisper, int8 on the CPU)."""

    @property
    def _llm_type(self) -> str:
        return "whisper-small"

    def transcribe(self, samples: np.ndarray) -> str:
        # Whisper drops the first word of a clip that starts with speech, as EMA's clips do
        # ("Hesabınıza", "Şu an"); half a second of leading silence keeps it.
        padded = np.concatenate([np.zeros(LEADING_SILENCE, np.float32), samples])
        segments, _ = _whisper().transcribe(padded, language=self.language, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments)


@cache
def _whistle() -> Any:
    from cactus import Whistle as CactusWhistle

    return CactusWhistle()


@cache
def _whisper() -> Any:
    from faster_whisper import WhisperModel

    name, revision = WHISPER_MODEL
    return WhisperModel(name, device="cpu", compute_type="int8", revision=revision)


@cache
def _ema() -> Any:
    from ema_lightning import EMA

    return EMA()


@cache
def _phonon() -> Any:
    import ptts

    return ptts.TTS(PHONON_MODEL, lang="en")


def load(model: str) -> float:
    """Load a model's weights now (``tr``, ``en``, ``whistle`` or ``whisper``); the seconds."""

    started = time.perf_counter()
    {"tr": _ema, "en": _phonon, "whistle": _whistle, "whisper": _whisper}[model]()
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
