"""The local speech example (#198): speech and transcription through Kedi with local models.

The speech engines are optional and not installed in CI, so they are replaced with tiny
fakes here. Everything Kedi does is real: the programs, ``[speech: Audio] <<``, the audio
prompt ``<<clip>>``, and the LangChain adapter's speech and media paths.
"""

from __future__ import annotations

import asyncio
import io
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kedi.agent_adapter import LangChainAdapter
from kedi.lang import compile_program, parse_program

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "local_speech"


@pytest.fixture
def speech_models(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    sys.modules.pop("speech_models", None)
    import speech_models

    heard: list[np.ndarray] = []

    class FakeEma:
        def say(self, text: str, seed: int | None = None) -> Any:
            return SimpleNamespace(audio=np.full(4800, 0.25, np.float32), sample_rate=48_000)

    class FakeWhistle:
        def transcribe(self, samples: np.ndarray, language: str | None = None) -> dict:
            heard.append(samples)
            return {"text": f" heard {len(samples)} samples in {language} "}

    class FakeWhisper:
        def transcribe(self, samples: np.ndarray, language: str | None, beam_size: int) -> Any:
            heard.append(samples)
            segments = [SimpleNamespace(text=" Hesabınıza "), SimpleNamespace(text=" iade edildi.")]
            return iter(segments), None

    monkeypatch.setattr(speech_models, "_ema", FakeEma)
    monkeypatch.setattr(speech_models, "_whistle", FakeWhistle)
    monkeypatch.setattr(speech_models, "_whisper", FakeWhisper)
    speech_models.heard = heard
    return speech_models


def run(source: str, adapter: Any, **values: Any) -> Any:
    runtime = compile_program(
        parse_program(source.read_text(), source_path=str(source)),
        adapter=adapter,
        runtime_globals=values,
    )
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def test_speak_returns_a_wav_from_the_local_speech_model(speech_models: Any) -> None:
    adapter = LangChainAdapter(speech_models.EmaLightning())
    speech = run(EXAMPLE / "speak.kedi", adapter, sentence="Toplantı yarın 09.30'da.")
    assert speech.media_type == "audio/wav"
    with wave.open(io.BytesIO(speech.data)) as recording:
        assert (recording.getframerate(), recording.getnframes()) == (48_000, 4800)
    assert speech_models.SYNTHESIS_SECONDS


def test_transcribe_hears_the_clip_resampled_to_16_khz(speech_models: Any) -> None:
    clip = run(EXAMPLE / "speak.kedi", LangChainAdapter(speech_models.EmaLightning()), sentence="x")
    adapter = LangChainAdapter(speech_models.Whistle(language="tr"))
    text = run(EXAMPLE / "transcribe.kedi", adapter, clip=clip)
    assert text == "heard 1600 samples in tr"
    (samples,) = speech_models.heard
    assert samples.dtype == np.float32 and abs(float(samples.mean()) - 0.25) < 1e-3


def test_whisper_hears_the_clip_after_leading_silence(speech_models: Any) -> None:
    clip = run(EXAMPLE / "speak.kedi", LangChainAdapter(speech_models.EmaLightning()), sentence="x")
    adapter = LangChainAdapter(speech_models.Whisper(language="tr"))
    assert run(EXAMPLE / "transcribe.kedi", adapter, clip=clip) == "Hesabınıza iade edildi."
    (samples,) = speech_models.heard
    silence = speech_models.LEADING_SILENCE
    assert len(samples) == silence + 1600 and not samples[:silence].any()


@pytest.mark.parametrize("transcriber", ["Whisper", "Whistle"])
@pytest.mark.parametrize(
    "source",
    [
        "> import: media\n[n: int] << Count the words in <<clip>>.\n= `n`\n",
        "> import: media\n@shout(text: str) -> str:\n  > tool:\n      risk: read_only\n"
        "  = `text.upper()`\n> use: shout\n[t] << Transcribe <<clip>>.\n= <t>\n",
    ],
    ids=["typed capture", "program tool"],
)
def test_a_transcriber_refuses_typed_output_and_tools(
    speech_models: Any, transcriber: str, source: str
) -> None:
    clip = run(EXAMPLE / "speak.kedi", LangChainAdapter(speech_models.EmaLightning()), sentence="x")
    runtime = compile_program(
        parse_program(source),
        adapter=LangChainAdapter(getattr(speech_models, transcriber)(language="en")),
        runtime_globals={"clip": clip},
    )
    try:
        with pytest.raises(Exception, match="no tools and no typed output"):
            runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
    assert speech_models.heard == []


def test_round_trip_scores_words_without_punctuation() -> None:
    sys.path.insert(0, str(EXAMPLE))
    try:
        import roundtrip
    finally:
        sys.path.remove(str(EXAMPLE))
    reference = roundtrip.words("The meeting starts at 9:30 on the 3rd floor.")
    assert reference[4] == "930"
    assert roundtrip.word_error_rate(
        reference, roundtrip.words("The meeting starts at 9.30 on the third floor.")
    ) == pytest.approx(1 / 9)
    assert roundtrip.words("Tren İzmir'e, rüzgâr ILIK.", "tr") == [
        "tren",
        "izmire",
        "rüzgar",
        "ılık",
    ]


def test_ema_lightning_refuses_a_voice_it_does_not_have(speech_models: Any) -> None:
    with pytest.raises(ValueError, match="EMA Lightning has one voice"):
        speech_models.EmaLightning(voice="zeynep").synthesize("Merhaba")


def test_peak_memory_is_reported_in_megabytes(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(EXAMPLE))
    try:
        import run as speech_run
    finally:
        sys.path.remove(str(EXAMPLE))
    usage = SimpleNamespace(ru_maxrss=512 * 2**20)
    monkeypatch.setattr(speech_run.resource, "getrusage", lambda who: usage)
    monkeypatch.setattr(speech_run.sys, "platform", "darwin")
    assert speech_run.peak_rss_mb() == 512
    usage.ru_maxrss = 512 * 2**10
    monkeypatch.setattr(speech_run.sys, "platform", "linux")
    assert speech_run.peak_rss_mb() == 512
