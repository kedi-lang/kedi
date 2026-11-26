"""Speak each sentence, then transcribe it back, all locally and through Kedi.

    python examples/local_speech/roundtrip.py              # English: Phonon 31m, Cactus Whistle
    python examples/local_speech/roundtrip.py --lang tr    # Turkish: EMA Lightning, Whisper small

The speech model speaks through ``[speech: Audio] <<`` (speak.kedi); the transcriber listens
through an audio prompt, ``<<clip>>`` (transcribe.kedi). The word error rate against the
sentence is an intelligibility check that needs no listener.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import kedi
from kedi.agent_adapter import LangChainAdapter
from kedi.media import Audio

sys.path.insert(0, str(Path(__file__).parent))
from speech_models import EmaLightning, Phonon, Whisper, Whistle, load  # noqa: E402

HERE = Path(__file__).parent

PAIRS = {
    "en": (Phonon, "en", Whistle, "whistle"),
    "tr": (EmaLightning, "tr", Whisper, "whisper"),
}


@kedi.bind(file=HERE / "speak.kedi")
def speak(sentence: str) -> Audio: ...


@kedi.bind(file=HERE / "transcribe.kedi")
def transcribe(clip: Audio) -> str: ...


def words(text: str, lang: str = "en") -> list[str]:
    """Lowercase words without punctuation: "9:30" and "9.30" both become "930".

    In Turkish, İ and I lower to i and ı, and the optional circumflex is dropped, so
    "İzmir'e" becomes "izmire" and "rüzgâr" becomes "rüzgar".
    """

    if lang == "tr":
        text = text.replace("İ", "i").replace("I", "ı").translate(str.maketrans("âîûÂÎÛ", "aiuaiu"))
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, start=1):
        current = [index]
        for position, heard in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[position] + 1,
                    current[-1] + 1,
                    previous[position - 1] + (word != heard),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(PAIRS), default="en")
    lang = parser.parse_args().lang
    speaker, speaker_key, listener, listener_key = PAIRS[lang]
    sentences = (HERE / f"sentences_{lang}.txt").read_text(encoding="utf-8").splitlines()
    cold = {speaker_key: load(speaker_key), listener_key: load(listener_key)}
    rows = []
    for sentence in sentences:
        with kedi.context(adapter=LangChainAdapter(speaker()), model=None):
            clip = speak(sentence)
        started = time.perf_counter()
        with kedi.context(adapter=LangChainAdapter(listener(language=lang)), model=None):
            heard = transcribe(clip)
        rows.append(
            {
                "sentence": sentence,
                "heard": heard,
                "word_error_rate": round(
                    word_error_rate(words(sentence, lang), words(heard, lang)), 3
                ),
                "transcription_s": round(time.perf_counter() - started, 3),
            }
        )
    mean = sum(row["word_error_rate"] for row in rows) / len(rows)
    report = {
        "cold_load_s": {k: round(v, 2) for k, v in cold.items()},
        "mean_wer": round(mean, 3),
        "rows": rows,
    }
    out = HERE / "out" / f"roundtrip_{lang}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for row in rows:
        print(
            f"WER {row['word_error_rate']:.2f}  {row['sentence']}\n          heard: {row['heard']}"
        )
    print(f"Mean WER {mean:.3f} over {len(rows)} sentences. Report: {out}")


if __name__ == "__main__":
    main()
