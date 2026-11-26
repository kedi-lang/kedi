"""Speak each English sentence, then transcribe it back, all locally and through Kedi.

    python examples/local_speech/roundtrip.py

Phonon 31m speaks through ``[speech: Audio] <<`` (speak.kedi); Cactus Whistle listens through
an audio prompt, ``<<clip>>`` (transcribe.kedi). The word error rate against the sentence is
an intelligibility check that needs no listener. Turkish has no local transcriber here yet.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import kedi
from kedi.agent_adapter import LangChainAdapter
from kedi.media import Audio

sys.path.insert(0, str(Path(__file__).parent))
from speech_models import Phonon, Whistle, load  # noqa: E402

HERE = Path(__file__).parent


@kedi.bind(file=HERE / "speak.kedi")
def speak(sentence: str) -> Audio: ...


@kedi.bind(file=HERE / "transcribe.kedi")
def transcribe(clip: Audio) -> str: ...


def words(text: str) -> list[str]:
    """Lowercase words without punctuation: "9:30" and "9.30" both become "930"."""

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
    sentences = (HERE / "sentences_en.txt").read_text(encoding="utf-8").splitlines()
    cold = {"phonon": load("en"), "whistle": load("whistle")}
    rows = []
    for sentence in sentences:
        with kedi.context(adapter=LangChainAdapter(Phonon()), model=None):
            clip = speak(sentence)
        started = time.perf_counter()
        with kedi.context(adapter=LangChainAdapter(Whistle(language="en")), model=None):
            heard = transcribe(clip)
        rows.append(
            {
                "sentence": sentence,
                "heard": heard,
                "word_error_rate": round(word_error_rate(words(sentence), words(heard)), 3),
                "transcription_s": round(time.perf_counter() - started, 3),
            }
        )
    mean = sum(row["word_error_rate"] for row in rows) / len(rows)
    report = {
        "cold_load_s": {k: round(v, 2) for k, v in cold.items()},
        "mean_wer": round(mean, 3),
        "rows": rows,
    }
    out = HERE / "out" / "roundtrip.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for row in rows:
        print(
            f"WER {row['word_error_rate']:.2f}  {row['sentence']}\n          heard: {row['heard']}"
        )
    print(f"Mean WER {mean:.3f} over {len(rows)} sentences. Report: {out}")


if __name__ == "__main__":
    main()
