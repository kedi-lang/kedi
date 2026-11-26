"""Speak the fixed sentence set through Kedi with a local model, and measure it.

    python examples/local_speech/run.py --lang tr     # EMA Lightning, Turkish
    python examples/local_speech/run.py --lang en     # Phonon 31m, English

Recordings and measurements go to ``examples/local_speech/out/<lang>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import resource
import sys
import time
import wave
from pathlib import Path

import kedi
from kedi.agent_adapter import LangChainAdapter
from kedi.media import Audio

sys.path.insert(0, str(Path(__file__).parent))
from speech_models import SYNTHESIS_SECONDS, EmaLightning, Phonon, load  # noqa: E402

HERE = Path(__file__).parent


@kedi.bind(file=HERE / "speak.kedi")
def speak(sentence: str) -> Audio: ...


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("tr", "en"), required=True)
    parser.add_argument("--repeat", type=int, default=2, help="runs over the whole set")
    args = parser.parse_args()

    sentences = (HERE / f"sentences_{args.lang}.txt").read_text(encoding="utf-8").splitlines()
    out = HERE / "out" / args.lang
    out.mkdir(parents=True, exist_ok=True)
    cold_load = load(args.lang)
    model = EmaLightning() if args.lang == "tr" else Phonon()
    rows = []
    with kedi.context(adapter=LangChainAdapter(model), model=None):
        for run in range(args.repeat):
            for index, sentence in enumerate(sentences, start=1):
                started = time.perf_counter()
                audio = speak(sentence)
                call = time.perf_counter() - started
                assert audio.data is not None and audio.media_type == "audio/wav"
                (out / f"{index:02}.wav").write_bytes(audio.data)
                with wave.open(io.BytesIO(audio.data)) as recording:
                    rate = recording.getframerate()
                    duration = recording.getnframes() / rate
                rows.append(
                    {
                        "run": run + 1,
                        "sentence": index,
                        "text": sentence,
                        "sample_rate": rate,
                        "duration_s": round(duration, 3),
                        "synthesis_s": round(SYNTHESIS_SECONDS[-1], 3),
                        "kedi_call_s": round(call, 3),
                        "real_time_factor": round(SYNTHESIS_SECONDS[-1] / duration, 3),
                        "sha256": hashlib.sha256(audio.data).hexdigest()[:16],
                    }
                )
    repeatable = all(
        len({row["sha256"] for row in rows if row["sentence"] == index}) == 1
        for index in range(1, len(sentences) + 1)
    )
    report = {
        "language": args.lang,
        "model": type(model).__name__,
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "cold_load_s": round(cold_load, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        "repeatable_with_seed": repeatable,
        "recordings": rows,
    }
    (out / "measurements.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"{type(model).__name__}: cold load {report['cold_load_s']} s, peak RSS {report['peak_rss_mb']} MB"
    )
    for row in rows:
        print(
            f"run {row['run']} #{row['sentence']}: {row['duration_s']:5.2f} s audio, "
            f"synthesis {row['synthesis_s']:.2f} s (RTF {row['real_time_factor']:.2f}), "
            f"Kedi call {row['kedi_call_s']:.2f} s"
        )
    print(f"Same audio on every run: {repeatable}. Recordings: {out}")


if __name__ == "__main__":
    main()
