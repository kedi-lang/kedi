"""Read every receipt image through Kedi with Mirage, and score it against the ground truth.

    python examples/local_vision/score.py

For each receipt: the character error rate of the writing Mirage read, whether the parsed
merchant, items and total match, and whether the program's add-up check agrees with the
truth (one receipt's printed total is wrong on purpose).
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import kedi
from kedi.agent_adapter import LangChainAdapter
from kedi.media import Image

sys.path.insert(0, str(Path(__file__).parent))
from vision_model import MirageReader, _mirage  # noqa: E402

HERE = Path(__file__).parent


@kedi.bind(file=HERE / "receipts.kedi")
def read_receipt(photo: Image) -> dict[str, Any]: ...


def character_error_rate(reference: str, hypothesis: str) -> float:
    previous = list(range(len(hypothesis) + 1))
    for index, char in enumerate(reference, start=1):
        current = [index]
        for position, seen in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[position] + 1, current[-1] + 1, previous[position - 1] + (char != seen)
                )
            )
        previous = current
    return previous[-1] / max(1, len(reference))


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def main() -> None:
    truth = json.loads((HERE / "receipts" / "truth.json").read_text())
    started = time.perf_counter()
    _mirage()
    cold = time.perf_counter() - started
    rows = []
    with kedi.context(adapter=LangChainAdapter(MirageReader()), model=None):
        for key, expected in truth.items():
            photo = Image.from_bytes((HERE / "receipts" / f"{key}.png").read_bytes(), "image/png")
            started = time.perf_counter()
            result = read_receipt(photo)
            seconds = time.perf_counter() - started
            receipt = result["receipt"]
            rows.append(
                {
                    "receipt": key,
                    "seconds": round(seconds, 3),
                    "cer": round(character_error_rate(expected["text"], result["writing"]), 3),
                    "merchant": receipt["merchant"] == expected["merchant"],
                    "items": receipt["items"] == expected["items"],
                    "total": receipt["total"] == expected["total"],
                    "check_agrees": receipt["adds_up"] == expected["adds_up"],
                    "writing": result["writing"],
                }
            )
    count = len(rows)
    report = {
        "model": "cactus mirage",
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "receipts": count,
        "mean_cer": round(sum(row["cer"] for row in rows) / count, 3),
        **{
            name: f"{sum(row[name] for row in rows)}/{count}"
            for name in ("merchant", "items", "total", "check_agrees")
        },
        "cold_load_s": round(cold, 2),
        "median_s": sorted(row["seconds"] for row in rows)[count // 2],
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "rows": rows,
    }
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for key in (
        "mean_cer",
        "merchant",
        "items",
        "total",
        "check_agrees",
        "cold_load_s",
        "median_s",
        "peak_rss_mb",
    ):
        print(f"{key}: {report[key]}")
    for row in rows:
        if not (row["merchant"] and row["items"] and row["total"] and row["check_agrees"]):
            print(f"{row['receipt']}: cer {row['cer']} read: {row['writing']!r}")


if __name__ == "__main__":
    main()
