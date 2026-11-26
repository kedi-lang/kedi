"""Score the refund program on the fixed set with a local model, or explicitly a cloud one.

    python examples/local_extraction/run.py                          # Cactus Needle 3, offline
    python examples/local_extraction/run.py --cloud google:gemini-3.5-flash-lite   # the same program, online

The cloud run is a separate, explicit comparison; it is never a fallback.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import kedi
from kedi.agent_adapter import PydanticAdapter

sys.path.insert(0, str(Path(__file__).parent))
from needle_model import NeedleModel  # noqa: E402

HERE = Path(__file__).parent
FIELDS = ("customer", "order_id", "amount", "reason")


@kedi.bind(file=HERE / "refunds.kedi")
def handle(message: str) -> dict[str, Any]: ...


def peak_rss_mb() -> float:
    """Peak resident memory of this process: ru_maxrss is bytes on macOS and KiB on Linux."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def score(case: dict[str, Any], result: Any) -> dict[str, Any]:
    """Field accuracy against explicit expectations; a failed call scores zero, not skipped."""

    if isinstance(result, Exception):
        # The Kedi error wraps the cause; report the cause, not the Kedi stack trace.
        cause = getattr(result, "original", None) or result
        message = str(cause).strip().splitlines()[-1] if str(cause).strip() else ""
        return {"valid": False, "error": f"{type(cause).__name__}: {message}"[:300], "fields": {}}
    refund = result["refund"].model_dump()
    fields = {name: refund[name] == case["expected"][name] for name in FIELDS}
    return {"valid": True, "refund": refund, "route": result["route"], "fields": fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cloud", help="a Pydantic AI model id, run explicitly for comparison")
    args = parser.parse_args()

    cases = [json.loads(line) for line in (HERE / "requests.jsonl").read_text().splitlines()]
    model: Any = args.cloud or NeedleModel()
    started = time.perf_counter()
    if args.cloud is None:
        from needle_model import _extract

        # Load the engine and weights before timing calls: cold load is reported separately.
        _extract(
            "warm up", {"name": "noop", "description": "", "parameters": {"type": "object"}}, "", 8
        )
    cold = time.perf_counter() - started
    rows = []
    with kedi.context(adapter=PydanticAdapter(model), model=None):
        for run in range(args.repeat):
            for index, case in enumerate(cases, start=1):
                started = time.perf_counter()
                try:
                    result: Any = handle(case["message"])
                except Exception as error:  # noqa: BLE001 - the report counts failures
                    result = error
                row = score(case, result) | {
                    "run": run + 1,
                    "case": index,
                    "seconds": round(time.perf_counter() - started, 3),
                }
                row["route_correct"] = row.get("route") == case["route"]
                rows.append(row)
    total = len(rows)
    field_hits = Counter(name for row in rows for name, hit in row["fields"].items() if hit)
    consistent = sum(
        len({json.dumps(row.get("refund"), sort_keys=True) for row in rows if row["case"] == index})
        == 1
        for index in range(1, len(cases) + 1)
    )
    warm = sorted(row["seconds"] for row in rows if row["run"] > 1)
    report = {
        "model": args.cloud or "cactus needle3",
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "cases": len(cases),
        "repeats": args.repeat,
        "schema_valid": f"{sum(row['valid'] for row in rows)}/{total}",
        "field_accuracy": {name: f"{field_hits[name]}/{total}" for name in FIELDS},
        "task_completion": f"{sum(row['route_correct'] and all(row['fields'].values()) for row in rows)}/{total}",
        "route_correct": f"{sum(row['route_correct'] for row in rows)}/{total}",
        "consistent_cases": f"{consistent}/{len(cases)}",
        "cold_load_s": round(cold, 2),
        "warm_median_s": warm[len(warm) // 2] if warm else None,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "rows": rows,
    }
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    name = "cloud" if args.cloud else "local"
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for key in (
        "model",
        "schema_valid",
        "field_accuracy",
        "task_completion",
        "route_correct",
        "consistent_cases",
        "cold_load_s",
        "warm_median_s",
        "peak_rss_mb",
    ):
        print(f"{key}: {report[key]}")
    for row in rows:
        if row["run"] == 1 and not (row["valid"] and all(row["fields"].values())):
            misses = [n for n, hit in row["fields"].items() if not hit]
            print(f"case {row['case']}: {row.get('error') or row['refund']} (wrong: {misses})")


if __name__ == "__main__":
    main()
