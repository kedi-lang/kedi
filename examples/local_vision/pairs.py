"""Run same_purchase.kedi on pairs of receipts, locally or explicitly on a cloud model.

    python examples/local_vision/pairs.py --local http://127.0.0.1:8092/v1
    python examples/local_vision/pairs.py --cloud google:gemini-3.5-flash-lite --delay 5

Each pair's expected answer comes from receipts/truth.json. Three pairs give the same receipt
twice: Kedi attaches it once and the prompt names it twice, so the model must recognize one
image under both names.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import kedi
from kedi.agent_adapter import PydanticAdapter
from kedi.media import Image

sys.path.insert(0, str(Path(__file__).parent))
from compare import CountingModel, score_failure  # noqa: E402

HERE = Path(__file__).parent
PAIRS = (
    ("cafe", "cafe"),
    ("cafe", "books"),
    ("pharmacy", "pharmacy"),
    ("garage", "hardware"),
    ("bakery", "bakery"),
    ("books", "bakery"),
)


@kedi.bind(file=HERE / "same_purchase.kedi")
def duplicate(first: Image, second: Image) -> dict[str, Any]: ...


def photo(key: str) -> Image:
    return Image.from_bytes((HERE / "receipts" / f"{key}.png").read_bytes(), "image/png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--local", metavar="URL", help="an OpenAI-compatible server's /v1 URL")
    backend.add_argument("--cloud", help="a Pydantic AI model id, run explicitly for comparison")
    parser.add_argument("--local-model", default="qwen3-vl-2b", help="the served model name")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    args = parser.parse_args()

    if args.local:
        provider = OpenAIProvider(base_url=args.local, api_key="local")
        model = CountingModel(OpenAIChatModel(args.local_model, provider=provider))
    else:
        model = CountingModel(args.cloud)
    truth = json.loads((HERE / "receipts" / "truth.json").read_text())
    rows = []
    with kedi.context(adapter=PydanticAdapter(model), model=None):
        for run in range(args.repeat):
            for first, second in PAIRS:
                if rows and args.delay:
                    time.sleep(args.delay)
                expected = {
                    "same_shop": truth[first]["merchant"] == truth[second]["merchant"],
                    "same_total": truth[first]["total"] == truth[second]["total"],
                }
                started = time.perf_counter()
                try:
                    result = duplicate(photo(first), photo(second))
                except Exception as error:  # noqa: BLE001 - the report counts failures
                    row = score_failure(error) | {"fields": dict.fromkeys(expected, False)}
                else:
                    match = result["match"].model_dump()
                    row = {
                        "valid": True,
                        "match": match,
                        "fields": {name: match[name] == value for name, value in expected.items()},
                    }
                rows.append(
                    row
                    | {
                        "run": run + 1,
                        "pair": f"{first}/{second}",
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
    total = len(rows)
    answered = [row for row in rows if not row.get("runtime_failure")]
    times = sorted(row["seconds"] for row in answered)
    answers = {
        pair: {
            json.dumps(row.get("match"), sort_keys=True) for row in answered if row["pair"] == pair
        }
        for pair in {row["pair"] for row in rows}
    }
    report = {
        "model": args.cloud or f"{args.local_model} at {args.local}",
        "pairs": len(PAIRS),
        "repeats": args.repeat,
        "runtime_failures": f"{total - len(answered)}/{total}",
        "schema_valid": f"{sum(row['valid'] for row in rows)}/{total}",
        "same_shop": f"{sum(row['fields']['same_shop'] for row in rows)}/{total}",
        "same_total": f"{sum(row['fields']['same_total'] for row in rows)}/{total}",
        "both_right": f"{sum(all(row['fields'].values()) for row in rows)}/{total}",
        "consistent_pairs": f"{sum(len(found) == 1 for found in answers.values())}/{len(answers)}",
        "median_s": times[len(times) // 2] if times else None,
        "rows": rows,
    }
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    name = "pairs_cloud" if args.cloud else "pairs_local"
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for key in (
        "model",
        "runtime_failures",
        "schema_valid",
        "same_shop",
        "same_total",
        "both_right",
        "consistent_pairs",
        "median_s",
    ):
        print(f"{key}: {report[key]}")
    for row in rows:
        if row["run"] == 1 and not all(row["fields"].values()):
            print(f"{row['pair']}: {row.get('error') or row['match']}")


if __name__ == "__main__":
    main()
