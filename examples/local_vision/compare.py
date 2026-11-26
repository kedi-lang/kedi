"""Run receipt_record.kedi on a local vision-language model, or explicitly a cloud one.

    python examples/local_vision/compare.py --local http://127.0.0.1:8092/v1
    python examples/local_vision/compare.py --cloud google:gemini-3.5-flash-lite --delay 5

The local model is any OpenAI-compatible server with image input, such as llama-server with
Qwen3-VL-2B (see the README). The program, the images, the Receipt type and the scoring are
the same for both; only the adapter's model changes. The cloud run is a separate, explicit
comparison, never a fallback.

A runtime failure, such as a quota error, is reported apart from the model's answers:
accuracy counts every case, consistency and latency only the answered ones.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.openai import OpenAIProvider

import kedi
from kedi.agent_adapter import PydanticAdapter
from kedi.media import Image

HERE = Path(__file__).parent
FIELDS = ("merchant", "items", "total", "check_agrees")


@kedi.bind(file=HERE / "receipt_record.kedi")
def read_receipt(photo: Image) -> dict[str, Any]: ...


class CountingModel(WrapperModel):
    """Counts model requests: more than one per case means Pydantic AI asked again."""

    requests = 0

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        CountingModel.requests += 1
        return await super().request(*args, **kwargs)


def score_failure(error: Exception) -> dict[str, Any]:
    """A failed case: the cause, and whether it was a runtime failure rather than an answer."""

    cause = getattr(error, "original", None) or error
    message = str(cause).strip().splitlines()[-1] if str(cause).strip() else ""
    return {
        "valid": False,
        "runtime_failure": isinstance(cause, ModelAPIError),
        "error": f"{type(cause).__name__}: {message}"[:300],
    }


def score(expected: dict[str, Any], result: Any) -> dict[str, Any]:
    if isinstance(result, Exception):
        return score_failure(result) | {"fields": dict.fromkeys(FIELDS, False)}
    receipt = result["receipt"].model_dump()
    fields = {
        "merchant": receipt["merchant"] == expected["merchant"],
        "items": receipt["items"] == expected["items"],
        "total": receipt["total"] == expected["total"],
        "check_agrees": result["adds_up"] == expected["adds_up"],
    }
    return {"valid": True, "receipt": receipt, "adds_up": result["adds_up"], "fields": fields}


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
            for key, expected in truth.items():
                if rows and args.delay:
                    time.sleep(args.delay)
                photo = Image.from_bytes(
                    (HERE / "receipts" / f"{key}.png").read_bytes(), "image/png"
                )
                before = CountingModel.requests
                started = time.perf_counter()
                try:
                    result: Any = read_receipt(photo)
                except Exception as error:  # noqa: BLE001 - the report counts failures
                    result = error
                rows.append(
                    score(expected, result)
                    | {
                        "run": run + 1,
                        "receipt_id": key,
                        "seconds": round(time.perf_counter() - started, 3),
                        "requests": CountingModel.requests - before,
                    }
                )
    total = len(rows)
    answered = [row for row in rows if not row.get("runtime_failure")]
    hits = Counter(name for row in rows for name, hit in row["fields"].items() if hit)
    answers = {
        key: {
            json.dumps(row.get("receipt"), sort_keys=True)
            for row in answered
            if row["receipt_id"] == key
        }
        for key in truth
    }
    repeated = [key for key in truth if sum(row["receipt_id"] == key for row in answered) > 1]
    times = sorted(row["seconds"] for row in answered if row["run"] > 1) or sorted(
        row["seconds"] for row in answered
    )
    report = {
        "model": args.cloud or f"{args.local_model} at {args.local}",
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "receipts": len(truth),
        "repeats": args.repeat,
        "runtime_failures": f"{total - len(answered)}/{total}",
        "schema_valid": f"{sum(row['valid'] for row in rows)}/{total}",
        "schema_valid_first_attempt": f"{sum(row['valid'] and row['requests'] == 1 for row in rows)}/{total}",
        "field_accuracy": {name: f"{hits[name]}/{total}" for name in FIELDS},
        "task_completion": f"{sum(all(row['fields'].values()) for row in rows)}/{total}",
        "consistent_receipts": f"{sum(len(answers[key]) == 1 for key in repeated)}/{len(repeated)}",
        "median_s": times[len(times) // 2] if times else None,
        "rows": rows,
    }
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    name = "compare_cloud" if args.cloud else "compare_local"
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for key in (
        "model",
        "runtime_failures",
        "schema_valid",
        "schema_valid_first_attempt",
        "field_accuracy",
        "task_completion",
        "consistent_receipts",
        "median_s",
    ):
        print(f"{key}: {report[key]}")
    for row in rows:
        if row["run"] == 1 and not all(row["fields"].values()):
            misses = [n for n, hit in row["fields"].items() if not hit]
            print(f"{row['receipt_id']}: {row.get('error') or row['receipt']} (wrong: {misses})")


if __name__ == "__main__":
    sys.exit(main())
