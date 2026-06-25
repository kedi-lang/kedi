"""Run examples/launch_kit.kedi with REAL LLM calls and measure the parallel speedup.

The program issues five independent LLM calls (name, tagline, tweet, personas, risk). Sequentially
that's five round-trips; with the parallel engine they overlap into roughly one.

Real run (needs a provider API key in your environment, e.g. OPENAI_API_KEY / GROQ_API_KEY):

    uv run python examples/run_launch_kit.py --model openai:gpt-4o-mini \
        --idea "an app that turns grocery receipts into healthy recipes"

Dry run (no key, simulated per-call latency — verifies the timing/parallel mechanism):

    uv run python examples/run_launch_kit.py --demo
"""

from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace
from typing import Any

import kedi

DEFAULT_IDEA = "an app that turns grocery receipts into healthy recipes"
N_CALLS = 5  # independent templates in launch_kit.kedi


class _SimAdapter:
    """Offline adapter: sleeps `latency` per call and returns canned copy (for --demo)."""

    _CANNED = {
        "name": "ReceiptChef",
        "tagline": "Turn receipts into recipes — eat better, waste less.",
        "tweet": "Snap your grocery receipt, get healthy recipes from what you bought. 🥦",
        "personas": "busy parents, budget students, health-conscious home cooks",
        "risk": "users may not keep paper receipts to scan.",
    }

    def __init__(self, latency: float) -> None:
        self.latency = latency

    def default_model(self) -> str:
        return "sim:offline"

    def produce_sync(
        self, *, template: str, output_schema: dict[str, Any] | None = None, **_: Any
    ) -> SimpleNamespace:
        time.sleep(self.latency)  # stand in for one model round-trip
        return SimpleNamespace(**{k: self._CANNED.get(k, f"<{k}>") for k in (output_schema or {})})


@kedi.bind(file="launch_kit.kedi")
def launch_kit(idea: str) -> str:  # body comes from the .kedi file; `idea` becomes a global
    ...


def _timed(idea: str) -> tuple[str, float]:
    start = time.perf_counter()
    result = launch_kit(idea=idea)
    return result, time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", default=DEFAULT_IDEA)
    ap.add_argument("--model", default=os.getenv("KEDI_ADAPTER_MODEL", "openai:gpt-4o-mini"))
    ap.add_argument(
        "--demo", action="store_true", help="offline simulated-latency run (no API key)"
    )
    ap.add_argument(
        "--latency", type=float, default=0.6, help="simulated per-call latency for --demo"
    )
    args = ap.parse_args()

    if args.demo:
        kedi.configure(adapter=_SimAdapter(args.latency))
        mode = f"DEMO (simulated {args.latency:.2f}s/call)"
    else:
        kedi.configure(model=args.model)
        mode = f"REAL model={args.model}"

    print(f"Idea: {args.idea}")
    print(f"Mode: {mode} | {N_CALLS} independent LLM calls\n")

    seq_out, seq_t = _timed(args.idea)  # sequential (default engine)
    with kedi.parallel(max_workers=N_CALLS):  # opt into the threaded engine
        par_out, par_t = _timed(args.idea)

    print("── sequential output ──")
    print(seq_out)
    print("\n── parallel output ──")
    print(par_out)
    print("\n── timing ──")
    print(f"sequential: {seq_t:6.2f}s")
    print(f"parallel:   {par_t:6.2f}s")
    if par_t:
        print(f"speedup:    {seq_t / par_t:6.2f}x  (ideal ≈ {N_CALLS}x for pure fan-out)")


if __name__ == "__main__":
    main()
