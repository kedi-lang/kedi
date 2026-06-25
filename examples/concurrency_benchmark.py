"""Wall-clock benchmark of non-blocking Kedi templates: parallel ON vs OFF.

Each LLM call is simulated with a fixed-latency adapter (no network), so wall-clock differences
isolate the concurrency benefit of the promise-pipelining engine across several program shapes:

  fan-out          N independent templates joined at the end        -> ~Nx
  linear chain     A->B->C->D (each needs the previous)             -> ~1x (correctly serial)
  parallel chains  3 independent A->B chains                        -> ~(#chains)x
  diamond          A -> {B, C} -> D                                 -> partial

Run:  uv run python examples/concurrency_benchmark.py
"""

from __future__ import annotations

import re
import time
from types import SimpleNamespace
from typing import Any

from kedi.engine import SequentialEngine, ThreadEngine
from kedi.lang import compile_program, parse_program

LATENCY = 0.25  # simulated seconds per LLM call
WORKERS = 8

_KEY_RE = re.compile(r"\[([A-Za-z_]\w*)(?::[^\]]*)?\]")


class SlowAdapter:
    """Minimal adapter that sleeps ``latency`` per call and returns content-derived outputs."""

    def __init__(self, latency: float = LATENCY) -> None:
        self.latency = latency

    def default_model(self) -> str:
        return "slow:model"

    def produce_sync(
        self, *, template: str, output_schema: dict[str, Any] | None = None, **_: Any
    ) -> SimpleNamespace:
        time.sleep(self.latency)  # stand in for one model round-trip
        keys = list(output_schema or {}) or list(dict.fromkeys(_KEY_RE.findall(template)))
        return SimpleNamespace(**{k: f"{k}-out" for k in keys})


# Each program is a procedure whose body issues several templates, plus a top-level call.
PROGRAMS: dict[str, tuple[str, int]] = {
    # name: (source, number of LLM calls)
    "fan-out (6 independent)": (
        "@gen():\n"
        "  >> task one [a]\n"
        "  >> task two [b]\n"
        "  >> task three [c]\n"
        "  >> task four [d]\n"
        "  >> task five [e]\n"
        "  >> task six [f]\n"
        "  = <a> <b> <c> <d> <e> <f>\n\n"
        "= <gen()>\n",
        6,
    ),
    "linear chain (depth 4)": (
        "@gen():\n"
        "  >> seed [a]\n"
        "  >> from <a> make [b]\n"
        "  >> from <b> make [c]\n"
        "  >> from <c> make [d]\n"
        "  = <d>\n\n"
        "= <gen()>\n",
        4,
    ),
    "parallel chains (3 x depth 2)": (
        "@gen():\n"
        "  >> seed one [a]\n"
        "  >> from <a> make [b]\n"
        "  >> seed two [c]\n"
        "  >> from <c> make [d]\n"
        "  >> seed three [e]\n"
        "  >> from <e> make [f]\n"
        "  = <b> <d> <f>\n\n"
        "= <gen()>\n",
        6,
    ),
    "diamond (A -> B,C -> D)": (
        "@gen():\n"
        "  >> root [a]\n"
        "  >> branch from <a> [b]\n"
        "  >> branch from <a> [c]\n"
        "  >> join <b> and <c> [d]\n"
        "  = <d>\n\n"
        "= <gen()>\n",
        4,
    ),
}


def _run(source: str, engine: Any) -> tuple[Any, float]:
    program = parse_program(source)
    rt = compile_program(program, adapter=SlowAdapter(), engine=engine)
    start = time.perf_counter()
    result = rt.run_main()
    return result, time.perf_counter() - start


def main() -> None:
    pool = ThreadEngine(max_workers=WORKERS)
    print(f"Simulated LLM latency: {LATENCY:.2f}s/call | pool workers: {WORKERS}\n")
    header = f"{'pattern':<32}{'calls':>6}{'serial':>9}{'parallel':>10}{'speedup':>9}  result"
    print(header)
    print("-" * len(header))
    for name, (source, calls) in PROGRAMS.items():
        seq_result, seq_t = _run(source, SequentialEngine())
        par_result, par_t = _run(source, pool)
        match = "ok" if seq_result == par_result else f"MISMATCH({seq_result!r}!={par_result!r})"
        speedup = seq_t / par_t if par_t else float("inf")
        print(f"{name:<32}{calls:>6}{seq_t:>8.2f}s{par_t:>9.2f}s{speedup:>8.2f}x  {match}")
    pool.shutdown()


if __name__ == "__main__":
    main()
