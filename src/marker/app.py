from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Tuple

from .lang import compile_program, parse_program
from .agent_adapter.adapters import PydanticAdapter, DSPyAdapter
from .tests_runner import run_tests, run_evals
from .codegen.runner import run_codegen


def run_tests_cli(source: str, *, model: str | None = None) -> Tuple[int, str]:
    ok, err = maybe_run_codegen(source, model=model)
    if not ok:
        return 1, (err or "Code generation failed")
    results = run_tests(source)
    failed = sum(1 for s in results for c in s.cases if not c.success)
    lines = ["Tests:"]
    for suite in results:
        for case in suite.cases:
            status = "OK" if case.success else f"FAIL: {case.error}"
            lines.append(f"- {suite.procedure}::{case.name}: {status}")
    return (1 if failed else 0, "\n".join(lines))


def run_evals_cli(source: str, *, model: str | None = None) -> Tuple[int, str]:
    ok, err = maybe_run_codegen(source, model=model)
    if not ok:
        return 1, (err or "Code generation failed")
    results = run_evals(source)
    lines = ["Evals:"]
    for suite in results:
        for metric in suite.metrics:
            fb = "" if metric.feedback is None else f" | {metric.feedback}"
            lines.append(f"- {suite.procedure}::{metric.name}: {metric.score}{fb}")
    return (0, "\n".join(lines))


def maybe_run_codegen(
    source_path: str, 
    *, 
    agent: str | None = None,
    model: str | None = None,
    retries: int | None = None,
    use_cache: bool = True
) -> Tuple[bool, str | None]:
    try:
        src_txt = Path(source_path).read_text(encoding="utf-8")
        prog = parse_program(src_txt)
        needs = any(getattr(p, "ai_prompt", None) for p in prog.procedures)
        if not needs:
            return True, None
        
        # Use provided values or fall back to defaults (already handled in CLI)
        agent_kind = agent or "pydantic_ai"
        max_retries = retries or 5
        
        ok = run_codegen(Path(source_path), agent_kind=agent_kind, model=model, max_retries=max_retries, use_cache=use_cache)
        if not ok:
            return False, "Code generation failed to produce passing tests."
        return True, None
    except Exception as e:
        return False, str(e)


def run_program_cli(source: str, *, adapter: str, model: str | None) -> Tuple[int, str]:
    src = Path(source).read_text(encoding="utf-8")
    # Merge cache for execution transparency
    cache_path = Path(source).with_suffix(".cache.marker")
    if cache_path.exists():
        src = src + "\n\n" + cache_path.read_text(encoding="utf-8")
    program = parse_program(src)
    adapter_impl = DSPyAdapter(model=model or "groq/qwen/qwen3-32b") if adapter == "dspy" else PydanticAdapter(model=model or "groq:qwen/qwen3-32b")
    rt = compile_program(program, adapter=adapter_impl)
    out = rt.run_main()
    return 0, ("" if out is None else str(out))


