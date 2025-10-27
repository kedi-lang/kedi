from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .agent_adapter.adapters import PydanticAdapter
from .lang import compile_program, parse_program
from .lang.ast import EvalSuite, TestSuite


@dataclass
class TestCaseResult:
    name: str
    success: bool
    error: Optional[str] = None


@dataclass
class TestRunResult:
    procedure: str
    cases: List[TestCaseResult]


@dataclass
class EvalMetricResult:
    name: str
    score: float
    feedback: Optional[str]


@dataclass
class EvalRunResult:
    procedure: str
    metrics: List[EvalMetricResult]


def _load_program_with_validations(src_main: str, *, adapter: Optional[Any]) -> Any:
    # src_main may be the full source content or a filesystem path
    text: str
    if "\n" in src_main or "@" in src_main:
        text = src_main
    else:
        p = Path(src_main)
        text = p.read_text(encoding="utf-8")
        # Merge companion .val.kedi and .cache.kedi if present
        val_path = p.with_suffix(".val.kedi")
        if val_path.exists():
            text = text + "\n\n" + val_path.read_text(encoding="utf-8")
        cache_path = p.with_suffix(".cache.kedi")
        if cache_path.exists():
            text = text + "\n\n" + cache_path.read_text(encoding="utf-8")
    prog = parse_program(text)
    rt = compile_program(prog, adapter=adapter)
    return rt


def run_tests(src_main: str, *, adapter: Optional[Any] = None) -> List[TestRunResult]:
    rt = _load_program_with_validations(
        src_main, adapter=adapter or PydanticAdapter(model="groq:qwen/qwen3-32b")
    )
    suites: List[TestSuite] = getattr(rt, "_validation_tests", [])
    results: List[TestRunResult] = []
    for suite in suites:
        case_results: List[TestCaseResult] = []
        for c in suite.cases:
            try:
                if c.is_block:
                    from .lang.compiler import _execute_python_block

                    _execute_python_block(c.code, rt, {})
                else:
                    # Backtick expression or statement: evaluate via inline python helper
                    from .lang.compiler import _eval_python_inline

                    _eval_python_inline(rt, c.code, {})
                case_results.append(TestCaseResult(name=c.name, success=True))
            except Exception as e:
                case_results.append(TestCaseResult(name=c.name, success=False, error=str(e)))
        results.append(TestRunResult(procedure=suite.procedure_name, cases=case_results))
    return results


def run_evals(src_main: str, *, adapter: Optional[Any] = None) -> List[EvalRunResult]:
    rt = _load_program_with_validations(
        src_main, adapter=adapter or PydanticAdapter(model="groq:qwen/qwen3-32b")
    )
    suites: List[EvalSuite] = getattr(rt, "_validation_evals", [])
    results: List[EvalRunResult] = []
    for suite in suites:
        metric_results: List[EvalMetricResult] = []
        for m in suite.metrics:
            score: float = 0.0
            feedback: Optional[str] = None
            try:
                if m.is_block:
                    from .lang.compiler import _execute_python_block

                    val = _execute_python_block(m.code, rt, {})
                else:
                    from .lang.compiler import _eval_python_inline

                    val = _eval_python_inline(rt, m.code, {})
                if isinstance(val, tuple) and len(val) == 2:
                    score = float(val[0])
                    feedback = None if val[1] is None else str(val[1])
                else:
                    score = float(val)
            except Exception as e:
                score = 0.0
                feedback = f"error: {e}"
            metric_results.append(EvalMetricResult(name=m.name, score=score, feedback=feedback))
        results.append(EvalRunResult(procedure=suite.procedure_name, metrics=metric_results))
    return results
