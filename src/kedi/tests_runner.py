from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .lang import compile_program, parse_program
from .lang.ast import EvalSuite, TestSuite
from .source_loading import load_kedi_source_bundle


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


def _load_program_with_validations(
    src_main: str,
    *,
    adapter: Optional[Any],
    runtime_globals: dict[str, Any] | None = None,
) -> Any:
    # src_main may be the full source content or a filesystem path
    text: str
    if "\n" in src_main or "@" in src_main:
        text = src_main
        source_path = None
        source_map = None
    else:
        p = Path(src_main)
        loaded = load_kedi_source_bundle(p.resolve())
        text = loaded.text
        source_path = str(p.resolve())
        source_map = loaded.source_map
    prog = parse_program(text, source_path=source_path, source_map=source_map)
    rt = compile_program(prog, adapter=adapter, runtime_globals=runtime_globals)
    return rt


def run_tests(
    src_main: str,
    *,
    adapter: Optional[Any] = None,
    runtime_globals: dict[str, Any] | None = None,
) -> List[TestRunResult]:
    from pathlib import Path

    from .lang.compiler import compile_program
    from .optimization.services.program_loader import load_program_with_optimized_prompts

    # Check if src_main is a file path or source code string
    source_path = None
    if isinstance(src_main, str):
        is_likely_source = ("\n" in src_main and "@" in src_main) or len(src_main) > 500
        if not is_likely_source:
            try:
                candidate_path = Path(src_main)
                if len(str(candidate_path)) < 500 and candidate_path.exists():
                    source_path = candidate_path.resolve()
            except (OSError, ValueError):
                pass

    # Load program with optimized prompts if available (only for file paths)
    if source_path:
        try:
            program = load_program_with_optimized_prompts(source_path)
            rt = compile_program(program, adapter=adapter, runtime_globals=runtime_globals)
        except Exception:
            rt = _load_program_with_validations(
                src_main,
                adapter=adapter,
                runtime_globals=runtime_globals,
            )
    else:
        rt = _load_program_with_validations(
            src_main,
            adapter=adapter,
            runtime_globals=runtime_globals,
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


def run_evals(
    src_main: str,
    *,
    adapter: Optional[Any] = None,
    runtime_globals: dict[str, Any] | None = None,
) -> List[EvalRunResult]:
    from pathlib import Path

    from .optimization.utils import normalize_metric_result
    from .optimization.services.program_loader import load_program_with_optimized_prompts

    # Check if src_main is a file path or source code string
    # Heuristic: if it looks like source code (has newlines or @ symbols), it's not a path
    source_path = None
    if isinstance(src_main, str):
        # Quick check: if it's very long or contains newlines and @, it's likely source code
        is_likely_source = ("\n" in src_main and "@" in src_main) or len(src_main) > 500
        if not is_likely_source:
            # Try to resolve as path - but catch errors if path is too long (source code)
            try:
                candidate_path = Path(src_main)
                # Only check exists if path is reasonable length (file paths are short)
                if len(str(candidate_path)) < 500 and candidate_path.exists():
                    source_path = candidate_path.resolve()
            except (OSError, ValueError):
                # Path resolution failed - likely source code, not a path
                pass

    # Load program with optimized prompts if available (only for file paths)
    if source_path:
        try:
            # Load with optimized prompts
            program = load_program_with_optimized_prompts(source_path)
            # Compile with optimized prompts
            rt = compile_program(program, adapter=adapter, runtime_globals=runtime_globals)
        except Exception:
            rt = _load_program_with_validations(
                src_main,
                adapter=adapter,
                runtime_globals=runtime_globals,
            )
    else:
        rt = _load_program_with_validations(
            src_main,
            adapter=adapter,
            runtime_globals=runtime_globals,
        )
    suites: List[EvalSuite] = getattr(rt, "_validation_evals", [])
    results: List[EvalRunResult] = []

    for suite in suites:
        metric_results: List[EvalMetricResult] = []

        # Evaluate datasets for evaluation (test_data preferred, fallback to training data)
        eval_datasets: dict[str, Any] = {}

        from .lang.compiler import _eval_python_inline, _execute_python_block

        # Evaluate test datasets first (preferred for evaluation)
        for dataset_name, dataset in suite.test_datasets.items():
            try:
                if dataset.is_block:
                    val = _execute_python_block(dataset.code, rt, {})
                else:
                    val = _eval_python_inline(rt, dataset.code, {})

                # Ensure it's iterable
                if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
                    raise ValueError(
                        f"Test data '{dataset_name}' must return an iterable (got {type(val).__name__})"
                    )

                # Convert dict.items() or list to list
                if isinstance(val, dict):
                    val = list(val.items())
                else:
                    val = list(val)

                eval_datasets[dataset_name] = val
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating test_data '{dataset_name}' in @eval: {suite.procedure_name}: {e}"
                ) from e

        # Fall back to training datasets for datasets not in test_data
        for dataset_name, dataset in suite.datasets.items():
            if dataset_name not in eval_datasets:
                try:
                    if dataset.is_block:
                        val = _execute_python_block(dataset.code, rt, {})
                    else:
                        val = _eval_python_inline(rt, dataset.code, {})

                    # Ensure it's iterable
                    if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
                        raise ValueError(
                            f"Training data '{dataset_name}' must return an iterable (got {type(val).__name__})"
                        )

                    # Convert dict.items() or list to list
                    if isinstance(val, dict):
                        val = list(val.items())
                    else:
                        val = list(val)

                    eval_datasets[dataset_name] = val
                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating training data '{dataset_name}' in @eval: {suite.procedure_name}: {e}"
                    ) from e

        # Evaluate metrics
        for m in suite.metrics:
            if m.dataset_name is None:
                # Legacy single-eval metric (should be deprecated but kept for compatibility)
                score: float = 0.0
                feedback: Optional[str] = None
                try:
                    if m.is_block:
                        val = _execute_python_block(m.code, rt, {})
                    else:
                        val = _eval_python_inline(rt, m.code, {})
                    score, feedback = normalize_metric_result(val)
                except Exception as e:
                    score = 0.0
                    feedback = f"error: {e}"
                metric_results.append(EvalMetricResult(name=m.name, score=score, feedback=feedback))
            else:
                # Dataset-aware metric - iterate over dataset (test_data preferred, fallback to training)
                dataset_name = m.dataset_name

                # Check dataset exists (test_data or training data)
                if dataset_name not in eval_datasets:
                    raise ValueError(
                        f"Metric '{m.name}' references dataset '{dataset_name}' but no data (test_data or training data) "
                        f"with this name found in @eval: {suite.procedure_name}."
                    )

                # Use dataset for evaluation
                eval_data = eval_datasets[dataset_name]

                eval_scores: list[float] = []
                eval_feedbacks: list[str] = []

                if len(eval_data) == 0:
                    eval_scores.append(0.0)
                    eval_feedbacks.append("no examples")
                else:
                    for example in eval_data:
                        try:
                            # Extract input from example
                            # Handle two formats:
                            # 1. Explicit tuples: ((input, expected_output)) - extract input
                            # 2. dict.items() or other: (key, value) - use as-is
                            if isinstance(example, tuple) and len(example) == 2:
                                first, second = example
                                # (input, expected_output): expected is often a dict, e.g. {"answer": 13}
                                if isinstance(second, dict):
                                    input_val = first
                                    expected_val = second
                                elif second is None:
                                    input_val = first
                                    expected_val = None
                                elif isinstance(first, tuple) and not isinstance(
                                    second, (list, dict, tuple)
                                ):
                                    input_val = first
                                    expected_val = second
                                else:
                                    # dict.items(): (key, value) — whole pair is the metric input
                                    input_val = example
                                    expected_val = None
                            else:
                                # Single value or non-standard format - use as-is
                                input_val = example
                                expected_val = None

                            # Bind dataset name with input value
                            metric_env = {dataset_name: input_val}

                            # Also bind expected output if available (for metrics that need it)
                            if expected_val is not None:
                                metric_env["expected"] = expected_val

                            if m.is_block:
                                val = _execute_python_block(m.code, rt, metric_env)
                            else:
                                val = _eval_python_inline(rt, m.code, metric_env)

                            score_val, feedback_val = normalize_metric_result(val)

                            eval_scores.append(score_val)
                            if feedback_val:
                                eval_feedbacks.append(feedback_val)
                        except Exception as e:
                            eval_scores.append(0.0)
                            eval_feedbacks.append(f"error: {e}")

                eval_agg_score = sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
                eval_feedback_str = "; ".join(eval_feedbacks) if eval_feedbacks else None
                metric_results.append(
                    EvalMetricResult(name=m.name, score=eval_agg_score, feedback=eval_feedback_str)
                )

        results.append(EvalRunResult(procedure=suite.procedure_name, metrics=metric_results))
    return results
