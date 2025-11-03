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
        
        # Evaluate datasets for evaluation (test_data preferred, fallback to training data)
        eval_datasets: dict[str, Any] = {}
        
        from .lang.compiler import _execute_python_block, _eval_python_inline
        
        # Evaluate test datasets first (preferred for evaluation)
        for dataset_name, dataset in suite.test_datasets.items():
            try:
                if dataset.is_block:
                    val = _execute_python_block(dataset.code, rt, {})
                else:
                    val = _eval_python_inline(rt, dataset.code, {})
                
                # Ensure it's iterable
                if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
                    raise ValueError(f"Test data '{dataset_name}' must return an iterable (got {type(val).__name__})")
                
                # Convert dict.items() or list to list
                if isinstance(val, dict):
                    val = list(val.items())
                else:
                    val = list(val)
                
                eval_datasets[dataset_name] = val
            except Exception as e:
                raise RuntimeError(f"Error evaluating test_data '{dataset_name}' in @eval: {suite.procedure_name}: {e}") from e
        
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
                        raise ValueError(f"Training data '{dataset_name}' must return an iterable (got {type(val).__name__})")
                    
                    # Convert dict.items() or list to list
                    if isinstance(val, dict):
                        val = list(val.items())
                    else:
                        val = list(val)
                    
                    eval_datasets[dataset_name] = val
                except Exception as e:
                    raise RuntimeError(f"Error evaluating training data '{dataset_name}' in @eval: {suite.procedure_name}: {e}") from e
        
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
                    if isinstance(val, tuple) and len(val) == 2:
                        score = float(val[0])
                        feedback = None if val[1] is None else str(val[1])
                    else:
                        score = float(val)
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
                            # Bind dataset name as the iteration variable
                            metric_env = {dataset_name: example}
                            if m.is_block:
                                val = _execute_python_block(m.code, rt, metric_env)
                            else:
                                val = _eval_python_inline(rt, m.code, metric_env)
                            
                            # Process result
                            if isinstance(val, bool):
                                score_val = 1.0 if val else 0.0
                                feedback_val = None
                            elif isinstance(val, tuple) and len(val) == 2:
                                score_val = float(val[0])
                                feedback_val = None if val[1] is None else str(val[1])
                            else:
                                score_val = float(val)
                                feedback_val = None
                            
                            eval_scores.append(score_val)
                            if feedback_val:
                                eval_feedbacks.append(feedback_val)
                        except Exception as e:
                            eval_scores.append(0.0)
                            eval_feedbacks.append(f"error: {e}")
                
                eval_agg_score = sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
                eval_feedback_str = "; ".join(eval_feedbacks) if eval_feedbacks else None
                metric_results.append(
                    EvalMetricResult(
                        name=m.name, score=eval_agg_score, feedback=eval_feedback_str
                    )
                )
        
        results.append(EvalRunResult(procedure=suite.procedure_name, metrics=metric_results))
    return results
