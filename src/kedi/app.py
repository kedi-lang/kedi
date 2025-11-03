from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

from .agent_adapter.adapters import DSPyAdapter, PydanticAdapter
from .codegen.runner import run_codegen
from .lang import compile_program, parse_program
from .tests_runner import run_evals, run_tests


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


def run_optimization_cli(
    source: str,
    *,
    optimizer: str = "mock",
    optimizer_model: str | None = None,
    reflection_model: str | None = None,
) -> Tuple[int, str]:
    """Run optimization on procedures with optimize spans.
    
    Returns:
        (exit_code, message) tuple
    """
    from pathlib import Path
    import pprint
    from typing import Any
    from .optimization import get_optimizer
    from .lang.compiler import _execute_python_block, _eval_python_inline
        
    source_path = Path(source).resolve()
    
    # Configure DSPy for GEPA if needed
    if optimizer == "gepa":
        try:
            import dspy
            from dspy import LM
            
            # Use provided optimizer_model or default
            main_model = optimizer_model or "openai/gpt-4o"
            # OpenAI reasoning models require temperature=1.0 and max_tokens >= 16000
            # gpt-4o supports max 16384, gpt-5 supports 32000
            if "gpt-5" in main_model.lower():
                main_lm = LM(model=main_model, temperature=1.0, max_tokens=32000)
            elif "gpt-4o" in main_model.lower():
                main_lm = LM(model=main_model, temperature=1.0, max_tokens=16384)
            else:
                main_lm = LM(model=main_model)
            dspy.settings.configure(lm=main_lm, cache=False)
        except ImportError:
            return 1, "DSPy is required for GEPA optimizer. Install with: pip install dspy-ai"
    
    # Get optimizer instance (with reflection_model for GEPA)
    if optimizer == "gepa" and reflection_model:
        from .optimization.gepa import GEPAOptimizer
        opt = GEPAOptimizer(reflection_model=reflection_model)
    else:
        opt = get_optimizer(optimizer)
    
    if not opt:
        return 1, f"Unknown optimizer: {optimizer}"
    
    # Delete cached optimized file and scores - always run fresh optimization
    optimized_path = source_path.with_suffix(".optimized.kedi")
    if optimized_path.exists():
        optimized_path.unlink()
    scores_path = source_path.with_suffix(".optimized_scores.kedi")
    if scores_path.exists():
        scores_path.unlink()
    
    # Parse program (including .val.kedi and .cache.kedi)
    text = source_path.read_text(encoding="utf-8")
    val_path = source_path.with_suffix(".val.kedi")
    if val_path.exists():
        text = text + "\n\n" + val_path.read_text(encoding="utf-8")
    cache_path = source_path.with_suffix(".cache.kedi")
    if cache_path.exists():
        text = text + "\n\n" + cache_path.read_text(encoding="utf-8")
    
    prog = parse_program(text)
    
    # Find procedures with optimize spans
    procs_with_optimize = [p for p in prog.procedures if p.optimize_spans]
    if not procs_with_optimize:
        return 0, "No procedures with optimize spans found"
    
    # Find eval suites
    evals = getattr(prog, "_validation_evals", [])
    eval_map = {e.procedure_name: e for e in evals}
    
    # Validate
    for proc in procs_with_optimize:
        if proc.name not in eval_map:
            return 1, f"Procedure '{proc.name}' needs @eval: {proc.name} suite"
        if not eval_map[proc.name].datasets:
            return 1, f"Procedure '{proc.name}' needs training data (> data:) in eval suite"
    
    # Compile runtime for data loading
    rt = compile_program(prog, adapter=None)
    
    # Optimize each procedure
    optimized_dict = {}
    for proc in procs_with_optimize:
        eval_suite = eval_map[proc.name]
        
        # Load training data (first dataset)
        train_data = []
        if eval_suite.datasets:
            dataset = list(eval_suite.datasets.values())[0]
            try:
                if dataset.is_block:
                    val = _execute_python_block(dataset.code, rt, {})
                else:
                    val = _eval_python_inline(rt, dataset.code, {})
                
                # Ensure it's iterable
                if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
                    return 1, f"Data '{dataset.name}' must return an iterable (got {type(val).__name__})"
                
                # Convert dict.items() or list to list
                if isinstance(val, dict):
                    data_list = list(val.items())
                else:
                    data_list = list(val)
                
                # For optimization, format as (input, expected_output) tuples
                # If data is already tuples, use as-is
                # If data is single objects, wrap them: (obj, None) for analytical metrics
                for item in data_list:
                    if isinstance(item, tuple):
                        # Already a tuple - use as-is
                        if len(item) == 2:
                            train_data.append(item)
                        else:
                            # Single value in tuple - make it (value, None) for analytical eval
                            train_data.append((item[0], None))
                    else:
                        # Single object - wrap as (obj, None) for analytical eval
                        train_data.append((item, None))
            except Exception as e:
                return 1, f"Error loading training data for '{proc.name}': {e}"
        
        if not train_data:
            return 1, f"No training data for '{proc.name}'"
        
        # Load test data (if available, from same dataset name)
        test_data_list = None
        if eval_suite.test_datasets and dataset.name in eval_suite.test_datasets:
            test_dataset = eval_suite.test_datasets[dataset.name]
            try:
                if test_dataset.is_block:
                    val = _execute_python_block(test_dataset.code, rt, {})
                else:
                    val = _eval_python_inline(rt, test_dataset.code, {})
                
                if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
                    return 1, f"Test data '{test_dataset.name}' must return an iterable (got {type(val).__name__})"
                
                # Format test data same as train data
                if isinstance(val, dict):
                    test_data_list = list(val.items())
                else:
                    test_data_list = list(val)
                    
                # Format as (input, expected_output) tuples
                formatted_test_data = []
                for item in test_data_list:
                    if isinstance(item, tuple) and len(item) == 2:
                        formatted_test_data.append(item)
                    elif isinstance(item, tuple):
                        formatted_test_data.append((item[0], None))
                    else:
                        formatted_test_data.append((item, None))
                test_data_list = formatted_test_data
            except Exception as e:
                return 1, f"Error loading test data for '{proc.name}': {e}"
        
        # For analytical metrics (no expected outputs), optimization requires expected outputs
        if all(item[1] is None for item in train_data):
            return 1, (
                f"Procedure '{proc.name}' has analytical metric (no expected outputs in training data). "
                f"Optimization requires expected outputs. "
                f"Either provide expected outputs in training data, or remove --optimize flag."
            )
        
        # Resolve param_types from type annotations
        param_types: dict[str, type] = {}
        if proc.param_type_annotations:
            for param_name in proc.params:
                type_anno = proc.param_type_annotations.get(param_name)
                if type_anno:
                    # Try to resolve type using runtime's type environment
                    try:
                        from .lang.compiler.environment import build_type_env
                        type_env = build_type_env(rt, {})
                        resolved_type = eval(type_anno, type_env, type_env)
                        if isinstance(resolved_type, type):
                            param_types[param_name] = resolved_type
                    except Exception:
                        # Type resolution failed - skip validation for this param
                        pass
        
        # Get metric from eval suite (first metric, one per procedure)
        if not eval_suite.metrics:
            return 1, f"Procedure '{proc.name}' needs at least one metric in eval suite"
        
        metric_def = eval_suite.metrics[0]
        
        # Validate metric's dataset exists
        if metric_def.dataset_name:
            if metric_def.dataset_name not in eval_suite.datasets:
                return 1, (
                    f"Metric '{metric_def.name}' references dataset '{metric_def.dataset_name}' "
                    f"which is not defined in @eval: {proc.name}. "
                    f"Available datasets: {', '.join(eval_suite.datasets.keys())}"
                )
        
        # Get procedure callable from runtime
        try:
            proc_func = rt._get_proc(proc.name)
        except KeyError:
            return 1, f"Procedure '{proc.name}' not found in compiled runtime"
        
        # Create metric closure that captures procedure and metric code
        def create_metric_closure(metric_code: str, is_block: bool, dataset_name: str):
            """Create metric function that calls procedure internally."""
            def metric_fn(input_val: Any, expected_output: Any) -> tuple[float, str | None]:
                """Metric function called by optimizer.
                
                Args:
                    input_val: Input value(s) - single value or tuple for multi-param
                    expected_output: Expected output for comparison
                
                Returns:
                    (score, feedback) tuple
                """
                try:
                    # Prepare metric environment with dataset variable
                    metric_env = {dataset_name: input_val}
                    
                    # If single param, also bind directly to param name for convenience
                    if len(proc.params) == 1:
                        metric_env[proc.params[0]] = input_val
                    else:
                        # Multi-param: unpack tuple into individual params
                        if isinstance(input_val, tuple):
                            for param_name, arg_val in zip(proc.params, input_val):
                                metric_env[param_name] = arg_val
                    
                    # Execute metric code (which internally calls procedure)
                    if is_block:
                        val = _execute_python_block(metric_code, rt, metric_env)
                    else:
                        val = _eval_python_inline(rt, metric_code, metric_env)
                    
                    # Process result - must return (float, str | None)
                    if isinstance(val, bool):
                        return (1.0 if val else 0.0, None)
                    elif isinstance(val, tuple) and len(val) == 2:
                        score = float(val[0])
                        feedback = None if val[1] is None else str(val[1])
                        return (score, feedback)
                    else:
                        score = float(val)
                        return (score, None)
                except Exception as e:
                    # Metric execution failed - return 0.0 score
                    return (0.0, f"error: {e}")
            
            return metric_fn
        
        metric_fn = create_metric_closure(
            metric_def.code,
            metric_def.is_block,
            dataset.name,
        )
        
        # Call optimizer with new interface
        try:
            optimized, train_score = opt.optimize_procedure(
                proc_name=proc.name,
                proc_params=proc.params,
                param_types=param_types,
                spans=proc.optimize_spans,
                train_data=train_data,
                test_data=test_data_list,
                metric=metric_fn,
            )
            optimized_dict[proc.name] = optimized
            # Store training performance in a separate file for eval to use
            train_scores_dict = {}
            if optimized_path.with_suffix(".optimized_scores.kedi").exists():
                import ast
                scores_content = optimized_path.with_suffix(".optimized_scores.kedi").read_text(encoding="utf-8")
                try:
                    train_scores_dict = ast.literal_eval(scores_content)
                except Exception:
                    train_scores_dict = {}
            train_scores_dict[proc.name] = train_score
            scores_str = pprint.pformat(train_scores_dict, width=120, indent=2, sort_dicts=False)
            optimized_path.with_suffix(".optimized_scores.kedi").write_text(scores_str, encoding="utf-8")
        except Exception as e:
            return 1, f"Optimization error for '{proc.name}': {e}"
    
    # Write optimized prompts to file and display them
    if optimized_dict:
        dict_str = pprint.pformat(optimized_dict, width=120, indent=2, sort_dicts=False)
        optimized_path.write_text(dict_str, encoding="utf-8")
        
        # Display optimized prompts like DSPy tutorial
        lines = [f"Optimization complete. Wrote optimized prompts to {optimized_path.name}", ""]
        for proc_name, prompts in optimized_dict.items():
            lines.append(f"Optimized prompts for '{proc_name}':")
            for span_name, prompt in prompts.items():
                lines.append(f"  [{span_name}]:")
                # Display prompt with indentation
                for prompt_line in prompt.split('\n'):
                    lines.append(f"    {prompt_line}")
                lines.append("")
        
        return 0, "\n".join(lines)
    
    return 0, ""


def run_evals_cli(
    source: str, 
    *, 
    model: str | None = None, 
    optimizer: str = "mock",
    optimizer_model: str | None = None,
    reflection_model: str | None = None,
    optimize: bool = False, 
    use_cache: bool = True
) -> Tuple[int, str]:
    # Run optimizer FIRST if requested, before codegen
    if optimize:
        code, msg = run_optimization_cli(
            source,
            optimizer=optimizer,
            optimizer_model=optimizer_model,
            reflection_model=reflection_model,
        )
        if code != 0:
            return code, msg
    
    # Run codegen AFTER optimizer (if optimizer ran, DSPy is already configured with optimizer model)
    ok, err = maybe_run_codegen(source, model=model)
    if not ok:
        return 1, (err or "Code generation failed")
    
    # Always use original source - optimized prompts will be loaded and applied during inference
    results = run_evals(source, adapter=None)  # Will use default adapter
    lines = ["Evals:"]
    for suite in results:
        # Display metrics (evaluated only on test data)
        for metric in suite.metrics:
            fb = "" if metric.feedback is None else f" | {metric.feedback}"
            lines.append(f"- {suite.procedure}::{metric.name}: {metric.score:.4f}{fb}")
    return (0, "\n".join(lines))


def maybe_run_codegen(
    source_path: str,
    *,
    agent: str | None = None,
    model: str | None = None,
    retries: int | None = None,
    use_cache: bool = True,
) -> Tuple[bool, str | None]:
    try:
        src_txt = Path(source_path).read_text(encoding="utf-8")
        prog = parse_program(src_txt)
        # Check for auto_spec (new) or ai_prompt (legacy)
        needs = any(
            getattr(p, "auto_spec", None) or getattr(p, "ai_prompt", None) for p in prog.procedures
        )
        if not needs:
            return True, None

        # Use provided values or fall back to defaults (already handled in CLI)
        agent_kind = agent or "pydantic_ai"
        max_retries = retries or 5

        ok = run_codegen(
            Path(source_path),
            agent_kind=agent_kind,
            model=model,
            max_retries=max_retries,
            use_cache=use_cache,
        )
        if not ok:
            return False, "Code generation failed to produce passing tests."
        return True, None
    except Exception as e:
        return False, str(e)


def run_program_cli(source: str, *, adapter: str, model: str | None) -> Tuple[int, str]:
    src = Path(source).read_text(encoding="utf-8")
    # Merge cache for execution transparency
    cache_path = Path(source).with_suffix(".cache.kedi")
    if cache_path.exists():
        src = src + "\n\n" + cache_path.read_text(encoding="utf-8")
    program = parse_program(src)
    adapter_impl = (
        DSPyAdapter(model=model or "groq/qwen/qwen3-32b")
        if adapter == "dspy"
        else PydanticAdapter(model=model or "groq:qwen/qwen3-32b")
    )
    rt = compile_program(program, adapter=adapter_impl)
    out = rt.run_main()
    return 0, ("" if out is None else str(out))
