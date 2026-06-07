from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Tuple

from .agent_adapter.adapters import DSPyAdapter, PydanticAdapter
from .codegen.runner import run_codegen
from .lang import compile_program, parse_program
from .lang.parser.diagnostics import KediParseError
from .source_loading import load_kedi_source_bundle
from .tests_runner import run_evals, run_tests
from .utils import ArgType, AttrDict


class _LazyRuntimeAdapter:
    """Instantiate the real adapter only when runtime actually uses it."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._inner: Any | None = None

    def _get_inner(self) -> Any:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    async def produce(self, *args, **kwargs):
        return await self._get_inner().produce(*args, **kwargs)

    def produce_sync(self, *args, **kwargs):
        return self._get_inner().produce_sync(*args, **kwargs)


def _build_runtime_adapter(adapter: str, model: str | None):
    """Create the adapter instance used for eval/test/program execution."""
    if adapter == "dspy":
        return _LazyRuntimeAdapter(lambda: DSPyAdapter(model=model or "groq/qwen/qwen3-32b"))
    return _LazyRuntimeAdapter(lambda: PydanticAdapter(model=model or "groq:qwen/qwen3-32b"))


CliArgs = AttrDict[str, ArgType]


def _runtime_globals(runtime_args: CliArgs | None) -> dict[str, Any]:
    return {"args": runtime_args or AttrDict({}, default=None)}


def _set_runtime_args(rt: Any, runtime_args: CliArgs | None) -> None:
    if runtime_args is not None:
        rt.set_initial_globals(_runtime_globals(runtime_args))


def run_tests_cli(
    source: str,
    *,
    adapter: str = "pydantic",
    model: str | None = None,
    runtime_args: CliArgs | None = None,
) -> Tuple[int, str]:
    ok, err = maybe_run_codegen(source, model=model)
    if not ok:
        return 1, (err or "Code generation failed")
    kwargs: dict[str, Any] = {"adapter": _build_runtime_adapter(adapter, model)}
    if runtime_args is not None:
        kwargs["runtime_globals"] = _runtime_globals(runtime_args)
    results = run_tests(source, **kwargs)
    failed = sum(1 for s in results for c in s.cases if not c.success)
    lines = ["Tests:"]
    for suite in results:
        for case in suite.cases:
            status = "OK" if case.success else f"FAIL: {case.error}"
            lines.append(f"- {suite.procedure}::{case.name}: {status}")
    return (1 if failed else 0, "\n".join(lines))


def _gepa_checkpoint_root(source_path: Path) -> Path:
    """Directory where DSPy GEPA stores ``gepa_state.bin`` (resume)."""
    return source_path.parent / f"{source_path.name}.gepa"


def run_optimization_cli(
    source: str,
    *,
    optimizer: str = "mock",
    optimizer_model: str | None = None,
    reflection_model: str | None = None,
    max_metric_calls: int = 30,
    optimizer_model_max_tokens: int = 2000,
    reflection_model_max_tokens: int = 32000,
    reflection_minibatch_size: int = 3,
    max_validation_examples: int | None = None,
    adapter: str = "pydantic",
    adapter_model: str | None = None,
    optimizer_fresh: bool = False,
    runtime_args: CliArgs | None = None,
) -> Tuple[int, str]:
    """Run optimization on procedures with optimize spans.

    Returns:
        (exit_code, message) tuple
    """
    from pathlib import Path

    from .optimization import get_optimizer
    from .optimization.config import OptimizerConfig
    from .optimization.services import (
        DSPyConfigurationService,
        MetricFactory,
        OptimizationDataLoader,
        OptimizationResultWriter,
        TypeResolver,
    )

    source_path = Path(source).resolve()

    # Parse and validate before creating any optimizer/runtime adapters.
    # Missing eval/data should fail deterministically without requiring API keys.
    loaded = load_kedi_source_bundle(source_path)
    prog = parse_program(
        loaded.text,
        source_path=str(source_path),
        source_map=loaded.source_map,
    )

    # Find procedures with optimize spans
    procs_with_optimize = [p for p in prog.procedures if p.optimize_spans]
    if not procs_with_optimize:
        return 0, "No procedures with optimize spans found"

    # Find eval suites. `parse_program` returns `ProgramWithValidation`
    # when `@eval:` blocks are present, exposing them as `.evals`.
    from .lang.ast import ProgramWithValidation

    evals = prog.evals if isinstance(prog, ProgramWithValidation) else []
    eval_map = {e.procedure_name: e for e in evals}

    # Validate
    for proc in procs_with_optimize:
        if proc.name not in eval_map:
            return 1, f"Procedure '{proc.name}' needs @eval: {proc.name} suite"
        if not eval_map[proc.name].datasets:
            return (
                1,
                f"Procedure '{proc.name}' needs training data (> data:) in eval suite",
            )

    # Create optimizer configuration only after cheap validation passes.
    config = OptimizerConfig.from_cli_options(
        max_metric_calls=max_metric_calls,
        optimizer_model_max_tokens=optimizer_model_max_tokens,
        reflection_model_max_tokens=reflection_model_max_tokens,
        optimizer_model=optimizer_model,
        reflection_model=reflection_model,
        reflection_minibatch_size=reflection_minibatch_size,
        max_validation_examples=max_validation_examples,
    )

    adapter_for_runtime = None
    if optimizer == "gepa":
        try:
            dspy_config_service = DSPyConfigurationService(config)
            main_model = config.optimizer_model or "openai/gpt-4o"
            config.optimizer_model = main_model
            dspy_config_service.configure_main_lm(main_model)
            from .agent_adapter.adapters import DSPyAdapter

            adapter_for_runtime = DSPyAdapter(model=main_model)
        except ImportError:
            return (
                1,
                "DSPy is required for GEPA optimizer. Install with: uv add dspy-ai",
            )
    else:
        model = adapter_model or "groq:qwen/qwen3-32b"
        if adapter == "dspy":
            try:
                from .agent_adapter.adapters import DSPyAdapter

                adapter_for_runtime = DSPyAdapter(model=model)
            except ImportError:
                return (
                    1,
                    "DSPy is required for dspy adapter. Install with: uv add dspy-ai",
                )
        else:
            adapter_for_runtime = PydanticAdapter(model=model)

    result_writer = OptimizationResultWriter(source_path)

    if optimizer_fresh:
        result_writer.delete_existing_optimized_files()
        gepa_root = _gepa_checkpoint_root(source_path)
        if gepa_root.exists():
            shutil.rmtree(gepa_root)
    else:
        from .optimization.services.prompt_loader import OptimizedPromptLoader

        prompt_loader = OptimizedPromptLoader(source_path)
        if prompt_loader.has_optimized_prompts():
            prompt_loader.inject_optimized_prompts(prog)

    rt = compile_program(prog, adapter=adapter_for_runtime)
    _set_runtime_args(rt, runtime_args)

    # Initialize services
    data_loader = OptimizationDataLoader(rt)
    type_resolver = TypeResolver(rt)
    metric_factory = MetricFactory(rt)

    # Get optimizer instance
    if optimizer == "gepa":
        from .optimization.gepa import GEPAOptimizer

        opt = GEPAOptimizer(config=config, runtime=rt, source_path=source_path)
    else:
        opt = get_optimizer(optimizer)

    if not opt:
        return 1, f"Unknown optimizer: {optimizer}"

    # Optimize each procedure
    optimized_dict = {}
    train_scores_dict = {}

    for proc in procs_with_optimize:
        eval_suite = eval_map[proc.name]

        # Load training data using service
        dataset_names = list(eval_suite.datasets.keys())
        dataset_name = dataset_names[0] if dataset_names else None
        if not dataset_name:
            return 1, f"Procedure '{proc.name}' needs training data"

        try:
            train_data = data_loader.load_training_data(eval_suite, dataset_name)
        except Exception as e:
            return 1, str(e)

        # Load test data using service
        test_data_list = None
        try:
            test_data_list = data_loader.load_test_data(eval_suite, dataset_name)
        except Exception:
            # Test data not available - that's okay
            pass

        # Note: Analytical metrics can have None expected outputs
        # The metric function will compute the expected value internally

        # Resolve types using service
        param_types = type_resolver.resolve_param_types(proc)

        # Get metric from eval suite
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

        # Create metric using service
        # Use metric_def.dataset_name if specified, otherwise fall back to dataset_name
        metric_dataset_name = metric_def.dataset_name or dataset_name
        metric_fn = metric_factory.create_metric(metric_def, proc, metric_dataset_name)

        # Call optimizer
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
            train_scores_dict[proc.name] = train_score
        except Exception as e:
            return 1, f"Optimization error for '{proc.name}': {e}"

    # Write results using service
    if optimized_dict:
        optimized_path = result_writer.write_optimized_prompts(optimized_dict)
        result_writer.write_training_scores(train_scores_dict)

        # Display results
        display_text = result_writer.format_prompts_for_display(optimized_dict, optimized_path)
        return 0, display_text

    return 0, ""  # pragma: no cover - successful optimization always records a result.


def run_evals_cli(
    source: str,
    *,
    adapter: str = "pydantic",
    adapter_model: str | None = None,
    codegen_model: str | None = None,
    optimizer: str = "mock",
    optimizer_model: str | None = None,
    reflection_model: str | None = None,
    optimize: bool = False,
    use_cache: bool = True,
    runtime_args: CliArgs | None = None,
) -> Tuple[int, str]:
    """Run eval suites. Set ``optimize=True`` only when not calling ``run_optimization_cli`` first
    (e.g. programmatic API). The CLI runs optimization in ``run_optimization_cli`` and passes
    ``optimize=False`` here so GEPA does not run twice.
    """
    from pathlib import Path

    source_path = Path(source).resolve()

    # Run optimizer FIRST if requested, before codegen
    if optimize:
        code, msg = run_optimization_cli(
            source,
            optimizer=optimizer,
            optimizer_model=optimizer_model,
            reflection_model=reflection_model,
            adapter=adapter,
            adapter_model=adapter_model,
            runtime_args=runtime_args,
        )
        if code != 0:
            return code, msg

    # Run codegen AFTER optimizer (if optimizer ran, DSPy is already configured with optimizer model)
    ok, err = maybe_run_codegen(source, model=codegen_model)
    if not ok:
        return 1, (err or "Code generation failed")

    kwargs: dict[str, Any] = {"adapter": _build_runtime_adapter(adapter, adapter_model)}
    if runtime_args is not None:
        kwargs["runtime_globals"] = _runtime_globals(runtime_args)
    results = run_evals(str(source_path), **kwargs)

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
        root = Path(source_path).resolve()
        loaded = load_kedi_source_bundle(root)
        prog = parse_program(
            loaded.text,
            source_path=str(root),
            source_map=loaded.source_map,
        )
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
        if isinstance(e, KediParseError):
            raise
        return False, str(e)


def run_program_cli(
    source: str,
    *,
    adapter: str,
    model: str | None,
    runtime_args: CliArgs | None = None,
) -> Tuple[int, str]:
    from pathlib import Path

    from .optimization.services.program_loader import load_program_with_optimized_prompts

    DEBUG = os.getenv("KEDI_DEBUG_OPTIMIZED", "0") == "1"

    source_path = Path(source).resolve()
    if DEBUG:
        print(f"[DEBUG run_program_cli] Source: {source_path}", file=sys.stderr)

    # Load program with optimized prompts injected
    program = load_program_with_optimized_prompts(source_path, debug=DEBUG)

    # Prefer offline execution when generated/native code fully covers the
    # program. Fall back to the configured adapter only if runtime actually
    # needs LLM-backed template expansion.
    rt = compile_program(program, adapter=None)
    _set_runtime_args(rt, runtime_args)
    try:
        out = rt.run_main()
        return 0, ("" if out is None else str(out))
    except RuntimeError as e:
        if str(e) != "No adapter provided to KediRuntime":
            raise

    adapter_impl = _build_runtime_adapter(adapter, model)
    rt = compile_program(program, adapter=adapter_impl)
    _set_runtime_args(rt, runtime_args)
    out = rt.run_main()
    return 0, ("" if out is None else str(out))
