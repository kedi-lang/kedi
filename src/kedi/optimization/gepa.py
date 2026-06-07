"""GEPA optimizer for prompt optimization using DSPy."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import dspy

from ..lang.ast import OptimizeSpan
from ..lang.compiler.environment import KediRuntime
from ..model_normalization import normalize_for_litellm
from .config import OptimizerConfig
from .interfaces import OptimizerAdapter
from .services.dspy_config import DSPyConfigurationService
from .services.dspy_example_formatter import DSPyExampleFormatter
from .services.dspy_module_builder import DSPyModuleBuilder
from .services.dspy_prompt_extractor import DSPyPromptExtractor
from .services.dspy_score_calculator import DSPyScoreCalculator
from .services.gepa_metric_wrapper import GEPAMetricWrapper
from .utils import normalize_optimized_prompt_prefix, span_to_instruction

_GEPA_STATE_NAME = "gepa_state.bin"

DSPY_AVAILABLE = True


class GEPAOptimizer(OptimizerAdapter):
    """GEPA optimizer using DSPy's GEPA for reflective prompt evolution.

    GEPA uses LLM reflection to evolve prompts based on execution traces
    and feedback. This optimizer bridges Kedi procedures to DSPy modules.

    Uses OptimizerConfig and service layer for all operations.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        runtime: KediRuntime | None = None,
        source_path: Path | None = None,
    ):
        """Initialize GEPA optimizer.

        Args:
            config: Optimizer configuration with model and token settings
            runtime: Optional Kedi runtime for structured type handling
            source_path: Path to source .kedi file for saving optimized prompts
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for GEPA optimizer. Install with: uv add dspy-ai")

        self.config = config
        self.runtime = runtime
        self.source_path = source_path

        # Initialize services
        self.dspy_config_service = DSPyConfigurationService(config)
        self.module_builder = DSPyModuleBuilder(runtime=cast(KediRuntime, runtime))
        self.example_formatter = DSPyExampleFormatter(cast(KediRuntime, runtime))
        self.metric_wrapper = GEPAMetricWrapper()
        self.prompt_extractor = DSPyPromptExtractor()
        self.score_calculator = DSPyScoreCalculator()

        # Initialize result writer if source path provided
        if source_path:
            from .services import OptimizationResultWriter

            self.result_writer = OptimizationResultWriter(source_path)
        else:
            self.result_writer = None

    def _gepa_log_dir(self, proc_name: str, span_name: str) -> Path | None:
        """Per-span directory for DSPy GEPA ``log_dir`` (``gepa_state.bin`` resume)."""
        if not self.source_path:
            return None
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"{proc_name}__{span_name}")
        return self.source_path.parent / f"{self.source_path.name}.gepa" / safe

    def optimize_procedure(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, type],
        spans: list[OptimizeSpan],
        train_data: Iterable[tuple[Any, Any]],
        test_data: Iterable[tuple[Any, Any]] | None,
        metric: Callable[[Any, Any], tuple[float, str | None]],
    ) -> tuple[dict[str, str], float]:
        """Optimize procedure spans using GEPA.

        Args:
            proc_name: Procedure name
            proc_params: List of procedure parameter names
            param_types: Mapping of parameter name -> Python type
            spans: List of optimize spans to optimize
            train_data: Iterable of (input, expected_output) tuples
            test_data: Optional test dataset
            metric: Function (input, expected_output) -> (score, feedback)

        Returns:
            Tuple of (optimized_prompts, train_score)
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for GEPA optimizer")

        if not spans:
            return {}, 0.0

        train_list = list(train_data)
        if not train_list:
            raise ValueError(f"No training data provided for '{proc_name}'")

        # Validate types
        self._validate_types(proc_name, proc_params, param_types, train_list)

        # Verify DSPy is configured
        self.dspy_config_service.verify_configured()
        test_list = list(test_data) if test_data is not None else None

        # Configure reflection LM if provided in config
        reflection_lm = None
        if self.config.reflection_model:
            print("\n🤔 Configuring reflection LM...")
            print(f"   Model: {self.config.reflection_model}")
            max_tokens = self.config.get_reflection_max_tokens()
            print(f"   Max tokens: {max_tokens}")
            print("   Temperature: 1.0")

            base_lm = dspy.LM(
                model=normalize_for_litellm(self.config.reflection_model),
                temperature=1.0,
                max_tokens=max_tokens,
                timeout=300,
            )

            # Wrap the LM to add call tracing
            # Note: DSPy wraps this as lambda x: reflection_lm(x)[0], so we need to ensure
            # our wrapper is compatible with that interface
            reflection_call_count = {"count": 0}

            class TracedReflectionLM:
                """Wrapper that traces reflection LM calls while maintaining DSPy compatibility."""

                def __init__(self, base_lm, model_name, call_count):
                    self.base_lm = base_lm
                    self.model_name = model_name
                    self.model = model_name  # DSPy / logging expect .model
                    self.call_count = call_count

                def __call__(self, prompt):
                    self.call_count["count"] += 1
                    print(
                        f"\n💭 Reflection LM call #{self.call_count['count']}",
                        flush=True,
                    )
                    print(f"   Model: {self.model_name}", flush=True)
                    result = self.base_lm(prompt)
                    print("   ✅ Reflection complete", flush=True)
                    return result

            reflection_lm = TracedReflectionLM(
                base_lm, self.config.reflection_model, reflection_call_count
            )
            print("   ✅ Reflection LM created successfully (with call tracing)")
        else:
            print("\n⚠️  No reflection model configured - GEPA will use default LM for reflection")

        optimized_prompts: dict[str, str] = {}
        span_scores: list[float] = []

        # Preserve original optimized prefixes so we can restore after optimization
        original_prefixes: dict[str, str | None] = {}
        for span in spans:
            if span.template_lines:
                original_prefixes[span.name] = span.template_lines[0].optimized_prompt_prefix
            else:
                original_prefixes[span.name] = None

        try:
            for span_idx, span in enumerate(spans, 1):
                print(f"\n{'=' * 80}")
                print(f"GEPA Optimization: Span {span_idx}/{len(spans)} - '{span.name}'")
                print(f"{'=' * 80}")

                # Prepare training/validation examples
                print("📊 Preparing datasets...")
                train_examples = self.example_formatter.format_examples(
                    train_list, proc_params, span=span, proc_name=proc_name
                )

                if test_list:
                    test_list_full = list(test_list)
                    # Trim test set if max_validation_examples is configured
                    if self.config.max_validation_examples is not None:
                        test_list_full = test_list_full[: self.config.max_validation_examples]
                    val_examples = self.example_formatter.format_examples(
                        test_list_full, proc_params, span=span, proc_name=proc_name
                    )
                elif len(train_examples) >= 4:
                    val_examples = train_examples[: max(1, len(train_examples) // 4)]
                else:
                    val_examples = train_examples

                print(f"   Training examples: {len(train_examples)}")
                print(f"   Validation examples: {len(val_examples)}")
                print(f"   Max metric calls (from config): {self.config.max_metric_calls}")

                print(f"\n🔧 Building DSPy module for span '{span.name}'...")
                dspy_module = self.module_builder.build_module(span, proc_params, param_types)

                # Metric used during GEPA optimization - points at the module being optimized
                # Obtain proc_def for sibling span overrides during metric evaluation
                print("📝 Wrapping metric function...")
                proc_def_for_wrapper = self._get_proc_def(proc_name)
                gepa_metric = self.metric_wrapper.wrap_metric(
                    metric,
                    proc_params,
                    span,  # Pass span as positional arg
                    module_ref=dspy_module,
                    proc_def=proc_def_for_wrapper,
                )

                print("\n🚀 Starting GEPA optimization...")
                print(f"   Optimizer model: {self.config.optimizer_model or 'default'}")
                print(f"   Reflection model: {self.config.reflection_model or 'default'}")
                print(f"   Reflection LM object: {reflection_lm}")
                if reflection_lm:
                    print(f"   Reflection LM model: {getattr(reflection_lm, 'model', 'unknown')}")
                print(f"\n{'─' * 80}")

                print("\n🔨 Creating GEPA optimizer...")
                print(f"   Passing reflection_lm to GEPA: {reflection_lm}")

                # Verify dspy.settings is configured correctly
                if hasattr(dspy.settings, "lm") and dspy.settings.lm:
                    model_name = getattr(dspy.settings.lm, "model", "unknown")
                    lm_kwargs = getattr(dspy.settings.lm, "kwargs", {})
                    print(f"   DSPy settings.lm: {model_name}")
                    print(f"   DSPy settings.lm kwargs: {lm_kwargs}")
                else:
                    print("   ⚠️  WARNING: dspy.settings.lm is not configured!")

                gepa_log_dir = self._gepa_log_dir(proc_name, span.name)
                if gepa_log_dir is not None:
                    gepa_log_dir.mkdir(parents=True, exist_ok=True)
                    state_file = gepa_log_dir / _GEPA_STATE_NAME
                    if state_file.exists():
                        print(
                            f"   GEPA checkpoint: resuming from {gepa_log_dir}\n"
                            f"   If optimization ends immediately, raise "
                            f"--optimizer-max-metric-calls above the prior total eval-units.",
                            flush=True,
                        )

                optimizer = dspy.GEPA(
                    metric=gepa_metric,
                    max_metric_calls=self.config.max_metric_calls,
                    reflection_lm=cast(Any, reflection_lm),
                    reflection_minibatch_size=self.config.reflection_minibatch_size,
                    track_stats=True,
                    log_dir=str(gepa_log_dir) if gepa_log_dir is not None else None,
                )
                print(f"   Reflection minibatch size: {self.config.reflection_minibatch_size}")
                print("   ✅ GEPA optimizer created successfully")

                est_seconds = len(train_examples) * 25
                print(
                    f"\n⏳ Starting compile ({len(train_examples)} examples, "
                    f"budget={self.config.max_metric_calls} calls, "
                    f"est ~{est_seconds // 60}m{est_seconds % 60:02d}s for baseline)..."
                )
                print(f"{'─' * 80}")

                compile_start = time.monotonic()
                optimized_module = optimizer.compile(
                    student=dspy_module,
                    trainset=train_examples,
                    valset=val_examples,
                )
                compile_elapsed = time.monotonic() - compile_start
                compile_min = int(compile_elapsed) // 60
                compile_sec = int(compile_elapsed) % 60

                metric_calls = 0
                progress = getattr(gepa_metric, "_progress", None)
                if isinstance(progress, dict):
                    metric_calls = int(progress.get("count", 0))

                print(f"{'─' * 80}")
                print(f"✅ GEPA optimization complete for span '{span.name}'")
                print(
                    f"   Duration: {compile_min}m{compile_sec:02d}s  |  "
                    f"DSPy metric callbacks: {metric_calls}  "
                    f"(GEPA budget: {self.config.max_metric_calls} eval-units; "
                    f"DSPy may call the metric more than once per unit)"
                )

                # Extract optimized prompt for this span
                print("\n🔍 Extracting optimized prompt...")
                previous_instruction = span_to_instruction(span)
                extracted = self.prompt_extractor.extract_prompts(optimized_module, [span])
                optimized_prompt = normalize_optimized_prompt_prefix(span, extracted.get(span.name))
                if optimized_prompt is not None:
                    optimized_prompts[span.name] = optimized_prompt
                    if span.template_lines:
                        span.template_lines[0].optimized_prompt_prefix = optimized_prompt or None

                    effective_instruction = span_to_instruction(span)
                    print(f"   Previous seed length: {len(previous_instruction)} chars")
                    print(f"   Optimized prefix length: {len(optimized_prompt)} chars")
                    print(f"   Effective prompt length: {len(effective_instruction)} chars")

                    if effective_instruction == previous_instruction:
                        print("   ⚠️  Prompt appears unchanged")
                        warnings.warn(
                            f"Optimized prompt for '{span.name}' appears unchanged. "
                            "GEPA may not have optimized this span.",
                            UserWarning,
                        )
                    else:
                        print("   ✨ Prompt successfully optimized")

                    # Save optimized prompts immediately after extraction, before scoring
                    # This ensures prompts are saved even if scoring fails or is interrupted
                    self._save_prompts_immediately(proc_name, optimized_prompts)
                else:
                    print("   ❌ No optimized prompt found")
                    warnings.warn(f"No optimized prompt found for span '{span.name}'", UserWarning)
                    continue

                # Re-wrap metric with the optimized module for scoring
                print("\n📈 Calculating final score on training set...")
                scoring_metric = self.metric_wrapper.wrap_metric(
                    metric,
                    proc_params,
                    span,  # Pass span as positional arg
                    module_ref=optimized_module,
                    proc_def=proc_def_for_wrapper,
                )

                span_score = self.score_calculator.calculate_score(
                    optimized_module,
                    train_examples,
                    scoring_metric,
                )
                span_scores.append(span_score)
                print(f"   Training score: {span_score:.4f}")
                print(f"{'=' * 80}\n")
        finally:
            # Restore original prefixes so caller gets pristine AST
            for span in spans:
                if not span.template_lines:
                    continue
                original_prefix = original_prefixes.get(span.name)
                span.template_lines[0].optimized_prompt_prefix = original_prefix

        overall_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
        return optimized_prompts, overall_score

    def _get_proc_def(self, proc_name: str):
        """Best-effort retrieval of ProcedureDef from runtime for the given name."""
        if not self.runtime:
            return None
        try:
            fn = self.runtime._procedures.get(proc_name)
            if not fn:
                return None
            import inspect

            if hasattr(fn, "__code__"):
                closure_vars = inspect.getclosurevars(fn)
                proc = closure_vars.nonlocals.get("proc") or closure_vars.globals.get("proc")
                return proc
        except Exception:
            return None
        return None

    def _save_prompts_immediately(
        self,
        proc_name: str,
        optimized_prompts: dict[str, str],
    ) -> None:
        """Save optimized prompts immediately after extraction.

        This ensures prompts are saved even if scoring fails or is interrupted.
        The JSON file will be updated incrementally as each span is optimized.

        Args:
            proc_name: Name of the procedure being optimized
            optimized_prompts: Dict mapping span_name -> optimized_prompt
        """
        if not self.result_writer:
            return

        try:
            # Load existing optimized prompts
            import json

            assert self.source_path is not None
            optimized_path = self.source_path.with_suffix(
                self.source_path.suffix + ".optimized.json"
            )
            existing_prompts: dict[str, dict[str, str]] = {}

            if optimized_path.exists():
                try:
                    with open(optimized_path, encoding="utf-8") as f:
                        existing_prompts = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_prompts = {}

            # Update with new prompts for this procedure
            existing_prompts[proc_name] = optimized_prompts

            # Write back to file
            self.result_writer.write_optimized_prompts(existing_prompts)
            print(f"   💾 Saved optimized prompts to {optimized_path.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to save prompts: {e}")

    def _validate_types(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, type],
        train_data: list[tuple[Any, Any]],
    ) -> None:
        """Basic type validation."""
        if not param_types:
            return

        for idx, example in enumerate(train_data):
            if not isinstance(example, tuple) or len(example) != 2:
                continue

            input_val, _ = example

            # Skip type validation - the metric will handle type checking
            # This allows flexible data formats like tuples to be passed through
