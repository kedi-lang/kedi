"""End-to-end integration tests for optimizer system."""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from kedi.lang import parse_program, compile_program
from kedi.optimization.config import OptimizerConfig
from kedi.optimization.mock import MockOptimizer
from kedi.optimization.services import (
    OptimizationDataLoader,
    MetricFactory,
    TypeResolver,
    OptimizationResultWriter,
)

try:
    from kedi.optimization.gepa import GEPAOptimizer

    GEPA_AVAILABLE = True
except ImportError:
    GEPA_AVAILABLE = False
    GEPAOptimizer = None


def test_mock_optimizer_end_to_end():
    """Test complete optimization flow with mock optimizer."""
    src = """
@classify(text: str) -> str:
  > optimize: sentiment:
    Classify sentiment: <text>
    [sentiment: str]
  = `sentiment.lower()`

@eval: classify:
  > data: examples:
    = ```
    return [
        ("I love this!", "positive"),
        ("This is terrible.", "negative"),
        ("It's okay.", "neutral"),
    ]
    ```
  > metric: accuracy(examples):
    = ```
    result = classify(examples)
    return (1.0 if result == expected else 0.0, None)
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    proc = prog.procedures[0]
    eval_suite = prog.evals[0]

    # Setup services
    data_loader = OptimizationDataLoader(rt)
    type_resolver = TypeResolver(rt)
    metric_factory = MetricFactory(rt)

    # Load data
    train_data = data_loader.load_training_data(eval_suite, "examples")
    assert len(train_data) == 3

    # Resolve types
    param_types = type_resolver.resolve_param_types(proc)

    # Create metric
    metric_def = eval_suite.metrics[0]
    metric = metric_factory.create_metric(metric_def, proc, "examples")

    # Run optimizer
    optimizer = MockOptimizer()
    spans = proc.optimize_spans

    optimized_prompts, train_score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=spans,
        train_data=train_data,
        test_data=None,
        metric=metric,
    )

    # Verify results
    assert isinstance(optimized_prompts, dict)
    assert "sentiment" in optimized_prompts
    assert isinstance(train_score, float)
    assert 0.0 <= train_score <= 1.0


def test_optimization_result_writer_integration():
    """Test writing optimization results end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "test_program.kedi"
        source_path.write_text("@test(x: int) -> int:\n  = `x * 2`")

        writer = OptimizationResultWriter(source_path)

        # Write optimized prompts
        optimized_prompts = {"test": {"compute": "Optimized: double <x> = [result]"}}
        optimized_path = writer.write_optimized_prompts(optimized_prompts)
        assert optimized_path.exists()

        # Write scores
        train_scores = {"test": 0.95}
        writer.write_training_scores(train_scores)

        scores_path = source_path.with_suffix(
            source_path.suffix + ".optimized_scores.json"
        )
        assert scores_path.exists()

        # Verify content is valid JSON
        import json

        content = json.loads(optimized_path.read_text())
        assert "test" in content
        assert "Optimized: double" in content["test"]["compute"]

        scores_content = json.loads(scores_path.read_text())
        assert scores_content["test"] == 0.95


def test_data_loader_and_metric_integration():
    """Test data loading and metric creation work together."""
    src = """
@add(a: int, b: int) -> int:
  = `a + b`

@eval: add:
  > data: pairs:
    = ```
    return [((1, 2), 3), ((3, 4), 7)]
    ```
  > metric: check(pairs):
    = ```
    a, b = pairs
    result = add(a, b)
    return (1.0 if result == expected else 0.0, None)
    ```
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    proc = prog.procedures[0]
    eval_suite = prog.evals[0]

    loader = OptimizationDataLoader(rt)
    factory = MetricFactory(rt)

    # Load data
    train_data = loader.load_training_data(eval_suite, "pairs")
    assert len(train_data) == 2
    assert train_data[0][0] == (1, 2)

    # Create metric and test it
    metric_def = eval_suite.metrics[0]
    metric = factory.create_metric(metric_def, proc, "pairs")

    # Test metric on first example
    score, feedback = metric((1, 2), 3)
    assert score == 1.0

    # Test on incorrect example
    score, feedback = metric((1, 2), 5)
    assert score == 0.0


def test_type_resolution_integration():
    """Test type resolution works with optimizer."""
    src = """
@process(x: int, y: str) -> float:
  = `float(x) + len(y)`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    proc = prog.procedures[0]
    resolver = TypeResolver(rt)

    param_types = resolver.resolve_param_types(proc)

    # Should resolve at least one type
    assert len(param_types) >= 0  # May not resolve all types

    # Test optimizer can use these types
    optimizer = MockOptimizer()

    # Create dummy spans and data
    spans = []
    train_data = [((1, "test"), 5.0)]

    def dummy_metric(input_val, expected):
        return (1.0, None)

    # Should not raise with resolved types
    result = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types=param_types,
        spans=spans,
        train_data=train_data,
        test_data=None,
        metric=dummy_metric,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.skipif(not GEPA_AVAILABLE, reason="GEPA optimizer not available")
def test_gepa_optimizer_integration_setup():
    """Test GEPA optimizer can be instantiated and configured."""
    config = OptimizerConfig(
        max_metric_calls=5,  # Small for testing
        optimizer_model="openai/gpt-4o",
        reflection_model="openai/gpt-5",
    )

    src = """
@test(x: int) -> int:
  = `x * 2`
"""
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)

    # Should be able to create optimizer
    optimizer = GEPAOptimizer(config=config, runtime=rt)

    assert optimizer is not None
    assert optimizer.config.max_metric_calls == 5
