"""Tests for optimizer interface validations."""

from __future__ import annotations

import pytest

from kedi.lang import parse_program
from kedi.optimization.mock import MockOptimizer


def test_one_metric_per_procedure():
    """Test that parser enforces one metric per procedure."""
    src = """
@foo(x):
  > optimize: test:
    [y] from <x>
  = done

@eval: foo:
  > data: inputs:
    = `[("a", "A"), ("b", "B")]`
  > metric: m1(inputs):
    = `1.0`
  > metric: m2(inputs):
    = `1.0`
"""
    # Should fail because multiple metrics are defined
    with pytest.raises(ValueError, match="Multiple metrics defined"):
        parse_program(src)


def test_one_dataset_per_metric():
    """Test that each metric references exactly one dataset."""
    src = """
@foo(x):
  > optimize: test:
    [y] from <x>
  = done

@eval: foo:
  > data: inputs:
    = `[("a", "A"), ("b", "B")]`
  > metric: m1(inputs):
    = `1.0`
"""
    prog = parse_program(src)
    eval_suite = prog.evals[0]  # type: ignore

    # Should have exactly one metric
    assert len(eval_suite.metrics) == 1

    # Metric should reference the dataset
    metric = eval_suite.metrics[0]
    assert metric.dataset_name == "inputs"
    assert metric.dataset_name in eval_suite.datasets


def test_mock_optimizer_type_validation_single_param():
    """Test MockOptimizer validates types for single-param procedures."""
    from kedi.lang.ast import OptimizeSpan, TemplateLine, TextSegment

    optimizer = MockOptimizer()

    def simple_metric(input_val, expected_output):
        return (1.0, None)

    # Create a span for testing
    span = OptimizeSpan(
        name="test_span",
        template_lines=[TemplateLine(segments=[TextSegment(text="[y] from <x>")])],
    )

    # Should fail type validation
    with pytest.raises(TypeError, match="expected int"):
        optimizer.optimize_procedure(
            proc_name="foo",
            proc_params=["x"],
            param_types={"x": int},
            spans=[span],
            train_data=[("not an int", "output")],
            test_data=None,
            metric=simple_metric,
        )


def test_mock_optimizer_type_validation_multi_param():
    """Test MockOptimizer validates types for multi-param procedures."""
    from kedi.lang.ast import OptimizeSpan, TemplateLine, TextSegment

    optimizer = MockOptimizer()

    def simple_metric(input_val, expected_output):
        return (1.0, None)

    # Create a span for testing
    span = OptimizeSpan(
        name="test_span",
        template_lines=[
            TemplateLine(segments=[TextSegment(text="[result] from <x> and <y>")])
        ],
    )

    # Should fail type validation - wrong tuple length
    with pytest.raises(TypeError, match="Expected tuple of 2 values"):
        optimizer.optimize_procedure(
            proc_name="foo",
            proc_params=["x", "y"],
            param_types={"x": str, "y": int},
            spans=[span],
            train_data=[("single_value", "output")],  # Missing y - not a tuple
            test_data=None,
            metric=simple_metric,
        )

    # Should fail type validation - wrong type
    with pytest.raises(TypeError, match="expected int"):
        optimizer.optimize_procedure(
            proc_name="foo",
            proc_params=["x", "y"],
            param_types={"x": str, "y": int},
            spans=[span],
            train_data=[(("correct", "wrong_type"), "output")],
            test_data=None,
            metric=simple_metric,
        )


def test_mock_optimizer_empty_train_data():
    """Test MockOptimizer raises ValueError for empty train_data."""
    optimizer = MockOptimizer()

    def simple_metric(input_val, expected_output):
        return (1.0, None)

    with pytest.raises(ValueError, match="No training data"):
        optimizer.optimize_procedure(
            proc_name="foo",
            proc_params=["x"],
            param_types={},
            spans=[],
            train_data=[],
            test_data=None,
            metric=simple_metric,
        )
