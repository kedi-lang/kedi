"""Tests for optimize and auto directives."""

from __future__ import annotations

import pytest

from kedi.lang import parse_program
from kedi.optimization.mock import MockOptimizer


def test_parse_optimize_directive():
    """Test parsing > optimize: name: directive."""
    src = """
@foo(country):
  > optimize: find_capital:
    [capital] is the capital of <country>.
  
  > optimize: find_population:
    <capital> has [population: int] people.
  
  = Done
"""
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert proc.name == "foo"
    assert len(proc.optimize_spans) == 2
    assert proc.optimize_spans[0].name == "find_capital"
    assert proc.optimize_spans[1].name == "find_population"
    # Check template lines are captured
    assert len(proc.optimize_spans[0].template_lines) == 1
    assert len(proc.optimize_spans[1].template_lines) == 1


def test_parse_auto_directive():
    """Test parsing > auto: directive."""
    src = """
@summarize(texts: list[str]) -> str:
  > auto:
    Takes a list of texts and returns a concise summary.
    This should be part of the auto spec.
"""
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert proc.name == "summarize"
    assert proc.auto_spec is not None
    assert "Takes a list of texts" in proc.auto_spec.prompt
    assert "This should be part" in proc.auto_spec.prompt


def test_parse_auto_multiline():
    """Test parsing multiline > auto: directive."""
    src = """
@process(items: list[str]) -> str:
  > auto:
    Process a list of items.
    Return a formatted string.
    Handle edge cases.
"""
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert proc.auto_spec is not None
    assert "Process a list" in proc.auto_spec.prompt
    assert "Handle edge cases" in proc.auto_spec.prompt


def test_parse_unknown_directive_error():
    """Test that unknown directive raises error for explicit directives."""
    # This should fail because "data" is only valid in @eval blocks, not procedures
    # Actually, legacy `>` lines are treated as codegen prompts, so this would work
    # But let's test a truly unknown directive pattern
    # Phase-6 refactor: the legacy "any unrecognized ``>`` directive
    # becomes ``proc.ai_prompt``" fallback was an undocumented
    # backward-compat path. The tree-sitter grammar accepts only
    # ``> auto:`` and ``> optimize: <name>:`` inside a procedure body
    # and rejects everything else as a parse error. Callers should
    # migrate to ``> auto:``.
    src2 = """
@foo():
  > directive: name: arg:
    Some content
"""
    with pytest.raises(Exception):
        parse_program(src2)


def test_parse_optimize_empty_error():
    """Test that optimize with no indented block raises error."""
    src = """
@foo():
  > optimize: empty:
  
  = Done
"""
    with pytest.raises(ValueError, match="must have an indented block"):
        parse_program(src)


def test_parse_optimize_mixed_with_other_stmts():
    """Test optimize directive mixed with other statements."""
    src = """
@foo(x):
  [y] = start
  > optimize: span1:
    [result] is <y>.
  
  `print("debug")`
  > optimize: span2:
    Final result: <result>
  
  = done
"""
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert len(proc.optimize_spans) == 2
    assert proc.optimize_spans[0].name == "span1"
    assert proc.optimize_spans[1].name == "span2"


def test_auto_and_optimize_together():
    """Test procedure with both auto and optimize directives."""
    src = """
@complex_proc(x):
  > auto:
    This procedure needs codegen.
  
  > optimize: prompt1:
    [output] from <x>.
  
  = result
"""
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert proc.auto_spec is not None
    assert len(proc.optimize_spans) == 1
    assert proc.optimize_spans[0].name == "prompt1"


def test_mock_optimizer_interface():
    """Test MockOptimizer implements the new interface correctly."""
    optimizer = MockOptimizer()

    src = """
@foo(country):
  > optimize: find_capital:
    [capital] is the capital of <country>.
"""
    prog = parse_program(src)
    proc = prog.procedures[0]

    # Create mock training data
    train_data = [("France", "Paris"), ("Japan", "Tokyo")]

    # Create simple metric that returns 1.0 for all inputs
    def simple_metric(input_val, expected_output):
        return (1.0, None)

    # Call optimizer with new interface
    result, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types={},
        spans=proc.optimize_spans,
        train_data=train_data,
        test_data=None,
        metric=simple_metric,
    )

    # Should return optimized prompts (with improvements)
    assert isinstance(result, dict)
    assert len(result) == 1
    assert "find_capital" in result
    # Mock optimizer adds improvements to prompts
    assert "[capital]" in result["find_capital"] or "Think step by step" in result["find_capital"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_mock_optimizer_multi_span():
    """Test MockOptimizer with multiple optimize spans."""
    optimizer = MockOptimizer()

    src = """
@foo(country):
  > optimize: find_capital:
    [capital] is the capital of <country>.
  
  > optimize: find_population:
    <capital> has [population: int] people.
"""
    prog = parse_program(src)
    proc = prog.procedures[0]

    train_data = [("France", "Paris")]

    def simple_metric(input_val, expected_output):
        return (1.0, None)

    result, score = optimizer.optimize_procedure(
        proc_name=proc.name,
        proc_params=proc.params,
        param_types={},
        spans=proc.optimize_spans,
        train_data=train_data,
        test_data=None,
        metric=simple_metric,
    )

    assert len(result) == 2
    assert "find_capital" in result
    assert "find_population" in result


def test_optimizer_interface_empty_spans():
    """Test optimizer interface with empty spans."""
    optimizer = MockOptimizer()

    def simple_metric(input_val, expected_output):
        return (1.0, None)

    result, score = optimizer.optimize_procedure(
        proc_name="foo",
        proc_params=["x"],
        param_types={},
        spans=[],
        train_data=[("input", "output")],
        test_data=None,
        metric=simple_metric,
    )

    assert result == {}
    assert score == 0.0
