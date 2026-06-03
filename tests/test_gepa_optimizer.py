"""Tests for GEPA optimizer."""

from __future__ import annotations

import pytest

from kedi.lang import parse_program

try:
    from kedi.optimization.gepa import GEPAOptimizer

    GEPA_AVAILABLE = GEPAOptimizer is not None
except ImportError:
    GEPA_AVAILABLE = False
    GEPAOptimizer = None  # type: ignore[assignment]


@pytest.mark.skipif(not GEPAOptimizer, reason="DSPy not available or GEPA optimizer not importable")
def test_gepa_optimizer_interface():
    """Test GEPA optimizer implements the interface correctly."""
    from kedi.optimization.config import OptimizerConfig

    # Create config with small budget for testing
    config = OptimizerConfig(max_metric_calls=10)
    optimizer = GEPAOptimizer(config=config, runtime=None)

    src = """
@foo(country):
  > optimize: find_capital:
    [capital] is the capital of <country>.
"""
    prog = parse_program(src)
    prog.procedures[0]

    # Create mock training data

    # Create simple metric
    def simple_metric(input_val, expected_output):
        # Mock metric - always returns 1.0 for testing
        return (1.0, None)

    # This would call GEPA, which requires actual model setup
    # For now, just verify the optimizer can be instantiated
    assert optimizer is not None
    assert optimizer.config.max_metric_calls == 10


def test_gepa_import_handling():
    """Test that GEPA import is handled gracefully."""
    from kedi.optimization import get_optimizer

    # get_optimizer("gepa") should return None since GEPA requires config
    # (it's instantiated directly in app.py with proper config)
    optimizer = get_optimizer("gepa")
    # Should be None because GEPA can't be instantiated without config
    assert optimizer is None
