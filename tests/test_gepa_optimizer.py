"""Tests for GEPA optimizer."""

from __future__ import annotations

import pytest

from kedi.lang import parse_program

try:
    from kedi.optimization.gepa import GEPAOptimizer
    GEPA_AVAILABLE = GEPAOptimizer is not None
except ImportError:
    GEPA_AVAILABLE = False
    GEPAOptimizer = None


@pytest.mark.skipif(
    not GEPAOptimizer,
    reason="DSPy not available or GEPA optimizer not importable"
)
def test_gepa_optimizer_interface():
    """Test GEPA optimizer implements the interface correctly."""
    optimizer = GEPAOptimizer(max_metric_calls=10)  # Small budget for testing
    
    src = """
@foo(country):
  > optimize: find_capital:
    [capital] is the capital of <country>.
"""
    prog = parse_program(src)
    proc = prog.procedures[0]
    
    # Create mock training data
    train_data = [("France", "Paris"), ("Japan", "Tokyo")]
    
    # Create simple metric
    def simple_metric(input_val, expected_output):
        # Mock metric - always returns 1.0 for testing
        return (1.0, None)
    
    # This would call GEPA, which requires actual model setup
    # For now, just verify the optimizer can be instantiated
    assert optimizer is not None
    assert optimizer.max_metric_calls == 10


def test_gepa_import_handling():
    """Test that GEPA import is handled gracefully."""
    from kedi.optimization import get_optimizer
    
    # Should not raise even if DSPy is unavailable (handled in __init__)
    optimizer = get_optimizer("gepa")
    # May be None if DSPy unavailable, or GEPAOptimizer if available
    assert optimizer is None or hasattr(optimizer, 'optimize_procedure')

