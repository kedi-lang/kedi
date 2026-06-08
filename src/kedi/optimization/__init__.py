"""Optimizer infrastructure for prompt optimization."""

from __future__ import annotations

from .interfaces import OptimizerAdapter
from .mock import MockOptimizer

GEPAOptimizer: type[OptimizerAdapter] | None

try:
    from .gepa import GEPAOptimizer as _GEPAOptimizerImpl

    GEPAOptimizer = _GEPAOptimizerImpl
    GEPA_AVAILABLE = True
except ImportError:
    GEPAOptimizer = None
    GEPA_AVAILABLE = False


def get_optimizer(name: str) -> OptimizerAdapter | None:
    """Get optimizer by name.

    Args:
        name: Optimizer name ('mock', 'gepa', etc.)

    Returns:
        OptimizerAdapter instance or None if not found

    Note:
        GEPA optimizer requires OptimizerConfig and runtime, so it cannot
        be instantiated via this function. It should be instantiated directly
        in app.py with proper configuration.
    """
    optimizers: dict[str, type[OptimizerAdapter]] = {
        "mock": MockOptimizer,
    }

    # GEPA requires config and runtime, so don't include it here
    # It's handled specially in app.py

    optimizer_class = optimizers.get(name.lower())
    if optimizer_class is None:
        return None
    return optimizer_class()
