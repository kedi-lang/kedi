"""Optimizer infrastructure for prompt optimization."""

from __future__ import annotations

from .interfaces import OptimizerAdapter
from .mock import MockOptimizer

try:
    from .gepa import GEPAOptimizer
    GEPA_AVAILABLE = True
except ImportError:
    GEPA_AVAILABLE = False
    GEPAOptimizer = None  # type: ignore


def get_optimizer(name: str) -> OptimizerAdapter | None:
    """Get optimizer by name.
    
    Args:
        name: Optimizer name ('mock', 'gepa', etc.)
    
    Returns:
        OptimizerAdapter instance or None if not found
    """
    optimizers: dict[str, type[OptimizerAdapter]] = {
        "mock": MockOptimizer,
    }
    
    if GEPA_AVAILABLE and GEPAOptimizer:
        optimizers["gepa"] = GEPAOptimizer
    
    optimizer_class = optimizers.get(name.lower())
    if optimizer_class is None:
        return None
    return optimizer_class()

