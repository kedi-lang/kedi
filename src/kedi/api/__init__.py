"""Public exports; implementations live in the owning submodules."""

from __future__ import annotations

from kedi.agent_adapter.approvals.policy import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRequest,
)

from .cache import CacheInfo, cache_info, clear_cache
from .configuration import configure, reset_config
from .contexts import context, parallel, session
from .contexts import interactive as interactive
from .decorators import bind, query
from .registration import approval, on, tool, type
from .state import AdapterLike, AgentName, FrameworkAdapterName
from .state import ClassT as ClassT
from .state import P as P
from .state import R as R

__all__ = [
    "AdapterLike",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalMode",
    "ApprovalPolicy",
    "ApprovalRequest",
    "AgentName",
    "CacheInfo",
    "FrameworkAdapterName",
    "bind",
    "approval",
    "cache_info",
    "clear_cache",
    "configure",
    "context",
    "on",
    "parallel",
    "query",
    "reset_config",
    "session",
    "tool",
    "type",
]
