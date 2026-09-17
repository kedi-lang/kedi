from __future__ import annotations

from ..errors import KediExecutionError


class AgentProfileOverrideUnsupported(KediExecutionError):
    """Raised when the active adapter cannot apply an agent profile override."""


class AgentToolOverrideUnsupported(KediExecutionError):
    """Raised when the active adapter cannot register tools for an agent run."""


class AgentSubagentUnsupported(KediExecutionError):
    """Raised when the active adapter cannot delegate to Kedi subagents."""
