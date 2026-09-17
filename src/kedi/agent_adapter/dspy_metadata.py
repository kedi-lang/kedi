from __future__ import annotations

from ..agent_profile.settings import DSPY_LM_SETTING_KEYS
from .meta import AdapterCapabilities, AdapterDescriptor

DSPY_ADAPTER = AdapterDescriptor(
    kind="agent-framework",
    shortname="dspy",
    capabilities=AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_subagents=True,
        supports_background_subagents=False,
        supports_artifacts=True,
        supports_native_artifacts=False,
        supports_stateful_history=False,
        supports_history_replay=False,
        supports_stream_events=False,
        settings=DSPY_LM_SETTING_KEYS,
    ),
)
