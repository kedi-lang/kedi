from __future__ import annotations

from typing import Any, Mapping

from kedi.agent_adapter.history_provenance import LINEAGE_STATE_KEY, HistoryLineage
from kedi.agent_adapter.request_budget import (
    HISTORY_USAGE_KEY,
    HistoryUsageCheckpoint,
    estimate_content_tokens,
    measured_history_tokens,
)


def estimate_history_tokens(state: Mapping[str, Any]) -> int:
    from langchain_core.messages import AIMessage, ToolMessage, messages_from_dict

    try:
        messages = messages_from_dict(state.get("langchain_messages", []))
    except (KeyError, TypeError, ValueError):
        return estimate_content_tokens(state)
    projected = []
    opaque_tokens = 0
    for message in messages:
        content = message.content
        if isinstance(message, AIMessage) and message.usage_metadata:
            output_tokens = message.usage_metadata["output_tokens"]
            if output_tokens > 0 and isinstance(content, list):
                content = [dict(block) if isinstance(block, dict) else block for block in content]
                hidden = False
                for block in content:
                    if not isinstance(block, dict) or block.get("type") not in {
                        "reasoning",
                        "thinking",
                    }:
                        continue
                    for key in ("signature", "encrypted_content"):
                        if block.get(key):
                            block[key] = "[provider continuation]"
                            hidden = True
                if hidden:
                    opaque_tokens += output_tokens
        item: dict[str, Any] = {
            "type": message.type,
            "content": content,
            "additional_kwargs": message.additional_kwargs,
            "name": message.name,
        }
        if isinstance(message, AIMessage):
            item["tool_calls"] = message.tool_calls
            item["invalid_tool_calls"] = message.invalid_tool_calls
        if isinstance(message, ToolMessage):
            item["tool_call_id"] = message.tool_call_id
        projected.append(item)
    # Keep local provenance available without charging it as model-visible text.
    extra = {
        key: value
        for key, value in state.items()
        if key != "langchain_messages"
        and not (key == LINEAGE_STATE_KEY and isinstance(value, HistoryLineage))
        and not (key == HISTORY_USAGE_KEY and isinstance(value, HistoryUsageCheckpoint))
    }
    measured = measured_history_tokens(state, "langchain_messages")
    if measured is not None:
        return measured + estimate_content_tokens(extra)
    return estimate_content_tokens(projected, extra) + opaque_tokens
