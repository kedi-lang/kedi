from __future__ import annotations

from typing import (
    Any,
)

from pydantic_ai.messages import (
    CompactionPart,
)

from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.subagents import SubagentConversationContext


def _capture_conversation_messages(
    conversation: AdapterConversationContext | SubagentConversationContext | None,
    result: Any,
) -> None:
    if conversation is None:
        return
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    conversation.result_state["pydantic_messages"] = ModelMessagesTypeAdapter.dump_python(
        result.all_messages(),
        mode="json",
    )
    if isinstance(conversation, AdapterConversationContext):
        conversation.mark_completed()
    if isinstance(conversation, AdapterConversationContext) and any(
        isinstance(part, CompactionPart)
        for message in result.new_messages()
        for part in message.parts
    ):
        conversation.cache_prefix_changed = True


def _apply_conversation_history(
    run_kwargs: dict[str, object],
    conversation: AdapterConversationContext | SubagentConversationContext | None,
) -> None:
    if (
        conversation is None
        or run_kwargs.get("message_history") is not None
        or "pydantic_messages" not in conversation.resume_state
    ):
        return
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    run_kwargs["message_history"] = ModelMessagesTypeAdapter.validate_python(
        conversation.resume_state["pydantic_messages"]
    )
