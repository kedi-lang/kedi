from __future__ import annotations

from typing import (
    Any,
    Sequence,
    cast,
)

from pydantic_ai.messages import (
    CompactionPart,
)

from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.subagents import SubagentConversationContext
from kedi.decisions import capture_decision_evidence


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


def _capture_typesafe_decision_evidence(result: Any) -> None:
    """Publish metadata from the final TypeSafe model response for this run only."""
    new_messages = getattr(result, "new_messages", None)
    if not callable(new_messages):
        return
    messages = cast(Sequence[Any], new_messages())
    for message in reversed(messages):
        details = getattr(message, "provider_details", None)
        if not isinstance(details, dict):
            details = getattr(message, "metadata", None)
        if not isinstance(details, dict):
            continue
        metadata = details.get("typesafe")
        if not isinstance(metadata, dict):
            continue
        model = getattr(message, "model_name", None)
        capture_decision_evidence(
            provider="typesafe",
            model=model if isinstance(model, str) else None,
            metadata=metadata,
        )
        return
