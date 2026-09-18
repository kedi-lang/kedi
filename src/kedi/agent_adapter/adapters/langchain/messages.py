from __future__ import annotations

from typing import (
    Any,
    Mapping,
    NoReturn,
    Sequence,
)

from kedi.agent_adapter.codemode import CodeModeError
from kedi.agent_adapter.events import AgentMessageResolver
from kedi.agent_adapter.invocation import InvocationUsage
from kedi.agent_adapter.prefix_cache import langchain_provider_cache_policy
from kedi.effort import pydantic_effort_value
from kedi.decisions import capture_decision_evidence


def _langchain_uses_explicit_cache_blocks(model: Any | None) -> bool:
    if model is None:
        return False
    return langchain_provider_cache_policy(model).backend in {"anthropic", "openrouter"}


def _langchain_with_cache_control(message: Any) -> Any:
    content = message.content
    if isinstance(content, str):
        blocks: list[Any] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list):
        blocks = [dict(block) if isinstance(block, Mapping) else block for block in content]
        for index in range(len(blocks) - 1, -1, -1):
            block = blocks[index]
            if isinstance(block, dict) and block.get("type") == "text":
                block.setdefault("cache_control", {"type": "ephemeral"})
                break
        else:
            return message
    else:
        return message
    return message.model_copy(update={"content": blocks})


def _langchain_without_cache_control(message: Any) -> Any:
    content = message.content
    if not isinstance(content, list):
        return message
    changed = False
    blocks: list[Any] = []
    for block in content:
        if isinstance(block, Mapping) and "cache_control" in block:
            normalized = dict(block)
            normalized.pop("cache_control", None)
            blocks.append(normalized)
            changed = True
        else:
            blocks.append(block)
    if not changed:
        return message
    return message.model_copy(update={"content": blocks})


def _raise_langchain_codemode_error(exc: CodeModeError) -> NoReturn:
    from langchain_core.tools import ToolException

    raise ToolException(str(exc)) from exc


def _langchain_event_handler(resolver: AgentMessageResolver) -> Any:
    from langchain_core.callbacks import AsyncCallbackHandler

    class KediAgentEventHandler(AsyncCallbackHandler):
        async def on_llm_end(
            self,
            response: Any,
            *,
            run_id: Any,
            **_kwargs: Any,
        ) -> None:
            generations = getattr(response, "generations", ())
            if not isinstance(generations, Sequence):
                return
            for group_index, group in enumerate(generations):
                if not isinstance(group, Sequence):
                    continue
                for generation_index, generation in enumerate(group):
                    message = getattr(generation, "message", None)
                    text = _langchain_completed_message_text(message)
                    if text:
                        resolver.complete_message(
                            f"langchain:{run_id}:{group_index}:{generation_index}",
                            text,
                        )
                        resolver.mark_message_boundary()
                    tool_calls = getattr(message, "tool_calls", None)
                    if isinstance(tool_calls, Sequence) and tool_calls:
                        resolver.mark_tool_boundary()

        async def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            **_kwargs: Any,
        ) -> None:
            del serialized, input_str, run_id
            resolver.mark_tool_boundary()

    return KediAgentEventHandler()


def _langchain_completed_message_text(message: Any) -> str:
    return (_langchain_content_text(getattr(message, "content", None)) or "").strip()


def _langchain_invocation_usage(
    result: Any,
    *,
    input_messages: object = None,
) -> InvocationUsage:
    if not isinstance(result, Mapping):
        return InvocationUsage(requests=1)
    messages = result.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return InvocationUsage(requests=1)
    messages = _langchain_current_messages(messages, input_messages)

    input_tokens = cache_read_tokens = cache_write_tokens = 0
    output_tokens = total_tokens = tool_calls = requests = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, Mapping):
            input_tokens += _usage_int(usage, "input_tokens", "inputTokens")
            input_details = usage.get("input_token_details")
            if isinstance(input_details, Mapping):
                cache_read_tokens += _usage_int(
                    input_details,
                    "cache_read",
                    "cached_tokens",
                )
                cache_write_tokens += _usage_int(
                    input_details,
                    "cache_creation",
                    "cache_write_tokens",
                )
            output_tokens += _usage_int(usage, "output_tokens", "outputTokens")
            total_tokens += _usage_int(usage, "total_tokens", "totalTokens")
            requests += 1
        calls = getattr(message, "tool_calls", None)
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
            tool_calls += len(calls)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return InvocationUsage(
        requests=max(requests, 1),
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _langchain_current_messages(
    messages: Sequence[Any],
    input_messages: object,
) -> Sequence[Any]:
    if not isinstance(input_messages, Sequence) or isinstance(input_messages, (str, bytes)):
        return messages
    if len(messages) < len(input_messages):
        return messages
    if all(
        _same_langchain_message(actual, expected)
        for actual, expected in zip(messages, input_messages, strict=False)
    ):
        return messages[len(input_messages) :]
    return messages


def _same_langchain_message(actual: Any, expected: Any) -> bool:
    if actual is expected or actual == expected:
        return True
    return _langchain_message_identity(actual) == _langchain_message_identity(expected)


def _langchain_message_identity(message: Any) -> tuple[object, object]:
    if isinstance(message, Mapping):
        role = message.get("role", message.get("type"))
        content = message.get("content")
    else:
        role = getattr(message, "type", getattr(message, "role", None))
        content = getattr(message, "content", None)
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    return role, content


def _usage_int(values: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = values.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _apply_langchain_effort_settings(
    settings: dict[str, Any],
    model: str | Any,
    effort: str | None,
) -> None:
    if effort is None:
        return
    mapped = pydantic_effort_value(effort)
    existing = settings.get("reasoning", getattr(model, "reasoning", None))
    if (isinstance(model, str) and model.startswith("openrouter:")) or isinstance(
        existing, Mapping
    ):
        reasoning = dict(existing) if isinstance(existing, Mapping) else {}
        reasoning["effort"] = mapped
        settings["reasoning"] = reasoning
        return
    settings["reasoning_effort"] = mapped


def _langchain_structured_response(result: Any) -> Any:
    if isinstance(result, Mapping):
        if "structured_response" in result:
            return result["structured_response"]
        raise ValueError("LangChain agent did not return structured_response.")
    try:
        return result.structured_response
    except AttributeError:
        raise ValueError("LangChain agent did not return structured_response.") from None


def _capture_langchain_typesafe_decision_evidence(
    result: Any,
    *,
    input_messages: object = None,
) -> None:
    if not isinstance(result, Mapping):
        return
    messages = result.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return
    messages = _langchain_current_messages(messages, input_messages)
    for message in reversed(messages):
        metadata = getattr(message, "response_metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        typesafe = metadata.get("typesafe")
        if not isinstance(typesafe, Mapping):
            continue
        model = metadata.get("model_name")
        capture_decision_evidence(
            provider="typesafe",
            model=model if isinstance(model, str) else None,
            metadata=typesafe,
        )
        return


def _langchain_text_response(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        if isinstance(result.get("content"), str):
            return result["content"]
        if isinstance(result.get("output"), str):
            return result["output"]
        messages = result.get("messages")
        if isinstance(messages, (list, tuple)) and messages:
            return _langchain_message_text(messages[-1])
    return _langchain_message_text(result)


def _langchain_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    text = _langchain_content_text(content)
    if text is not None:
        return text
    raise ValueError("LangChain agent did not return a text response.")


def _langchain_content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
