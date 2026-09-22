from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from kedi.agent_adapter.codemode.tool_specs import append_codemode_instructions
from kedi.agent_adapter.events import AgentMessageResolver
from kedi.agent_adapter.hooks import current_hook_dispatcher, handler_identity
from kedi.agent_adapter.invocation import InvocationUsage
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import CodeModeSettings


def _claude_session_fingerprint(
    options: Mapping[str, Any],
    *,
    tools: Sequence[ToolSpec],
    codemode_settings: CodeModeSettings | None = None,
    hook_identities: Mapping[str, Sequence[str]] | None = None,
) -> str:
    payload = {
        "options": dict(options),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "json_schema": tool.json_schema,
                "return_schema": tool.return_schema,
                "metadata": tool.metadata,
                "risk": tool.risk,
                "artifact_result": tool.artifact_result,
                "sequential": tool.sequential,
            }
            for tool in tools
        ],
        "codemode": codemode_settings,
        "kedi_hooks": hook_identities or {},
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_claude_fingerprint_default,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _active_claude_hook_identities() -> dict[str, tuple[str, ...]]:
    dispatcher = current_hook_dispatcher()
    if dispatcher is None:
        return {}
    return {
        event: tuple(handler_identity(handler) for handler in handlers)
        for event, handlers in dispatcher.snapshot.handlers.items()
    }


def _claude_fingerprint_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', '')}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _set_claude_system_prompt(
    options: dict[str, Any],
    instructions: str | None,
    *,
    codemode_enabled: bool,
    codemode_preloaded_tools: Sequence[Mapping[str, Any]] = (),
) -> None:
    existing = options.get("system_prompt")
    if existing is None:
        prompt = dict(CLAUDE_CODE_SYSTEM_PROMPT_PRESET)
        if instructions is not None:
            prompt["append"] = instructions
        options["system_prompt"] = prompt
        return
    if not codemode_enabled:
        return
    if isinstance(existing, str):
        options["system_prompt"] = append_codemode_instructions(
            existing,
            enabled=True,
            preloaded_tools=codemode_preloaded_tools,
        )
        return
    if isinstance(existing, Mapping):
        prompt = dict(existing)
        append = prompt.get("append")
        if append is not None and not isinstance(append, str):
            raise TypeError("Claude system_prompt append must be a string")
        prompt["append"] = append_codemode_instructions(
            append,
            enabled=True,
            preloaded_tools=codemode_preloaded_tools,
        )
        options["system_prompt"] = prompt
        return
    raise TypeError("Claude system_prompt must be a string or preset mapping")


def _output_json_schema(output_type: type[Any]) -> dict[str, Any]:
    if issubclass(output_type, BaseModel):
        return output_type.model_json_schema()
    return {"type": _json_schema_type_name(output_type)}


def _json_schema_type_name(output_type: type[Any]) -> str:
    if output_type is str:
        return "string"
    if output_type is int:
        return "integer"
    if output_type is float:
        return "number"
    if output_type is bool:
        return "boolean"
    return "object"


def _structured_output_value(value: Any, output_type: type[Any]) -> Any:
    if issubclass(output_type, BaseModel):
        if isinstance(value, output_type):
            return value
        return output_type.model_validate(value)
    return value


def _claude_text_response(message: Any) -> str:
    result = getattr(message, "result", None)
    if isinstance(result, str):
        return result
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(result, Mapping):
        text = result.get("text")
        if isinstance(text, str):
            return text
    raise ValueError("Claude Agent SDK did not return a text response.")


def _observe_claude_message(
    sdk: Any,
    message: Any,
    resolver: AgentMessageResolver,
    message_index: int,
) -> int:
    assistant_cls = getattr(sdk, "AssistantMessage", None)
    if assistant_cls is not None and isinstance(message, assistant_cls):
        text = _claude_assistant_text(sdk, message)
        if text:
            message_index += 1
            resolver.complete_message(f"claude:{message_index}", text)
            resolver.mark_message_boundary()
        if _claude_has_tool_use(sdk, message):
            resolver.mark_tool_boundary()
        return message_index

    user_cls = getattr(sdk, "UserMessage", None)
    if user_cls is not None and isinstance(message, user_cls):
        resolver.mark_tool_boundary()
    return message_index


def _claude_assistant_text(sdk: Any, message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return ""
    text_cls = getattr(sdk, "TextBlock", None)
    if text_cls is None:
        return ""
    parts = [block.text for block in content if isinstance(block, text_cls) and block.text]
    return "".join(parts).strip()


def _claude_has_tool_use(sdk: Any, message: Any) -> bool:
    content = message.content
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return False
    tool_use_cls = getattr(sdk, "ToolUseBlock", None)
    return tool_use_cls is not None and any(isinstance(block, tool_use_cls) for block in content)


def _is_result_message(sdk: Any, message: Any) -> bool:
    result_cls = getattr(sdk, "ResultMessage", None)
    if result_cls is not None and isinstance(message, result_cls):
        return True
    return hasattr(message, "structured_output") or hasattr(message, "is_error")


def _claude_non_retryable_system_error(sdk: Any, message: Any) -> str | None:
    system_message = getattr(sdk, "SystemMessage", None)
    if (
        system_message is None
        or not isinstance(message, system_message)
        or message.subtype != "api_retry"
    ):
        return None
    status = message.data.get("error_status")
    if status not in {400, 401, 403, 404, 422}:
        return None
    error = message.data.get("error")
    detail = f": {error}" if isinstance(error, str) and error else ""
    if status in {401, 403}:
        return (
            f"Claude authentication failed (HTTP {status}){detail}; "
            "check ANTHROPIC_API_KEY and Claude endpoint settings"
        )
    return f"Claude request failed with non-retryable HTTP {status}{detail}"


def _claude_invocation_usage(result: Any) -> InvocationUsage:
    usage = getattr(result, "usage", None)
    values = usage if isinstance(usage, Mapping) else {}
    input_tokens = _usage_int(values, "input_tokens", "inputTokens")
    output_tokens = _usage_int(values, "output_tokens", "outputTokens")
    total_tokens = _usage_int(values, "total_tokens", "totalTokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    turns = getattr(result, "num_turns", 0)
    return InvocationUsage(
        requests=turns if isinstance(turns, int) else 0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _usage_int(values: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = values.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


CLAUDE_CODE_SYSTEM_PROMPT_PRESET: Mapping[str, str] = {
    "type": "preset",
    "preset": "claude_code",
}
