"""LangChain native message bridge for exact artifact-backed history archival."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from .archive import (
    PROVIDER_INPUT_BASELINE_KEY,
    ArtifactHistoryProcessor,
    HistoryArchiveSettings,
    ProviderInputPressure,
)
from .langchain_codec import LangChainHistoryCodec

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class LangChainArtifactHistoryProcessor(ArtifactHistoryProcessor["BaseMessage"]):
    def __init__(self, settings: HistoryArchiveSettings | None = None) -> None:
        super().__init__(LangChainHistoryCodec(), settings)

    def serialize(self, messages: list[BaseMessage]) -> object:
        from langchain_core.messages import messages_to_dict

        return messages_to_dict(messages)

    def provider_input_pressure(self, messages: list[BaseMessage]) -> ProviderInputPressure | None:
        from langchain_core.messages import AIMessage

        baseline: int | None = None
        input_tokens: list[int] = []
        for message in messages:
            stored = message.additional_kwargs.get(PROVIDER_INPUT_BASELINE_KEY)
            if isinstance(stored, int) and not isinstance(stored, bool) and stored > 0:
                baseline = stored
            if isinstance(message, AIMessage) and message.usage_metadata:
                current = message.usage_metadata.get("input_tokens")
                if isinstance(current, int) and not isinstance(current, bool) and current > 0:
                    input_tokens.append(current)
        if not input_tokens:
            return None
        if baseline is None:
            baseline = input_tokens[0]
        return ProviderInputPressure(
            baseline_tokens=baseline,
            current_tokens=input_tokens[-1],
            growth_tokens=max(0, input_tokens[-1] - baseline),
        )

    def with_provider_input_baseline(
        self, message: BaseMessage, baseline_tokens: int
    ) -> BaseMessage:
        return message.model_copy(
            update={
                "additional_kwargs": {
                    **message.additional_kwargs,
                    PROVIDER_INPUT_BASELINE_KEY: baseline_tokens,
                }
            }
        )

    def tool_outcomes(self, messages: list[BaseMessage]) -> tuple[str, ...]:
        from langchain_core.messages import AIMessage, ToolMessage

        calls: Counter[str] = Counter()
        results: Counter[tuple[str, str]] = Counter()
        names: dict[str, str] = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    calls[call["name"]] += 1
                    identifier = call.get("id")
                    if identifier:
                        names[identifier] = call["name"]
            elif isinstance(message, ToolMessage):
                name = message.name or names.get(message.tool_call_id, "unknown")
                status = "success" if message.status == "success" else "failed"
                results[(name, status)] += 1
        return tuple(
            f"{name}: calls={calls[name]}; "
            + ", ".join(
                f"{status}={count}"
                for (tool, status), count in sorted(results.items())
                if tool == name
            )
            for name in sorted(calls)
        )
