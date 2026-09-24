from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kedi.agent_adapter.history_processing import (
        HistoryProcessorCallback,
        HistoryProcessorCondition,
    )
    from kedi.core import KediRuntime


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryProcessingSettings:
    """A lexical override; a present policy with processor=None disables processing."""

    processor: HistoryProcessorCallback[Any] | None = None
    condition: HistoryProcessorCondition | None = None
    inherit_processor: bool = False

    def __post_init__(self) -> None:
        if self.processor is not None and not callable(self.processor):
            raise TypeError("history processor must be callable or None")
        if self.condition is not None and not callable(self.condition):
            raise TypeError("history processor condition must be callable or None")
        if self.processor is None and self.condition is not None and not self.inherit_processor:
            raise ValueError("history processor condition requires a processor")

    def merge(self, other: HistoryProcessingSettings) -> HistoryProcessingSettings:
        if not other.inherit_processor or self.inherit_processor:
            return other
        return HistoryProcessingSettings(processor=self.processor, condition=other.condition)

    def resolve(
        self,
        default_processor: HistoryProcessorCallback[Any] | None,
    ) -> tuple[HistoryProcessorCallback[Any] | None, HistoryProcessorCondition | None]:
        processor = default_processor if self.inherit_processor else self.processor
        if processor is None and self.condition is not None:
            raise ValueError("history processor condition requires a processor")
        return processor, self.condition


def bind_history_processor(rt: KediRuntime, callback: Any) -> Any:
    """Freeze a native callback's procedure target at directive evaluation."""
    target = getattr(callback, "__kedi_procedure_target__", None)
    if target is None:
        return callback

    def invoke(context: Any) -> Any:
        return rt._invoke_procedure(target, [context])

    return invoke
