from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, TypeAlias

if TYPE_CHECKING:
    from .media import MediaAttachment

PromptKind: TypeAlias = Literal["raw", "template", "claim"]
CallKind: TypeAlias = Literal["invoke", "produce"]


def _frozen_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_frozen_value(item) for item in value), key=repr))
    return value


def _frozen_semantic_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _frozen_value(value)


def _mapping_snapshot(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptDraft:
    """One fully rendered user prompt before adapter-native lowering.

    ``content`` is the prompt text. ``media`` holds the attachments the text names; adapters
    place them, each with a ``name:`` label, before the text in the same user message.
    """

    content: str
    kind: PromptKind
    media: tuple[MediaAttachment, ...] = ()

    def with_content(self, content: str) -> PromptDraft:
        """Replace the text (for example after a hook edit); the attachments stay."""

        return replace(self, content=content)

    def with_media(self, media: tuple[MediaAttachment, ...]) -> PromptDraft:
        return replace(self, media=tuple(media))


@dataclass(frozen=True, slots=True, kw_only=True)
class InstructionPart:
    """A cache-stable instruction fragment with explicit provenance."""

    identity: str
    source: Literal["profile", "capability", "skills", "artifacts", "runtime"]
    content: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDefinition:
    """Immutable tool contract, including tools pending native activation."""

    name: str
    description: str | None
    input_schema: Mapping[str, Any] | None
    output_schema: Mapping[str, Any] | None
    risk: str
    required_before_output: bool
    artifact_result: str
    sequential: bool
    capability_owners: tuple[str, ...] = ()
    deferred_capability: str | None = None

    def __post_init__(self) -> None:
        if self.input_schema is not None:
            object.__setattr__(
                self,
                "input_schema",
                _frozen_semantic_mapping(self.input_schema),
            )
        if self.output_schema is not None:
            object.__setattr__(
                self,
                "output_schema",
                _frozen_semantic_mapping(self.output_schema),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TextOutput:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaOutput:
    schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _frozen_semantic_mapping(self.schema))


@dataclass(frozen=True, slots=True, kw_only=True)
class TypedOutput:
    output_type: Any


OutputContract: TypeAlias = TextOutput | SchemaOutput | TypedOutput


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationCoordinates:
    session_id: str
    lane_identity: str
    cache_epoch: int
    cache_key: str
    history_count: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class RunMetadata:
    run_id: str
    turn_sequence: int
    prompt_kind: PromptKind
    call_kind: CallKind


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunRequest:
    """Resolved semantic request handed to adapter-native lowering."""

    prompt: PromptDraft
    instructions: tuple[InstructionPart, ...] = ()
    conversation: ConversationCoordinates | None = None
    tool_definitions: tuple[ToolDefinition, ...] = ()
    output: OutputContract = field(default_factory=TextOutput)
    metadata: RunMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "tool_definitions", tuple(self.tool_definitions))

    @property
    def instruction_text(self) -> str | None:
        rendered = "\n\n".join(part.content for part in self.instructions if part.content)
        return rendered or None


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterCallOptions:
    """Per-call controls and live state intentionally excluded from AgentRunRequest."""

    conversation_context: Any | None = None
    native_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "native_kwargs", _mapping_snapshot(self.native_kwargs))


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterRunOutcome:
    """Completed adapter state captured before one atomic local conversation commit."""

    value: Any
    portable_result: Any
    continuation_state: Mapping[str, Any] = field(default_factory=dict)
    cleanup: Callable[[], None] | None = None
    cache_prefix_changed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "continuation_state",
            _mapping_snapshot(self.continuation_state),
        )


__all__ = [
    "AdapterCallOptions",
    "AdapterRunOutcome",
    "AgentRunRequest",
    "CallKind",
    "ConversationCoordinates",
    "InstructionPart",
    "OutputContract",
    "PromptDraft",
    "PromptKind",
    "RunMetadata",
    "SchemaOutput",
    "TextOutput",
    "ToolDefinition",
    "TypedOutput",
]
