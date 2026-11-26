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


def thaw_json(value: Any) -> Any:
    """A frozen request mapping or tuple as plain JSON values, for a backend request."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
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

    @property
    def parameters(self) -> dict[str, Any]:
        """The input schema as a plain JSON object, ``{"type": "object"}`` without one."""

        return thaw_json(self.input_schema) if self.input_schema is not None else {"type": "object"}

    @property
    def returns(self) -> dict[str, Any] | None:
        """The return schema as a plain JSON object, when the tool declares one."""

        return None if self.output_schema is None else thaw_json(self.output_schema)


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
class RenderedRequest:
    """Everything a model sees for one request, as Kedi rendered it.

    An adapter maps it onto its backend's messages: ``system`` (or ``instruction_parts``)
    as system or developer text, ``user`` and ``media`` as the user turn, the output schema
    as the backend's structured-output or output-tool contract, and ``tools`` as its tool
    definitions. When the backend lacks native structured output or tool calling, the
    ``protocol.output`` and ``protocol.tools`` parts carry them instead and are already in
    ``system``.
    """

    system: str | None
    instruction_parts: tuple[InstructionPart, ...]
    user: str
    media: tuple[MediaAttachment, ...]
    output_json_schema: Mapping[str, Any] | None
    output_tool_name: str | None
    tools: tuple[ToolDefinition, ...]
    prompt_kind: PromptKind
    call_kind: CallKind


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunRequest:
    """Resolved semantic request handed to adapter-native lowering.

    ``protocol`` names the protocol parts the adapter's backend needs (``output``, ``tools``);
    :meth:`render` adds them. They stay out of ``instructions``, so hooks, telemetry and the
    built-in adapters see the same instructions as before.
    """

    prompt: PromptDraft
    instructions: tuple[InstructionPart, ...] = ()
    conversation: ConversationCoordinates | None = None
    tool_definitions: tuple[ToolDefinition, ...] = ()
    output: OutputContract = field(default_factory=TextOutput)
    metadata: RunMetadata
    protocol: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "tool_definitions", tuple(self.tool_definitions))
        object.__setattr__(self, "protocol", frozenset(self.protocol))

    @property
    def instruction_text(self) -> str | None:
        rendered = "\n\n".join(part.content for part in self.instructions if part.content)
        return rendered or None

    def render(self, *, tools: tuple[ToolDefinition, ...] | None = None) -> RenderedRequest:
        """The request as the model sees it.

        ``tools`` replaces the request's tool definitions with the adapter's final set, for
        an adapter that adds tools during the run (MCP servers, for example), so the tool
        catalog names them too.
        """

        from .protocol import OUTPUT_TOOL_NAME, protocol_parts

        definitions = self.tool_definitions if tools is None else tuple(tools)
        schema = output_json_schema(self.output, request=self)
        parts = (
            *self.instructions,
            *protocol_parts(
                self.protocol,
                output_json_schema=schema,
                tools=definitions,
            ),
        )
        system = "\n\n".join(part.content for part in parts if part.content)
        return RenderedRequest(
            system=system or None,
            instruction_parts=parts,
            user=self.prompt.content,
            media=self.prompt.media,
            output_json_schema=schema,
            output_tool_name=None if schema is None else OUTPUT_TOOL_NAME,
            tools=definitions,
            prompt_kind=self.metadata.prompt_kind,
            call_kind=self.metadata.call_kind,
        )


def output_json_schema(
    output: OutputContract, *, request: AgentRunRequest | None = None
) -> Mapping[str, Any] | None:
    """The JSON Schema Kedi builds for an output contract, or ``None`` for text.

    A template's fields go through ``build_output_model``, so the model docstring and field
    descriptions are included.
    """

    if isinstance(output, TextOutput):
        return None
    from pydantic import BaseModel, TypeAdapter

    from ..output_schema import build_output_model

    if isinstance(output, SchemaOutput):
        return build_output_model(dict(output.schema), request=request).model_json_schema()
    output_type = output.output_type
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return output_type.model_json_schema()
    return TypeAdapter(output_type).json_schema()


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
    "RenderedRequest",
    "RunMetadata",
    "SchemaOutput",
    "TextOutput",
    "ToolDefinition",
    "TypedOutput",
    "output_json_schema",
]
