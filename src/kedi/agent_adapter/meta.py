from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from typing_extensions import runtime_checkable

from ..resolved_types import ResolvedKediType

T = TypeVar("T", covariant=True)
AgentAdapterKind = Literal["agent-framework", "agent-harness"]
AdapterCapability = Literal[
    "structured_output",
    "tool_registration",
    "mcp",
    "profile_override",
    "model_override",
    "effort",
    "settings",
    "codemode",
    "native_approvals",
    "native_approval_handler",
    "subagents",
    "background_subagents",
    "artifacts",
    "native_artifacts",
    "stateful_history",
    "history_replay",
    "native_compaction",
    "stream_events",
]


@dataclass(frozen=True)
class AdapterCapabilities:
    supports_structured_output: bool | None = None
    supports_tool_registration: bool | None = None
    supports_mcp: bool | None = None
    supports_profile_override: bool | None = None
    supports_model_override: bool | None = None
    supports_effort: bool | None = None
    supports_settings: bool | None = None
    supports_codemode: bool | None = None
    supports_native_approvals: bool | None = None
    supports_native_approval_handler: bool | None = None
    supports_subagents: bool | None = None
    supports_background_subagents: bool | None = None
    supports_artifacts: bool | None = None
    supports_native_artifacts: bool | None = None
    supports_stateful_history: bool | None = None
    supports_history_replay: bool | None = None
    supports_native_compaction: bool | None = None
    supports_stream_events: bool | None = None
    supported_json_schema_formats: frozenset[str] | None = None
    settings: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.supports_artifacts and not (
            self.supports_structured_output and self.supports_tool_registration
        ):
            raise ValueError(
                "Artifact-capable adapters must support structured output and tool registration"
            )
        if self.supports_native_artifacts and not self.supports_artifacts:
            raise ValueError("Native artifact support requires artifact support")

    def supports(self, feature: AdapterCapability) -> bool:
        values = {
            "structured_output": self.supports_structured_output,
            "tool_registration": self.supports_tool_registration,
            "mcp": self.supports_mcp,
            "profile_override": self.supports_profile_override,
            "model_override": self.supports_model_override,
            "effort": self.supports_effort,
            "settings": self.supports_settings,
            "codemode": self.supports_codemode,
            "native_approvals": self.supports_native_approvals,
            "native_approval_handler": self.supports_native_approval_handler,
            "subagents": self.supports_subagents,
            "background_subagents": self.supports_background_subagents,
            "artifacts": self.supports_artifacts,
            "native_artifacts": self.supports_native_artifacts,
            "stateful_history": self.supports_stateful_history,
            "history_replay": self.supports_history_replay,
            "native_compaction": self.supports_native_compaction,
            "stream_events": self.supports_stream_events,
        }
        return bool(values[feature])


@runtime_checkable
class AgentAdapter(Protocol[T]):
    kind: AgentAdapterKind
    shortname: str
    capabilities: AdapterCapabilities

    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> T:
        """Produce structured output from a field schema or prebuilt output type."""
        ...

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> T:
        """Synchronous structured output from a field schema or prebuilt type."""
        ...

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        """Invoke the adapter with a raw text prompt and return the model's text response."""
        ...

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        """Synchronous version of ``invoke``."""
        ...
