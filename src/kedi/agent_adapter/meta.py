from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from typing_extensions import runtime_checkable

from ..resolved_types import ResolvedKediType

T = TypeVar("T", covariant=True)
AgentAdapterKind = Literal["agent-framework", "agent-harness"]
HookBridge = Literal["native", "kedi"]
HookToolOrigin = Literal["kedi", "mcp", "adapter_native", "provider_builtin"]
HookCapabilityEvent = Literal[
    "user_prompt_submit",
    "pre_tool_use",
    "post_tool_use",
    "post_tool_use_failure",
]
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
    "hooks",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class HookEventCapability:
    bridge: HookBridge
    origins: frozenset[HookToolOrigin] = frozenset()
    supports_mutation: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class HookCapabilities:
    user_prompt_submit: HookEventCapability | None = None
    pre_tool_use: HookEventCapability | None = None
    post_tool_use: HookEventCapability | None = None
    post_tool_use_failure: HookEventCapability | None = None

    @classmethod
    def full(
        cls,
        *,
        tool_bridge: HookBridge,
        tool_origins: frozenset[HookToolOrigin],
    ) -> HookCapabilities:
        return cls(
            user_prompt_submit=HookEventCapability(
                bridge="kedi",
                supports_mutation=True,
            ),
            pre_tool_use=HookEventCapability(
                bridge=tool_bridge,
                origins=tool_origins,
                supports_mutation=True,
            ),
            post_tool_use=HookEventCapability(
                bridge=tool_bridge,
                origins=tool_origins,
            ),
            post_tool_use_failure=HookEventCapability(
                bridge=tool_bridge,
                origins=tool_origins,
            ),
        )

    @classmethod
    def prompt_only(cls) -> HookCapabilities:
        return cls(
            user_prompt_submit=HookEventCapability(
                bridge="kedi",
                supports_mutation=True,
            )
        )

    def supports(self, event: HookCapabilityEvent) -> bool:
        return getattr(self, event) is not None

    def unsupported(
        self,
        events: frozenset[HookCapabilityEvent] | tuple[HookCapabilityEvent, ...],
    ) -> tuple[HookCapabilityEvent, ...]:
        return tuple(event for event in events if not self.supports(event))


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
    hooks: HookCapabilities | None = None
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
            "hooks": self.hooks is not None,
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
