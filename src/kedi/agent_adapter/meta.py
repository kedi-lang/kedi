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
    supported_json_schema_formats: frozenset[str] | None = None
    settings: frozenset[str] = field(default_factory=frozenset)

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
        output_schema: dict[str, ResolvedKediType],
        **kwargs: Any,
    ) -> T:
        """
        Adapter protocol for agents to produce outputs based on a given template and output type.
        Args:
            template (str): The input template or prompt for the agent.
            output_schema (dict): A dictionary defining the output schema with field names as keys and their types as values.
            **kwargs: Additional keyword arguments for the agent.
        Returns:
            T: The output produced by the agent, matching the specified output type.
        """
        ...

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **kwargs: Any,
    ) -> T:
        """
        Synchronous version of the produce method.
        Args:
            template (str): The input template or prompt for the agent.
            output_schema (dict): A dictionary defining the output schema with field names as keys and their types as values.
            **kwargs: Additional keyword arguments for the agent.
        Returns:
            T: The output produced by the agent, matching the specified output type.
        """
        ...

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        """Invoke the adapter with a raw text prompt and return the model's text response."""
        ...

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        """Synchronous version of ``invoke``."""
        ...
