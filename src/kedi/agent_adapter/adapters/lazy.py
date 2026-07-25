from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Sequence

from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind
from kedi.agent_adapter.subagents import SupportsSubagentExecution

if TYPE_CHECKING:
    from ...agent_profile import AgentProfile
    from ..subagents import SubagentExecutionRequest
    from ..subagents.limits import SubagentUsageLimits
    from ..tool_override import ToolSpec


class LazyAdapter:
    """Builds the real adapter on first LLM use so the program runs exactly once.

    Programs fully covered by generated/native code never construct an adapter (no
    API key needed); a template that actually invokes the model builds it lazily, so
    top-level side effects fire once instead of twice. Forwards to the built adapter,
    structurally satisfying the adapter protocols exposed by the selected adapter.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        kind: AgentAdapterKind = "agent-framework",
        shortname: str = "lazy",
        capabilities: AdapterCapabilities | None = None,
    ) -> None:
        self._factory = factory
        self.kind = kind
        self.shortname = shortname
        self.capabilities = capabilities or AdapterCapabilities()
        self._real: Any = None
        self._lock = Lock()

    def _ensure(self) -> Any:
        if self._real is None:
            with self._lock:
                if self._real is None:
                    self._real = self._factory()
        return self._real

    def default_model(self) -> str | None:
        return self._ensure().default_model()

    def telemetry_model_name(self) -> str | None:
        return self._subagent_adapter().telemetry_model_name()

    def produce_sync(self, *args: Any, **kwargs: Any) -> Any:
        return self._ensure().produce_sync(*args, **kwargs)

    async def produce(self, *args: Any, **kwargs: Any) -> Any:
        return await self._ensure().produce(*args, **kwargs)

    def invoke_sync(self, *args: Any, **kwargs: Any) -> str:
        return self._ensure().invoke_sync(*args, **kwargs)

    async def invoke(self, *args: Any, **kwargs: Any) -> str:
        return await self._ensure().invoke(*args, **kwargs)

    def profile_override(self, profile: AgentProfile) -> AbstractContextManager[None]:
        return self._ensure().profile_override(profile)

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._ensure().tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._ensure().register_tool(tool)

    def subagent_scope(
        self,
        request: SubagentExecutionRequest,
    ) -> AbstractContextManager[None]:
        return self._subagent_adapter().subagent_scope(request)

    def subagent_run_kwargs(
        self,
        limits: SubagentUsageLimits,
    ) -> Mapping[str, Any]:
        return self._subagent_adapter().subagent_run_kwargs(limits)

    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        return await self._subagent_adapter().produce_subagent(
            template=template,
            output_schema=output_schema,
            **kwargs,
        )

    def _subagent_adapter(self) -> SupportsSubagentExecution:
        adapter = self._ensure()
        if not isinstance(adapter, SupportsSubagentExecution):
            raise RuntimeError(
                f"Adapter {self.shortname!r} does not implement Kedi child execution"
            )
        return adapter

    @property
    def supports_native_subagent_resume(self) -> bool:
        return bool(getattr(self._ensure(), "supports_native_subagent_resume", False))

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._ensure().scoped_tools
