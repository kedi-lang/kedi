from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, TypeVar

from typing_extensions import runtime_checkable

from kedi.agent_adapter.hooks import HookEventName, HookHandler, HookRegistrationMixin
from kedi.agent_adapter.meta import AdapterCapabilities, AgentAdapterKind
from kedi.agent_adapter.subagents import SupportsSubagentExecution
from kedi.telemetry import add_counter, record_histogram, span

if TYPE_CHECKING:
    from ...agent_profile import AgentProfile
    from ..conversation import AdapterConversationContext
    from ..subagents import SubagentExecutionRequest
    from ..subagents.limits import SubagentUsageLimits
    from ..tools.specs import ToolSpec


@runtime_checkable
class _SupportsDefaultModel(Protocol):
    def default_model(self) -> str | None: ...


@runtime_checkable
class _SupportsTelemetryModelName(Protocol):
    def telemetry_model_name(self) -> str | None: ...


class LazyAdapter(HookRegistrationMixin):
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
        model: str | None = None,
        capabilities: AdapterCapabilities | None = None,
        hook_handler: HookHandler | None = None,
    ) -> None:
        self._init_hook_registration(hook_handler)
        self._factory = factory
        self.kind = kind
        self.shortname = shortname
        self._configured_model = model
        self.capabilities = capabilities or AdapterCapabilities()
        self._real: Any = None
        self._lock = Lock()

    def _ensure(self) -> Any:
        if self._real is None:
            with self._lock:
                if self._real is None:
                    started_at = perf_counter()
                    outcome = "success"
                    kind = "harness" if self.kind == "agent-harness" else "framework"
                    with span(
                        "agent",
                        f"initialize adapter {self.shortname}",
                        "initialize_adapter",
                        attributes={
                            "kedi.adapter.name": self.shortname,
                            "kedi.adapter.kind": kind,
                        },
                    ) as active:
                        try:
                            self._real = self._factory()
                            if isinstance(self._real, HookRegistrationMixin):
                                self._real._prepend_hook_snapshot(self.hook_snapshot())
                        except BaseException as exc:
                            outcome = "error"
                            active.set_attribute("error.type", type(exc).__name__)
                            raise
                        else:
                            active.add_event("adapter initialized")
                        finally:
                            dimensions = {
                                "adapter": self.shortname,
                                "adapter_kind": kind,
                                "outcome": outcome,
                            }
                            add_counter(
                                "agent",
                                "kedi.adapter.initializations",
                                attributes=dimensions,
                            )
                            record_histogram(
                                "agent",
                                "kedi.adapter.initialization.duration",
                                max(0.0, perf_counter() - started_at),
                                unit="s",
                                attributes=dimensions,
                            )
        return self._real

    def resolve(self) -> Any:
        """Return the concrete adapter for request identity and native lowering."""

        return self._ensure()

    def on(
        self,
        events: HookEventName | str | Sequence[HookEventName | str],
    ) -> Callable[[_HandlerT], _HandlerT]:
        register_outer = super().on(events)

        def register(handler: _HandlerT) -> _HandlerT:
            with self._lock:
                register_outer(handler)
                if isinstance(self._real, HookRegistrationMixin):
                    self._real.on(events)(handler)
            return handler

        return register

    @property
    def is_initialized(self) -> bool:
        return self._real is not None

    def default_model(self) -> str | None:
        adapter = self._ensure()
        if isinstance(adapter, _SupportsDefaultModel):
            return adapter.default_model()
        return None

    @property
    def configured_model(self) -> str | None:
        """Return a portable model ID without initializing an unused backend."""
        if self._configured_model is not None:
            return self._configured_model
        if isinstance(self._real, _SupportsDefaultModel):
            return self._real.default_model()
        return None

    def telemetry_model_name(self) -> str | None:
        adapter = self._ensure()
        if isinstance(adapter, _SupportsTelemetryModelName):
            return adapter.telemetry_model_name()
        if isinstance(adapter, _SupportsDefaultModel):
            return adapter.default_model()
        return None

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

    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> AbstractContextManager[None]:
        return self._ensure().conversation_scope(conversation)

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


_HandlerT = TypeVar("_HandlerT", bound=Callable[..., Any])
