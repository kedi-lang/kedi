from __future__ import annotations

from contextlib import AbstractContextManager
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Sequence

from kedi.agent_adapter.adapters import DSPyAdapter, LangChainAdapter, PydanticAdapter

if TYPE_CHECKING:
    from ..agent_profile import AgentProfile
    from .tool_override import ToolSpec

AvailableAdapter = DSPyAdapter | LangChainAdapter | PydanticAdapter


class LazyAdapter:
    """Builds the real adapter on first LLM use so the program runs exactly once.

    Programs fully covered by generated/native code never construct an adapter (no
    API key needed); a template that actually invokes the model builds it lazily, so
    top-level side effects fire once instead of twice. Forwards to the built adapter,
    structurally satisfying SupportsProfileOverride / SupportsToolRegistration.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._real: Any = None
        self._lock = Lock()

    def _ensure(self) -> AvailableAdapter:
        if self._real is None:
            with self._lock:
                if self._real is None:
                    self._real = self._factory()
        return self._real

    def default_model(self) -> str | None:
        return self._ensure().default_model()

    def produce_sync(self, *args: Any, **kwargs: Any) -> Any:
        return self._ensure().produce_sync(*args, **kwargs)

    async def produce(self, *args: Any, **kwargs: Any) -> Any:
        return await self._ensure().produce(*args, **kwargs)

    def profile_override(self, profile: AgentProfile) -> AbstractContextManager[None]:
        return self._ensure().profile_override(profile)

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        return self._ensure().tool_scope(tools)

    def register_tool(self, tool: ToolSpec) -> None:
        self._ensure().register_tool(tool)

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._ensure().scoped_tools
