from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from ..agent_profile import AgentProfile
    from .tool_override import ToolSpec


class LazyAdapter:
    """Adapter wrapper that builds the real adapter only on first LLM use.

    ``kedi <program>`` used to run the program twice: once offline (``adapter=None``)
    and, if a template actually needed the LLM, again with a real adapter — which
    re-executed every top-level side effect (BUG-007). Passing a ``LazyAdapter`` to
    ``compile_program`` lets the program run exactly once: it is only constructed when
    a template genuinely invokes the model (``KediRuntime._kedi_run`` is reached only
    when there are required outputs), so programs fully covered by generated/native
    code never build an adapter (no API key required) while side effects fire once.

    It structurally satisfies ``SupportsProfileOverride`` and
    ``SupportsToolRegistration`` by forwarding to the lazily built adapter.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._real: Any | None = None

    def _ensure(self) -> Any:
        if self._real is None:
            self._real = self._factory()
        return self._real

    # ``default_model`` is only consulted by ``_kedi_run`` when a template is about to
    # run, so building here is still "first real use".
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
