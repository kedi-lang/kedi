"""Session-owned binding for embedding one Kedi-configured agent."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from threading import Lock
from typing import Any, Generator, TYPE_CHECKING

from typing_extensions import final

from .agent_adapter.conversation import ConversationState, SupportsConversationState
from .agent_adapter.meta import AgentAdapter, SupportsProfileOverride
from .agent_adapter.tools.specs import SupportsToolRegistration, ToolSpec
from .agent_profile import (
    AgentProfile,
    AgentProfileOverrideUnsupported,
    AgentToolOverrideUnsupported,
)

if TYPE_CHECKING:
    from .artifacts import ArtifactManager

__all__ = ("AgentSurface",)


@final
class AgentSurface:
    """Bind Kedi runtime services to an adapter for complete native agent runs.

    The caller owns the adapter and runtime. Conversation state is borrowed unless
    ``close_conversation_state`` is enabled explicitly.
    """

    def __init__(
        self,
        *,
        agent: AgentAdapter[Any],
        profile: AgentProfile,
        tool_specs: tuple[ToolSpec, ...],
        conversation_state: ConversationState | None,
        artifact_manager: ArtifactManager | None = None,
        close_conversation_state: bool = False,
    ) -> None:
        if not isinstance(agent, SupportsProfileOverride):
            raise AgentProfileOverrideUnsupported(
                "Bound agent surfaces require adapter profile override support"
            )
        if tool_specs and not isinstance(agent, SupportsToolRegistration):
            raise AgentToolOverrideUnsupported(
                "Bound agent surface tools require adapter tool registration support"
            )
        if conversation_state is not None and not (
            agent.capabilities.supports("stateful_history")
            and isinstance(agent, SupportsConversationState)
        ):
            raise AgentProfileOverrideUnsupported(
                f"Adapter {agent.shortname!r} does not support bound conversation state"
            )

        self.agent = agent
        self.profile = profile
        self.tool_specs = tool_specs
        self.conversation_state = conversation_state
        self._artifact_manager = (
            artifact_manager
            if artifact_manager is not None
            else conversation_state.artifacts
            if conversation_state is not None
            else None
        )
        self._close_conversation_state = close_conversation_state
        self._profile_adapter = agent
        self._tool_adapter = agent if isinstance(agent, SupportsToolRegistration) else None
        self._conversation_adapter = agent if isinstance(agent, SupportsConversationState) else None
        self._run_lock = Lock()
        self._lifecycle_lock = Lock()
        self._closed = False

    @contextmanager
    def run_scope(self) -> Generator[None, None, None]:
        """Keep all Kedi bindings active for one complete run or stream lifetime."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("AgentSurface is closed")
            if not self._run_lock.acquire(blocking=False):
                raise RuntimeError("AgentSurface is already running")

        try:
            with ExitStack() as stack:
                stack.enter_context(self._profile_adapter.profile_override(self.profile))
                if self._tool_adapter is not None:
                    stack.enter_context(self._tool_adapter.tool_scope(self.tool_specs))
                if self._artifact_manager is not None and self.profile.artifacts is not None:
                    from .agent_adapter.tools.scope import artifact_admission_scope

                    stack.enter_context(
                        artifact_admission_scope(
                            manager=self._artifact_manager,
                            policy=self.profile.artifacts,
                        )
                    )
                if self.conversation_state is not None:
                    from .agent_adapter.request import conversation_lane_identity

                    conversation = stack.enter_context(
                        self.conversation_state.adapter_turn(
                            self.agent.shortname,
                            lane_identity=conversation_lane_identity(
                                self.agent,
                                self.profile,
                                self.tool_specs,
                            ),
                        )
                    )
                    conversation.require_explicit_completion()
                    if self._conversation_adapter is None:  # pragma: no cover - constructor guard
                        raise RuntimeError("Bound conversation adapter is unavailable")
                    stack.enter_context(self._conversation_adapter.conversation_scope(conversation))
                yield
        finally:
            self._run_lock.release()

    def close(self) -> None:
        """Close resources explicitly assigned to this surface."""

        with self._lifecycle_lock:
            if self._closed:
                return
            if self._run_lock.locked():
                raise RuntimeError("Cannot close AgentSurface while it is running")
            self._closed = True
        if self._close_conversation_state and self.conversation_state is not None:
            self.conversation_state.close()

    async def aclose(self) -> None:
        """Asynchronously close resources explicitly assigned to this surface."""

        self.close()

    def __enter__(self) -> AgentSurface:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("AgentSurface is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> AgentSurface:
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
