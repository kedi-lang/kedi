"""Process-local incremental execution for Kedi source fragments."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from .agent_adapter.conversation import ConversationState
from .agent_profile import AgentProfile
from .lang.compiler.program_compiler import ProgramCompiler
from .lang.parser import parse_program

if TYPE_CHECKING:
    from .agent_adapter.meta import AgentAdapter
    from .agent_adapter.subagents.limits import SubagentUsageLimits
    from .executors.base import Executor


class InteractiveSession:
    """Execute complete Kedi fragments against one persistent process-local runtime."""

    def __init__(
        self,
        *,
        adapter: AgentAdapter[Any] | None = None,
        executor: Executor | None = None,
        runtime_globals: dict[str, Any] | None = None,
        engine: Any | None = None,
        default_agent_profile: AgentProfile | None = None,
        cwd: str | Path | None = None,
        subagent_max_depth: int = 5,
        subagent_max_concurrency: int = 4,
        subagent_timeout_seconds: float | None = 120.0,
        subagent_usage_limits: SubagentUsageLimits | None = None,
        subagent_state_path: str | Path | None = None,
        conversation_state: ConversationState | None = None,
        close_conversation_state: bool = False,
    ) -> None:
        self._cwd = Path.cwd() if cwd is None else Path(cwd)
        self._cwd = self._cwd.resolve()
        self._conversation_state = conversation_state
        self._close_conversation_state = close_conversation_state
        self._compiler = ProgramCompiler(
            adapter,
            executor=executor,
            runtime_globals=runtime_globals,
            engine=engine,
            default_agent_profile=default_agent_profile,
            subagent_max_depth=subagent_max_depth,
            subagent_max_concurrency=subagent_max_concurrency,
            subagent_timeout_seconds=subagent_timeout_seconds,
            subagent_usage_limits=subagent_usage_limits,
            subagent_state_path=subagent_state_path,
            conversation_state=conversation_state,
        )
        self._env = self._compiler.initialize_fragments()
        self._top_level_state = self._compiler.new_top_level_state()
        self._execute_lock = Lock()
        self._lifecycle_lock = Lock()
        self._closed = False
        self._cell_number = 0
        self._source_names: set[str] = set()

    def execute(self, source: str, *, source_name: str | None = None) -> object:
        """Parse and execute one complete fragment exactly once."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("InteractiveSession is closed")
            if not self._execute_lock.acquire(blocking=False):
                raise RuntimeError("InteractiveSession is already executing")
            self._cell_number += 1
            cell_number = self._cell_number

        try:
            resolved_name = source_name or f"<interactive:{cell_number}>"
            if resolved_name in self._source_names:
                raise ValueError(f"Interactive source name {resolved_name!r} was already used")
            self._source_names.add(resolved_name)

            program = parse_program(source, source_path=resolved_name)
            prepared = self._compiler.prepare_fragment(
                program,
                import_source_path=self._import_source_path(resolved_name),
            )
            return self._compiler.execute_fragment(
                prepared,
                self._env,
                self._top_level_state,
            )
        finally:
            self._execute_lock.release()

    def _import_source_path(self, source_name: str) -> Path:
        if source_name.startswith("<") and source_name.endswith(">"):
            return self._cwd / ".kedi-interactive.kedi"
        path = Path(source_name)
        return path if path.is_absolute() else self._cwd / path

    def close(self) -> None:
        """Close resources owned by this session. Calling twice is harmless."""
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._execute_lock.locked():
                raise RuntimeError("Cannot close InteractiveSession while it is executing")
            self._closed = True
        if self._close_conversation_state and self._conversation_state is not None:
            self._conversation_state.close()

    def __enter__(self) -> InteractiveSession:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("InteractiveSession is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
