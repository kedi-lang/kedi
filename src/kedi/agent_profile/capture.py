from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Any

from ..errors import KediExecutionError
from .profile import AgentProfile


class CapturedAgentProfile:
    """One definition-site configuration prefix, evaluated only if called directly.

    Normal program execution supplies the already evaluated lexical profile when
    it reaches the procedure declaration. Exported procedures may instead run
    without executing the program; they share these source-ordered capture cells.
    """

    def __init__(
        self,
        previous: AgentProfile | CapturedAgentProfile,
        evaluate: Callable[[dict[str, Any]], AgentProfile],
        env: dict[str, Any],
    ) -> None:
        self._previous = previous.frozen_copy() if isinstance(previous, AgentProfile) else previous
        self._evaluate = evaluate
        self._env = dict(env)
        self._definition_cell = _CaptureCell()

    def resolve(self) -> AgentProfile:
        cell = self._cell()
        with cell.lock:
            if cell.value is None:
                if cell.resolving:
                    raise KediExecutionError("Recursive agent profile capture evaluation")
                cell.resolving = True
                try:
                    previous = resolve_profile_capture(self._previous)
                    cell.value = previous.merge(cell.patch(self._evaluate, self._env)).frozen_copy()
                finally:
                    cell.resolving = False
            return cell.value.frozen_copy()

    def apply(
        self, previous: AgentProfile, env: dict[str, Any]
    ) -> tuple[AgentProfile, AgentProfile]:
        """Bind the declaration to this execution's lexical state, not its caller."""
        cell = self._cell()
        with cell.lock:
            patch = cell.patch(self._evaluate, env)
            cell.value = previous.merge(patch).frozen_copy()
            return patch.frozen_copy(), cell.value.frozen_copy()

    def _cell(self) -> _CaptureCell:
        return _execution_cells.get().get(self, self._definition_cell)

    def __deepcopy__(self, memo: dict[int, Any]) -> CapturedAgentProfile:
        # Like a bound closure, a definition-site cell keeps its identity when a
        # workflow snapshots a procedure. Concrete results are copied by resolve.
        memo[id(self)] = self
        return self


class _CaptureCell:
    def __init__(self) -> None:
        self.lock = RLock()
        self.resolving = False
        self.value: AgentProfile | None = None
        self._patch: AgentProfile | None = None
        self._evaluating = False

    def patch(
        self, evaluate: Callable[[dict[str, Any]], AgentProfile], env: dict[str, Any]
    ) -> AgentProfile:
        if self._patch is None:
            if self._evaluating:
                raise KediExecutionError("Recursive agent profile capture evaluation")
            self._evaluating = True
            try:
                self._patch = evaluate(env).frozen_copy()
            finally:
                self._evaluating = False
        return self._patch


_execution_cells: ContextVar[dict[CapturedAgentProfile, _CaptureCell]] = ContextVar(
    "kedi_profile_capture_cells", default={}
)


@contextmanager
def profile_capture_scope(captures: Iterable[CapturedAgentProfile]) -> Iterator[None]:
    cells = {**_execution_cells.get(), **{capture: _CaptureCell() for capture in captures}}
    token = _execution_cells.set(cells)
    try:
        yield
    finally:
        _execution_cells.reset(token)


def resolve_profile_capture(
    captured: AgentProfile | CapturedAgentProfile,
) -> AgentProfile:
    return captured.resolve() if isinstance(captured, CapturedAgentProfile) else captured
