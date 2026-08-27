"""Deterministic runtime helpers for native Kedi control flow."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from ...core import KediRuntime
from ...errors import KediTraceFrame, LoopIterationLimitError
from ..ast import ConditionalLoopStmt, IfStmt, LoopStmt
from .environment import ScopeFrame, assert_not_cli_args_identifier
from .evaluation import evaluate_python_inline, render_expression


def evaluate_condition(
    rt: KediRuntime,
    stmt: IfStmt | ConditionalLoopStmt,
    env: dict[str, Any],
) -> bool:
    """Evaluate one deterministic or model-classified control condition."""
    label = "loop condition" if isinstance(stmt, ConditionalLoopStmt) else "if condition"
    with rt.trace_frame(
        KediTraceFrame(
            kind="condition",
            label=label,
            span=stmt.condition_loc or stmt.loc,
        )
    ):
        if stmt.condition_segments is not None:
            claim = render_expression(rt, stmt.condition_segments, env, context="condition")
            return rt._evaluate_claim(claim)
        result = evaluate_python_inline(
            rt,
            stmt.condition_code,
            env,
            span=stmt.condition_loc or stmt.loc,
        )
        if type(result) is not bool:
            raise TypeError(
                "Kedi control-flow condition must return an exact bool; "
                f"received {type(result).__name__}"
            )
        return result


def evaluate_iterable(
    rt: KediRuntime,
    stmt: LoopStmt,
    env: dict[str, Any],
) -> Iterator[Any]:
    """Evaluate one loop header and return its iterator without materializing it."""
    evaluated = evaluate_python_inline(
        rt,
        stmt.iterable_code,
        env,
        span=stmt.iterable_loc or stmt.loc,
    )
    if not hasattr(evaluated, "__iter__") or not callable(getattr(evaluated, "__iter__", None)):
        raise TypeError(
            f"Kedi loop expression must return an Iterable; received {type(evaluated).__name__}"
        )
    try:
        return iter(evaluated)
    except TypeError as exc:
        raise TypeError(
            "Kedi loop expression returned an invalid Iterable; "
            f"received {type(evaluated).__name__}"
        ) from exc


@contextmanager
def managed_loop_iterator(values: Iterator[Any]) -> Iterator[Iterator[Any]]:
    """Close a loop iterator when traversal finishes or its body fails."""
    try:
        yield values
    finally:
        close = getattr(values, "close", None)
        if callable(close):
            close()


@contextmanager
def branch_trace(rt: KediRuntime, stmt: IfStmt, *, alternative: bool) -> Iterator[None]:
    """Add the selected branch to Kedi's source-mapped traceback stack."""
    with rt.trace_frame(
        KediTraceFrame(
            kind="if",
            label="else branch" if alternative else "if branch",
            span=stmt.loc,
        )
    ):
        yield


@contextmanager
def loop_trace(rt: KediRuntime, stmt: LoopStmt | ConditionalLoopStmt) -> Iterator[None]:
    """Cover iterable evaluation and traversal with one loop trace frame."""
    with rt.trace_frame(KediTraceFrame(kind="loop", label="loop", span=stmt.loc)):
        yield


def assert_conditional_loop_capacity(rt: KediRuntime, completed_iterations: int) -> None:
    """Permit exactly the configured count and reject the next true iteration."""
    if completed_iterations >= rt.loop_iteration_limit:
        raise LoopIterationLimitError(
            f"Kedi conditional loop exceeded its iteration limit ({rt.loop_iteration_limit})"
        )


@contextmanager
def child_scope(rt: KediRuntime, parent: dict[str, Any]) -> Iterator[ScopeFrame]:
    """Activate one lexical child frame and restore the previous frame afterward."""
    frame = ScopeFrame(parent=parent)
    token = rt._current_env.set(frame)
    try:
        yield frame
    finally:
        rt._current_env.reset(token)


@contextmanager
def loop_binding(
    rt: KediRuntime,
    env: dict[str, Any],
    name: str,
    value: Any,
) -> Iterator[ScopeFrame]:
    """Create one isolated iteration frame containing its synthetic binder."""
    assert_not_cli_args_identifier(name)
    with child_scope(rt, env) as frame:
        frame.declare_synthetic(name, value)
        yield frame


def frame_promise_dependencies(frame: ScopeFrame) -> list[Any]:
    """Return unique futures produced inside one iteration frame."""
    from ...promise import KediPromise

    dependencies: list[Any] = []
    seen: set[int] = set()
    for value in dict.values(frame):
        if not isinstance(value, KediPromise):
            continue
        identity = id(value._future)
        if identity in seen:
            continue
        seen.add(identity)
        dependencies.append(value._future)
    return dependencies


@contextmanager
def active_scope(rt: KediRuntime, frame: ScopeFrame) -> Iterator[None]:
    """Reactivate a retained frame for its deferred map continuation."""
    token = rt._current_env.set(frame)
    try:
        yield
    finally:
        rt._current_env.reset(token)


def await_map_futures(futures: list[Any]) -> None:
    """Drain every map continuation and then surface the first failure."""
    first_error: BaseException | None = None
    for future in futures:
        try:
            future.result()
        except BaseException as exc:  # noqa: BLE001 - preserve scheduler failure semantics
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def run_map_continuations(
    rt: KediRuntime,
    stmt: LoopStmt,
    frames: list[ScopeFrame],
    continuation: Any,
) -> None:
    """Schedule one continuation per retained iteration frame and join the stage."""
    if not frames or stmt.map_body is None:
        return

    from ...telemetry import add_counter, record_histogram, runtime_span

    attributes = {"kedi.map.iteration_count": len(frames)}
    with rt.trace_frame(KediTraceFrame(kind="map", label="map stage", span=stmt.map_loc)):
        with runtime_span(
            "map stage",
            "map_stage",
            attributes=attributes,
            metric_attributes={},
        ):
            add_counter(
                "runtime",
                "kedi.map.iterations",
                len(frames),
                attributes={"state": "scheduled"},
            )
            futures: list[Any] = []

            def execute(index: int, frame: ScopeFrame) -> None:
                add_counter(
                    "runtime",
                    "kedi.map.iterations",
                    attributes={"state": "started"},
                )
                try:
                    continuation(index, frame)
                except BaseException:  # noqa: BLE001 - retain the original map failure
                    add_counter(
                        "runtime",
                        "kedi.map.iterations",
                        attributes={"state": "failed"},
                    )
                    raise
                add_counter(
                    "runtime",
                    "kedi.map.iterations",
                    attributes={"state": "completed"},
                )

            for index, frame in enumerate(frames):
                dependencies = frame_promise_dependencies(frame)
                future = rt._engine.schedule(
                    dependencies,
                    lambda frame=frame, index=index: execute(index, frame),
                )
                futures.append(future)
            wait_started_at = perf_counter()
            try:
                await_map_futures(futures)
            finally:
                record_histogram(
                    "runtime",
                    "kedi.map.wait.duration",
                    max(0.0, perf_counter() - wait_started_at),
                    unit="s",
                )
