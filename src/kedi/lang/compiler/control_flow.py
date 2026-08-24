"""Deterministic runtime helpers for native Kedi control flow."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from ...core import KediRuntime
from ...errors import KediTraceFrame, LoopIterationLimitError
from ..ast import ConditionalLoopStmt, IfStmt, LoopStmt
from .environment import assert_not_cli_args_identifier
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
def loop_binding(
    rt: KediRuntime,
    env: dict[str, Any],
    name: str,
    value: Any,
) -> Iterator[None]:
    """Install a synthetic binder and restore every environment it can mutate."""
    assert_not_cli_args_identifier(name)
    targets = _binding_targets(rt, env)
    snapshots = [
        (target, name in target, dict.__getitem__(target, name) if name in target else None)
        for target in targets
    ]

    # ScopedDict propagates ordinary assignments to its closure. A loop binder is
    # synthetic and local, so install it directly and restore any body propagation
    # from the snapshots below.
    dict.__setitem__(env, name, value)
    current = rt._current_env.get()
    if isinstance(current, dict) and current is not env:
        dict.__setitem__(current, name, value)

    try:
        yield
    finally:
        for target, existed, previous in reversed(snapshots):
            if existed:
                dict.__setitem__(target, name, previous)
            elif name in target:
                dict.__delitem__(target, name)


def _binding_targets(rt: KediRuntime, env: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(target: object) -> None:
        while isinstance(target, dict) and id(target) not in seen:
            mapping = cast(dict[str, Any], target)
            targets.append(mapping)
            seen.add(id(mapping))
            target = getattr(mapping, "_closure", None)

    add(env)
    add(rt._current_env.get())
    return targets
