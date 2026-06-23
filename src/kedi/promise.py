"""Promise-pipelining primitives for non-blocking Kedi template calls.

A template's output fields become :class:`KediPromise` objects wrapping a shared
``concurrent.futures.Future``. The interpreter is *promise-aware*: it threads promises
through the environment, builds dependency edges, and derives lazy ``.map()`` transforms
— it NEVER blocks the main thread on a promise. Blocking (``resolve``) happens at exactly
two genuine points: when user Python touches a value (via ``KediEnv`` lookup) and at the
end-of-run drain.

Design principle (loud, never silent): a :class:`KediPromise` is OPAQUE. Its real type is
``KediPromise`` so ``isinstance(x, KediPromise)`` is cheap and never forces. Every *value-use*
operation (``str``, attribute access, ``bool``, iteration, arithmetic, ``==``, ``hash``,
calling, indexing, …) RAISES :class:`KediPromiseLeak`. A promise that reaches a value context
without being resolved indicates a missed deferral or broken interpreter routing — it crashes
loudly instead of silently serializing or producing a wrong result.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, Callable, NoReturn, Sequence

__all__ = [
    "KediPromise",
    "KediPromiseLeak",
    "force",
    "when_all",
    "lazy_join",
]

# Slot names that must never be treated as value-use by ``__getattr__``.
_INTERNAL_ATTRS = frozenset({"_future", "_key", "_lock", "_resolved", "_value"})


class KediPromiseLeak(RuntimeError):
    """A :class:`KediPromise` was used as a concrete value without being resolved.

    This is always a bug — either a consumption point that should ``force()`` / defer the
    value did neither, or the interpreter's name-resolution routing (e.g. ``KediEnv``) failed
    to resolve a promise before handing it to user code. Surfaced loudly, never swallowed.
    """


class KediPromise:
    """An opaque, loud view over one output key of a shared ``Future``.

    One ``Future`` is created per template invocation (one LLM request) and produces a
    ``{key: value}`` mapping; one ``KediPromise`` view exists per output key, all sharing the
    Future. ``resolve()`` blocks on the Future, selects this view's key, memoizes, and returns
    the concrete value. ``map(fn)`` derives a new promise via a completion callback (no worker
    slot). Everything else raises :class:`KediPromiseLeak`.
    """

    __slots__ = ("_future", "_key", "_lock", "_resolved", "_value")

    def __init__(self, future: Future, key: str | None) -> None:
        self._future = future
        self._key = key
        self._lock = threading.Lock()
        self._resolved = False
        self._value: Any = None

    # -- the only legitimate value-producing operations --------------------------------

    def resolve(self) -> Any:
        """Block on the underlying Future, select this key, memoize, and return the value."""
        if self._resolved:
            return self._value
        with self._lock:
            if self._resolved:  # pragma: no cover - double-checked locking (race-only)
                return self._value
            result = self._future.result()  # blocks; re-raises a failed Future's exception
            value = result[self._key] if self._key is not None else result
            self._value = value
            self._resolved = True
            return value

    def map(self, fn: Callable[[Any], Any]) -> "KediPromise":
        """Return a derived promise that applies ``fn`` to the resolved value.

        Runs ``fn`` in the source Future's completion callback (no pool slot), so laziness and
        pipelining are preserved. A failure in the source or in ``fn`` propagates to the derived
        promise.
        """
        derived: Future = Future()
        key = self._key

        def _cb(f: Future) -> None:
            exc = f.exception()
            if exc is not None:
                derived.set_exception(exc)
                return
            try:
                base = f.result()
                base_val = base[key] if key is not None else base
                derived.set_result(fn(base_val))  # bare value; resolved via key=None
            except BaseException as inner:  # noqa: BLE001 - propagate to the derived promise
                derived.set_exception(inner)

        self._future.add_done_callback(_cb)
        return KediPromise(derived, None)

    # -- safe, non-resolving introspection ---------------------------------------------

    def __repr__(self) -> str:
        # NEVER resolves and NEVER raises — safe for logging / debuggers / error rendering.
        state = "resolved" if self._resolved else "pending"
        return f"<KediPromise key={self._key!r} {state}>"

    # Copy/pickle must not silently force or pickle the live Future: identity is correct.
    def __copy__(self) -> "KediPromise":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "KediPromise":
        return self

    def __reduce__(self) -> Any:
        raise KediPromiseLeak(
            "Cannot pickle a KediPromise; resolve it first (this should never happen internally)."
        )

    # -- everything else is LOUD --------------------------------------------------------

    def _leak(self, op: str) -> NoReturn:
        raise KediPromiseLeak(
            f"KediPromise(key={self._key!r}) was used via {op} without being resolved. "
            "An unresolved promise reached a value context — this is a deferral/routing bug. "
            "Internal code must force() it at a genuine consumption point or thread it as a dep."
        )

    def __getattr__(self, name: str) -> Any:
        # Dunder lookups (copy/pickle/etc.) and internal slot probes must not be value-use.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _INTERNAL_ATTRS:  # pragma: no cover - recursion guard for pre-init slot access
            raise AttributeError(name)
        self._leak(f"attribute access .{name}")

    def __str__(self) -> str:
        self._leak("str()")

    def __format__(self, spec: str) -> str:
        self._leak("format()/f-string")

    def __bytes__(self) -> bytes:
        self._leak("bytes()")

    def __bool__(self) -> bool:
        self._leak("bool()/truth-test")

    def __len__(self) -> int:
        self._leak("len()")

    def __iter__(self) -> Any:
        self._leak("iteration")

    def __next__(self) -> Any:
        self._leak("next()")

    def __getitem__(self, item: Any) -> Any:
        self._leak("indexing []")

    def __contains__(self, item: Any) -> bool:
        self._leak("'in'")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._leak("call()")

    def __eq__(self, other: Any) -> bool:
        self._leak("==")

    def __ne__(self, other: Any) -> bool:
        self._leak("!=")

    def __hash__(self) -> int:
        self._leak("hash()")

    def __lt__(self, other: Any) -> bool:
        self._leak("<")

    def __le__(self, other: Any) -> bool:
        self._leak("<=")

    def __gt__(self, other: Any) -> bool:
        self._leak(">")

    def __ge__(self, other: Any) -> bool:
        self._leak(">=")

    def __add__(self, other: Any) -> Any:
        self._leak("+")

    def __radd__(self, other: Any) -> Any:
        self._leak("+")

    def __sub__(self, other: Any) -> Any:
        self._leak("-")

    def __mul__(self, other: Any) -> Any:
        self._leak("*")

    def __rmul__(self, other: Any) -> Any:
        self._leak("*")

    def __int__(self) -> int:
        self._leak("int()")

    def __float__(self) -> float:
        self._leak("float()")

    def __index__(self) -> int:
        self._leak("__index__")


def force(value: Any) -> Any:
    """Resolve a :class:`KediPromise` to its concrete value; pass everything else through."""
    if isinstance(value, KediPromise):
        return value.resolve()
    return value


def when_all(deps: Sequence[Future], then: Callable[[], Any]) -> None:
    """Invoke ``then`` exactly once after every Future in ``deps`` has completed.

    Uses ``Future.add_done_callback`` (continuation-passing): no thread ever *blocks* waiting on
    a dependency, so a bounded worker pool cannot starve on dependency chains. ``then`` is
    invoked regardless of whether deps succeeded — the downstream job resolves the deps and a
    failed dep's exception propagates there (attributed). The countdown is lock-guarded so the
    final two completions racing on two threads cannot double-fire.
    """
    pending = [d for d in deps if d is not None]
    if not pending:
        then()
        return

    lock = threading.Lock()
    state = {"remaining": len(pending), "fired": False}

    def _cb(_f: Future) -> None:
        with lock:
            state["remaining"] -= 1
            do_fire = state["remaining"] == 0 and not state["fired"]
            if do_fire:
                state["fired"] = True
        if do_fire:
            then()

    for dep in pending:
        dep.add_done_callback(_cb)


def lazy_join(
    parts: Sequence[Any],
    combine: Callable[[list[Any]], Any],
) -> Any:
    """Combine ``parts`` (a mix of concrete values and promises) via ``combine``.

    If no part is a promise, ``combine`` runs immediately and the concrete result is returned.
    Otherwise a derived promise is returned whose value is ``combine([resolved parts...])``,
    computed after the promise parts resolve. ``combine`` must encapsulate the exact join/strip
    logic of the caller so deferred and immediate paths produce byte-identical output.
    """
    promise_futures = [p._future for p in parts if isinstance(p, KediPromise)]
    if not promise_futures:
        return combine(list(parts))

    derived: Future = Future()

    def _build() -> None:
        try:
            resolved = [p.resolve() if isinstance(p, KediPromise) else p for p in parts]
            derived.set_result(combine(resolved))  # bare value; resolved via key=None
        except BaseException as exc:  # noqa: BLE001 - propagate to the derived promise
            derived.set_exception(exc)

    when_all(promise_futures, _build)
    return KediPromise(derived, None)
