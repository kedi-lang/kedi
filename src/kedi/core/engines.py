from __future__ import annotations

import atexit
import os
from threading import Lock
from typing import (
    Any,
    Dict,
)

_SHARED_PARALLEL_ENGINES: Dict[int, Any] = {}


_SHARED_PARALLEL_LOCK = Lock()


_PARALLEL_TRUE = frozenset({"1", "true", "yes", "on"})


_PARALLEL_FALSE = frozenset({"0", "false", "no", "off"})


def shared_thread_engine(max_workers: int = 8) -> Any:
    """Return a process-wide shared ``ThreadEngine``, one bounded pool per distinct worker count.

    A fresh engine per runtime would spawn a thread pool per program run; sharing a pool per size
    keeps concurrency capped while still honoring a different ``max_workers`` request (the engine's
    own ``__init__`` runs the interpreter routing self-check).
    """
    with _SHARED_PARALLEL_LOCK:
        engine = _SHARED_PARALLEL_ENGINES.get(max_workers)
        if engine is None:
            from kedi.engine import ThreadEngine

            engine = ThreadEngine(max_workers=max_workers)
            _SHARED_PARALLEL_ENGINES[max_workers] = engine
        return engine


@atexit.register
def _shutdown_shared_engines() -> None:
    """Shut down every cached parallel engine's thread pool at interpreter exit.

    The cache holds one bounded pool per distinct worker count for the process lifetime; this
    releases their worker threads on exit instead of leaning solely on interpreter teardown.
    """
    with _SHARED_PARALLEL_LOCK:
        engines = list(_SHARED_PARALLEL_ENGINES.values())
        _SHARED_PARALLEL_ENGINES.clear()
    for engine in engines:
        engine.shutdown()


def _default_engine(*, max_workers: int = 8) -> Any:
    """Default to bounded concurrency, with an optional ``KEDI_PARALLEL`` override.

    ``0/false/no/off`` -> sequential; unset/empty or ``1/true/yes/on`` -> parallel;
    a positive integer -> parallel with that many workers; anything else raises (no silent flip).
    """
    from kedi.engine import SequentialEngine

    raw = (os.getenv("KEDI_PARALLEL") or "").strip()
    low = raw.lower()
    if low in _PARALLEL_FALSE:
        return SequentialEngine()
    if not low or low in _PARALLEL_TRUE:
        return shared_thread_engine(max_workers=max_workers)
    try:
        workers = int(raw)
    except ValueError:
        raise RuntimeError(
            f"Invalid KEDI_PARALLEL value {raw!r}: use 0/1/true/false/on/off or a worker count."
        ) from None
    if workers <= 0:
        return SequentialEngine()
    return shared_thread_engine(max_workers=workers)
