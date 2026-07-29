from __future__ import annotations

import atexit
import weakref
from threading import Condition, Thread
from typing import Protocol


class CleanupTarget(Protocol):
    @property
    def cleanup_interval_seconds(self) -> float: ...

    def cleanup_expired(self) -> int: ...


class ArtifactCleanupService:
    """One daemon cleanup loop shared by all active artifact managers."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._targets: weakref.WeakSet[CleanupTarget] = weakref.WeakSet()
        self._thread: Thread | None = None
        self._closing = False
        self._generation = 0

    def register(self, target: CleanupTarget) -> None:
        with self._condition:
            if self._closing:
                return
            self._targets.add(target)
            self._generation += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = Thread(
                    target=self._run,
                    name="kedi-artifact-cleanup",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()

    def unregister(self, target: CleanupTarget) -> None:
        with self._condition:
            self._targets.discard(target)
            self._generation += 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._generation += 1
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._closing:
                    return
                generation = self._generation
                targets = tuple(self._targets)
            # Never call a target while holding the service condition. ArtifactManager
            # registers while holding its own lock, so the inverse order can deadlock.
            timeout = min(
                (target.cleanup_interval_seconds for target in targets),
                default=60.0,
            )
            with self._condition:
                if self._closing:
                    return
                if generation == self._generation:
                    self._condition.wait(timeout=max(0.1, timeout))
                if self._closing:
                    return
                targets = tuple(self._targets)
            for target in targets:
                try:
                    target.cleanup_expired()
                except Exception:
                    continue


cleanup_service = ArtifactCleanupService()
atexit.register(cleanup_service.close)
