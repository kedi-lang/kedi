"""ACP adapter client cache synchronization (audit #214, item 3.3).

_client_for_run did get-then-create on the client dict with no lock, so
concurrent invokes could create two clients for one key (one leaking, its
process never closed), and close() iterated the dict live while another
thread could insert. A cached client that had closed itself (e.g. on an
unresponsive cancel) also stayed in the cache, so the next invoke got the
closed client instead of a fresh one. get/create/close are now serialized
and closed cached clients are replaced.
"""

from __future__ import annotations

import threading
import time

import pytest

from kedi.agent_adapter.adapters import acp as acp_module
from kedi.agent_adapter.adapters.acp import ACPAdapter, ACPClient


def _adapter() -> ACPAdapter:
    return ACPAdapter(command=["true"])


def test_client_for_run_reuses_one_client_per_key() -> None:
    adapter = _adapter()
    try:
        first, _, _ = adapter._client_for_run({})
        second, _, _ = adapter._client_for_run({})
        assert first is second
        assert not first.closed
    finally:
        adapter.close()


def test_client_for_run_replaces_closed_cached_client() -> None:
    adapter = _adapter()
    try:
        first, _, _ = adapter._client_for_run({})
        first.close()
        assert first.closed
        second, _, _ = adapter._client_for_run({})
        assert second is not first
        assert not second.closed
    finally:
        adapter.close()


def test_client_for_run_creates_one_client_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight threads racing get-or-create with a slow constructor must end
    up sharing exactly one client."""
    created: list[ACPClient] = []
    real_client = ACPClient

    class SlowClient(real_client):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            time.sleep(0.05)
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            created.append(self)

    monkeypatch.setattr(acp_module, "ACPClient", SlowClient)
    adapter = _adapter()
    barrier = threading.Barrier(8)
    results: list[ACPClient] = []
    errors: list[BaseException] = []

    def get_client() -> None:
        try:
            barrier.wait(timeout=5)
            client, _, _ = adapter._client_for_run({})
            results.append(client)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted
            errors.append(exc)

    threads = [threading.Thread(target=get_client) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        adapter.close()
    assert errors == []
    assert len(created) == 1
    assert len({id(client) for client in results}) == 1


def test_close_races_with_client_creation_without_errors() -> None:
    """close() iterating the cache while another thread creates a client
    must not raise (dict changed size during iteration) or leak."""
    adapter = _adapter()
    try:
        first, _, _ = adapter._client_for_run({"cwd": "/tmp"})
        second, _, _ = adapter._client_for_run({"cwd": "/"})
        assert first is not second
        barrier = threading.Barrier(2)
        original_close = first.close

        def slow_close() -> None:
            barrier.wait(timeout=5)
            original_close()

        first.close = slow_close  # type: ignore[method-assign]
        errors: list[BaseException] = []
        created: list[ACPClient] = []

        def close_adapter() -> None:
            try:
                adapter.close()
            except BaseException as exc:  # noqa: BLE001 - collected and asserted
                errors.append(exc)

        def create_client() -> None:
            try:
                barrier.wait(timeout=5)
                client, _, _ = adapter._client_for_run({"cwd": "/var"})
                created.append(client)
            except BaseException as exc:  # noqa: BLE001 - collected and asserted
                errors.append(exc)

        closer = threading.Thread(target=close_adapter)
        creator = threading.Thread(target=create_client)
        closer.start()
        creator.start()
        closer.join(timeout=10)
        creator.join(timeout=10)
        assert not closer.is_alive() and not creator.is_alive()
        assert errors == []
        # The client created during close() is fresh and owned by the cache.
        assert len(created) == 1
        assert not created[0].closed
        assert adapter._clients
    finally:
        adapter.close()
