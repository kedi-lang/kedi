from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from kedi.agent_adapter.conversation import ConversationState
from kedi.artifacts import ArtifactError, ArtifactManager, ArtifactPolicy


@pytest.mark.parametrize("store", ["memory", "file"])
@pytest.mark.parametrize("retirement", ["release", "expiry"])
def test_active_query_finishes_before_retired_payload_is_reclaimed(
    tmp_path, monkeypatch, store, retirement
):
    clock = [0.0]
    manager = ArtifactManager(monotonic=lambda: clock[0])
    state = ConversationState(artifacts=manager)
    policy = ArtifactPolicy(store=store, path=tmp_path, threshold_bytes=0, ttl_seconds=60)
    ref = manager.artifactize(
        "CURRENT shipment receipt: shipment-42. Delivered 37 units.", policy=policy
    ).ref.ref_id
    state.record_exchange("Keep this receipt", {"ref": ref})
    before = state.snapshot()
    entered, retired = Event(), Event()
    backend = manager._ref_stores[ref]
    original = backend.read_projection

    def projection(session_id, ref_id, **kwargs):
        entered.set()
        assert retired.wait(5), "Retirement did not finish while the query held a lease"
        return original(session_id, ref_id, **kwargs)

    monkeypatch.setattr(backend, "read_projection", projection)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            query = pool.submit(manager.query, ref, "CURRENT shipment receipt delivered")
            try:
                assert entered.wait(5)
                if retirement == "release":
                    manager.release(ref)
                else:
                    clock[0] = 61.0
                    manager.cleanup_expired()
                assert state.snapshot() == before
            finally:
                retired.set()
            assert "shipment-42" in str(query.result(timeout=5))
        assert manager.stats()["active_bytes"] == 0
        with pytest.raises(ArtifactError):
            manager.read(ref)
    finally:
        retired.set()
        state.close()
