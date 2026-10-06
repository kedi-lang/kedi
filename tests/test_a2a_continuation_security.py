import asyncio
import json
import logging
import ssl

import httpx
import pytest
from a2a.server.tasks import TaskUpdater
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError

from tests.a2a_gateway import Gateway
from tests.a2a_peer import SuspensionPeer
from tests.a2a_test_support import client_probe, disk_full_hook, served_program


def test_disk_fault_rolls_back_real_transaction_once(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'state.sqlite3'}")
    marker = tmp_path / "fault.json"
    hook = disk_full_hook(marker)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE tasks (value TEXT)"))
            connection.execute(text("CREATE TABLE kedi_a2a_sessions (value TEXT)"))
        event.listen(engine, "before_cursor_execute", hook)
        with pytest.raises(OperationalError, match="database or disk is full"):
            with engine.begin() as connection:
                connection.execute(text("INSERT INTO tasks VALUES ('completed')"))
                connection.execute(text("INSERT INTO kedi_a2a_sessions VALUES ('checkpoint')"))
        assert json.loads(marker.read_text())["error"] == "SQLITE_FULL"
        with engine.begin() as connection:
            assert connection.scalar(text("SELECT count(*) FROM tasks")) == 0
            assert connection.scalar(text("SELECT count(*) FROM kedi_a2a_sessions")) == 0
            connection.execute(text("INSERT INTO tasks VALUES ('completed')"))
            connection.execute(text("INSERT INTO kedi_a2a_sessions VALUES ('checkpoint')"))
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT count(*) FROM tasks")) == 1
            assert connection.scalar(text("SELECT count(*) FROM kedi_a2a_sessions")) == 1
    finally:
        event.remove(engine, "before_cursor_execute", hook)
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("state", ["input_required", "auth_required"])
@pytest.mark.parametrize("operation", ["explicit", "implicit"])
async def test_official_peer_continuation_retains_ids(tmp_path, backend, state, operation):
    served = served_program(tmp_path, backend)
    peer = SuspensionPeer(served, "test-only", state, [])
    async with peer:
        result = await asyncio.wait_for(
            client_probe(peer.endpoint, "test-only", operation=operation), 30
        )
        assert "error" not in result, result
        assert result["output"] == "receipt"
        assert result["suspended"]["state"] == state
        assert result["snapshot_state"] == state
        assert result["final"]["state"] == "completed"
        for key in ("task_id", "context_id"):
            assert result["final"][key] == result["suspended"][key]
        assert [request["continuation"] for request in peer.requests] == [False, True]
    assert peer.running.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["untrusted-cert", "card-origin", "card-redirect", "rpc-redirect"])
async def test_tls_origin_and_redirect_probes_do_not_reach_sink(tmp_path, mode):
    served = served_program(tmp_path, "pydantic")
    async with SuspensionPeer(served, "test-only", "input_required", []) as peer:
        async with Gateway(peer.endpoint, tmp_path, mode) as gateway:
            # The alternate origin itself is reachable with the same valid CA.
            trust = ssl.create_default_context(cafile=str(gateway.ca))
            async with httpx.AsyncClient(verify=trust, trust_env=False) as client:
                response = await client.get(gateway.sink + "/health")
                assert response.status_code == 200
            gateway.sink_requests.clear()
            result = await asyncio.wait_for(
                client_probe(
                    gateway.endpoint,
                    "test-only",
                    ca=None if mode == "untrusted-cert" else gateway.ca,
                ),
                30,
            )
            assert "error" in result and "output" not in result, result
            detail = " ".join(item["message"] for item in result["error"])
            if mode == "untrusted-cert":
                assert "CERTIFICATE_VERIFY_FAILED" in detail
                assert not gateway.requests
            elif mode == "card-origin":
                assert "different origin" in detail
            else:
                assert "307" in detail or "redirect" in detail.lower()
            assert gateway.sink_requests == []
            assert peer.requests == []


@pytest.mark.asyncio
async def test_trusted_tls_supports_continuation(tmp_path):
    served = served_program(tmp_path, "pydantic")
    async with SuspensionPeer(served, "test-only", "input_required", []) as peer:
        async with Gateway(peer.endpoint, tmp_path, "tls") as gateway:
            result = await asyncio.wait_for(
                client_probe(gateway.endpoint, "test-only", ca=gateway.ca, operation="explicit"),
                30,
            )
            assert result.get("output") == "receipt", result
            assert not gateway.sink_requests


@pytest.mark.asyncio
async def test_delayed_continuations_complete_http_responses(tmp_path, caplog):
    logger = logging.getLogger("uvicorn.error")
    logger.addHandler(caplog.handler)
    try:
        for state in ("input_required", "auth_required"):
            for backend in ("pydantic", "langchain"):
                served = served_program(tmp_path, backend)
                peer = SuspensionPeer(served, "test-only", state, [])
                execute = peer.delegate.execute

                async def delayed(context, event_queue, execute=execute):
                    await asyncio.sleep(0.15)
                    await execute(context, event_queue)

                peer.delegate.execute = delayed
                async with peer:
                    for operation in ("explicit", "implicit"):
                        result = await asyncio.wait_for(
                            client_probe(peer.endpoint, "test-only", operation=operation), 30
                        )
                        assert result.get("output") == "receipt", result
                assert peer.running.done()
                assert peer.responses and all(item["complete"] for item in peer.responses)
        errors = [
            record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR
        ]
        assert errors == []
    finally:
        logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
async def test_peer_shutdown_does_not_interrupt_other_peer(tmp_path):
    first = SuspensionPeer(served_program(tmp_path, "pydantic"), "test-only", "auth_required", [])
    second = SuspensionPeer(
        served_program(tmp_path, "langchain"), "test-only", "input_required", []
    )
    entered, release = asyncio.Event(), asyncio.Event()
    execute = first.delegate.execute
    close = second.served.aclose

    async def delayed(context, event_queue):
        await TaskUpdater(event_queue, context.task_id, context.context_id).start_work()
        await asyncio.sleep(0.1)
        entered.set()
        await release.wait()
        await execute(context, event_queue)

    async def slow_shutdown():
        await asyncio.sleep(0.8)
        await close()

    first.delegate.execute = delayed
    second.served.aclose = slow_shutdown
    async with first:
        async with second:
            probe = asyncio.create_task(
                client_probe(first.endpoint, "test-only", operation="implicit")
            )
            try:
                await asyncio.wait_for(entered.wait(), 15)
            except BaseException:
                probe.cancel()
                await asyncio.gather(probe, return_exceptions=True)
                raise
        release.set()
        try:
            result = await asyncio.wait_for(probe, 30)
            assert result.get("output") == "receipt", result
        finally:
            if not probe.done():
                probe.cancel()
                await asyncio.gather(probe, return_exceptions=True)
    assert all(item["complete"] for item in first.responses), first.responses
