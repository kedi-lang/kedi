"""Offline peers and controlled checkpoint failures for A2A regression tests."""

import asyncio
import json
import os
import sqlite3
import sys

from pydantic_ai.models.test import TestModel

from kedi.a2a.runtime import prepare_a2a_program
from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from tests.test_a2a_persistence import _ScriptedChatModel

PROMPT = "Return the requested receipt."


async def client_probe(endpoint, secret, *, ca=None, operation="invoke"):
    env = dict(os.environ, KEDI_TEST_A2A_SECRET=secret)
    if ca is not None:
        env["SSL_CERT_FILE"] = str(ca)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.files.a2a_continuation_client",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        output, stderr = await asyncio.wait_for(
            process.communicate(
                json.dumps(
                    {"endpoint": endpoint, "operation": operation, "prompt": PROMPT}
                ).encode()
            ),
            180,
        )
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            "A2A client probe exited: "
            + stderr.decode(errors="replace").replace(secret, "[REDACTED]")
        )
    result = json.loads(output)
    result["process_exit"] = process.returncode
    return result


def disk_full_hook(marker):
    def execute(cursor, statement, parameters, context):
        if "INSERT INTO kedi_a2a_sessions" in statement and not marker.exists():
            marker.write_text(
                json.dumps({"error": "SQLITE_FULL", "boundary": "session_checkpoint_insert"})
            )
            raise sqlite3.OperationalError("database or disk is full")

    return execute


def served_program(tmp_path, backend):
    path = tmp_path / "worker.kedi"
    path.write_text(
        "> profile: worker:\n  > instructions: Return the requested value.\n> export:\n  worker\n"
    )
    adapter = (
        PydanticAdapter(TestModel(call_tools=[], custom_output_text="receipt"))
        if backend == "pydantic"
        else LangChainAdapter(_ScriptedChatModel(reply="receipt"))
    )
    return prepare_a2a_program(path, entry="worker", adapter=adapter, model="test:model")
