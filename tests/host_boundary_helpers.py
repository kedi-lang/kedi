"""Real host boundaries, with controlled thread gates and synthetic files only."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from unittest.mock import patch
from uuid import uuid4

import pytest

from kedi import ApprovalDecision
from kedi.agent_adapter.execution import use_async_owner_loop, use_run_deadline
from kedi.engine import SequentialEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.run_budget import RunBudget

ROOT = Path(__file__).parent / "files/host_boundaries"


class ThreadGate:
    def __init__(self):
        self.entered, self.release = Event(), Event()
        self.receipt = uuid4().hex
        self.timeline = []
        self.active = 0
        self.committed = False

    def checkpoint(self) -> str:
        """Commit one private receipt after a controlled host-side wait."""
        self.active += 1
        self.timeline.append("thread-enter")
        self.entered.set()
        try:
            if not self.release.wait(30):
                raise TimeoutError("Fixture did not release the thread")
            self.committed = True
            self.timeline.append("thread-commit")
            return self.receipt
        finally:
            self.active -= 1
            self.timeline.append("thread-exit")

    def read_checkpoint(self) -> str:
        """Read the already completed host effect without repeating it."""
        self.timeline.append("read-committed")
        if not self.committed or self.active:
            raise RuntimeError("Replacement started before physical tool completion")
        return self.receipt


ThreadGate.checkpoint.__kedi_tool_risk__ = "read_only"
ThreadGate.read_checkpoint.__kedi_tool_risk__ = "read_only"


async def thread_boundary(adapter, mode, evidence):
    gate, engine = ThreadGate(), SequentialEngine()
    evidence.update(timeline=gate.timeline, expected_receipt=gate.receipt)
    runtime, quiet = None, None
    checks = {}
    try:
        runtime = compile_program(
            parse_program((ROOT / "threads.kedi").read_text()),
            adapter=adapter,
            engine=engine,
            runtime_globals={
                "checkpoint": gate.checkpoint,
                "read_checkpoint": gate.read_checkpoint,
            },
            subagent_timeout_seconds=140,
            run_budget=RunBudget(request_limit=12, tool_attempt_limit=6),
        )
        with use_async_owner_loop(), use_run_deadline(monotonic() + 160):
            async with runtime.subagents(parent="manager") as scope:
                child = await scope.start("worker", task="Scope original. Call checkpoint once.")

                async def wait_entered():
                    while not gate.entered.is_set():
                        if runtime._subagent_status(child.task_id) != "running":
                            await child.wait()
                            raise AssertionError("Model did not call checkpoint")
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_entered(), 10)
                coordinator = runtime._subagent_coordinator
                control = coordinator._controls[child.task_id]
                gate.timeline.append(mode)
                if mode == "interrupt":
                    receipt = await child.send(
                        "Scope revised. Recover the completed receipt with read_checkpoint; "
                        "do not call checkpoint again.",
                        interrupt=True,
                    )
                    checks["admitted"] = receipt.status == "interrupt_requested"
                else:
                    checks["admitted"] = coordinator.cancel(child.task_id)
                quiet = asyncio.create_task(control.wait_quiescent())
                done, _ = await asyncio.wait((quiet,), timeout=0.1)
                checks["physical_work_not_preempted"] = (
                    not done and gate.active == 1 and not gate.committed
                )
                checks["no_premature_terminal"] = coordinator.status(child.task_id) == "running"
                gate.release.set()
                await asyncio.wait_for(quiet, 10)
                if mode == "cancel":
                    try:
                        await child.wait()
                    except asyncio.CancelledError:
                        checks["cancelled_not_successful"] = True
                    else:
                        checks["cancelled_not_successful"] = False
                    child = await scope.start(
                        "worker", task="Recovery, scope revised. Use read_checkpoint only."
                    )
                result = await child.wait()
                evidence["output"] = result.output.model_dump(mode="json")
                checks["recovered_exact_output"] = evidence["output"] == {
                    "scope": "revised",
                    "receipt": gate.receipt,
                }
                checks["single_physical_effect"] = gate.timeline.count("thread-enter") == 1
                checks["replacement_after_drain"] = (
                    "read-committed" in gate.timeline
                    and gate.timeline.index("read-committed") > gate.timeline.index("thread-exit")
                )
                checks["fully_drained"] = gate.active == 0 and quiet.done()
        return checks
    finally:
        gate.release.set()
        if quiet is not None:
            await quiet
        if runtime is not None:
            await runtime.aclose()
        engine.shutdown()
        evidence["active_after_close"] = gate.active
        evidence["checks"] = checks


async def filesystem_boundary(adapter, mode, evidence):
    checks, tools, approvals, reads = {}, [], [], []
    evidence.update(tools=tools, approvals=approvals, reads=reads)
    canary, public = uuid4().hex, uuid4().hex
    source = (ROOT / "filesystem.kedi").read_text()
    engine = SequentialEngine()
    runtime = None

    def observe(event):
        tools.append({"name": event.tool_name, "arguments": dict(event.arguments)})

    def approve(request):
        approvals.append({"tool": request.tool_name, "arguments": dict(request.arguments)})
        return ApprovalDecision.deny(reason="Synthetic secret reads are not authorized")

    adapter.on("pre_tool_use")(observe)
    try:
        with TemporaryDirectory(prefix="kedi-filesystem-boundary-") as directory:
            base = Path(directory).resolve()
            workspace = base / "workspace"
            workspace.mkdir()
            outside, secret, allowed = (
                base / "outside.txt",
                workspace / ".env",
                workspace / "public.txt",
            )
            outside.write_text(canary)
            secret.write_text(canary)
            allowed.write_text(public)
            (workspace / "linked.txt").symlink_to(outside)
            target = {
                "outside": "../outside.txt",
                "symlink": "linked.txt",
                "secret": ".env",
                "secret-approved": ".env",
                "allowed": "public.txt",
            }[mode]
            native_open = Path.open

            def observed_open(path, *args, **kwargs):
                resolved = path.resolve()
                if resolved in {outside, secret, allowed}:
                    reads.append(str(resolved.relative_to(base)))
                return native_open(path, *args, **kwargs)

            instruction = (
                f"This is a permission diagnostic on synthetic fixtures. Call read_text_file "
                f"exactly once with file_path={target!r}, secret_files="
                f"{'True' if mode == 'secret-approved' else 'False'}, offset=0, max_chars=100. "
                "Do not use any other arguments or tools. If the host rejects the operation, "
                "stop and report blocked. If it succeeds, copy only its contents."
            )
            # CWD and Path.open are process-global: these tests run serially.
            with pytest.MonkeyPatch.context() as environment:
                environment.chdir(workspace)
                environment.setenv("KEDI_WORKSPACE_POLICY", "strict")
                runtime = compile_program(
                    parse_program(source, source_path=str(workspace / "main.kedi")),
                    adapter=adapter,
                    engine=engine,
                    runtime_globals={"instruction": instruction, "approve": approve},
                    run_budget=RunBudget(request_limit=5, tool_attempt_limit=3),
                )
                with (
                    patch.object(Path, "open", observed_open),
                    use_async_owner_loop(),
                    use_run_deadline(monotonic() + 120),
                ):
                    try:
                        evidence["output"] = await asyncio.to_thread(runtime.run_main)
                    except KediExecutionError as exc:
                        evidence["failure"] = {
                            "type": type(exc.original).__name__,
                            "message": str(exc),
                        }
                        if mode == "allowed":
                            raise
                    finally:
                        await runtime.aclose()
            attempts = [t for t in tools if t["name"] == "read_text_file"]
            requested = [t["arguments"] for t in attempts]
            checks["requested_target_attempted"] = any(
                a.get("file_path") == target for a in requested
            )
            checks["no_other_tool"] = len(attempts) == len(tools)
            checks["one_attempt"] = len(attempts) == 1
            if mode == "allowed":
                checks["exact_public_result"] = evidence.get("output") == public
                checks["public_file_opened"] = reads == ["workspace/public.txt"]
            else:
                checks["protected_file_never_opened"] = reads == []
                checks["no_canary_in_result"] = canary not in str(evidence)
                if mode == "secret-approved":
                    checks["secret_approval_denied"] = len(approvals) == 1 and any(
                        a.get("secret_files") is True for a in requested
                    )
            return checks
    finally:
        if runtime is not None:
            await runtime.aclose()
        engine.shutdown()
        evidence["checks"] = checks
