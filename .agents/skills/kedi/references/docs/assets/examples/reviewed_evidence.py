"""Embed the reviewed-evidence program with local, bounded fixture tools."""

import asyncio
from pathlib import Path

from kedi import ApprovalDecision, SubagentUsageLimits, current_subagent_execution, tool
from kedi.lang import compile_program, parse_program


def run_review(adapter, output_dir):
    destination = Path(output_dir)
    destination = destination.resolve() / "review.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    observations = []
    approvals = []
    writes = []

    @tool(risk="read_only")
    def read_evidence(suite: str) -> dict:
        """Read the local parser test fixture; other suites are unavailable."""
        if suite != "parser":
            raise ValueError("Unknown test suite")
        child = current_subagent_execution()
        if child is None:
            raise RuntimeError("Evidence must be read inside a child invocation")
        observations.append((child.run_id, child.profile.name, child.conversation_id))
        return {"suite": suite, "passed": 12, "failures": ["unicode_identifier"]}

    @tool(risk="mutating")
    def write_report(path: str, text: str) -> str:
        """Write the approved report and return its actual path."""
        requested_path = Path(path)
        if requested_path.resolve() != destination:
            raise ValueError("Unapproved destination")
        destination.write_text(text, encoding="utf-8")
        writes.append({"path": path, "text": text})
        return str(destination)

    def approve_report(request):
        approvals.append(request)
        if request.tool_name != "write_report":
            return ApprovalDecision.deny(reason="Only report writes are permitted")
        return ApprovalDecision.edit(
            {**request.arguments, "path": str(destination)},
            reason="Confine the write to this application's report file",
        )

    source = Path(__file__)
    source = source.with_suffix(".kedi")
    runtime = compile_program(
        parse_program(source.read_text(encoding="utf-8")),
        adapter=adapter,
        runtime_globals={
            "read_evidence": read_evidence,
            "write_report": write_report,
            "approve_report": approve_report,
        },
        subagent_max_depth=2,
        subagent_max_concurrency=2,
        subagent_timeout_seconds=60,
        subagent_usage_limits=SubagentUsageLimits(request_limit=4, tool_calls_limit=4),
    )
    try:
        receipt = runtime.run_main()
        return {
            "receipt": receipt,
            "observations": observations,
            "approvals": approvals,
            "writes": writes,
        }
    finally:
        asyncio.run(runtime.aclose())
