"""`> agent: rpc:` selects a JSON-RPC bridge from Kedi source (#197 DSL selection)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_adapter import JsonRpcAdapter
from kedi.lang import compile_program, parse_program
from kedi.lsp.features.diagnostics import compute_diagnostics

PEER = Path(__file__).parent / "fixtures" / "jsonrpc_peer.py"
PROMPT = "[reply] << Say hello.\n= <reply>\n"


def agent(scenario: str, *, plain: bool = False) -> str:
    if plain:
        return f"> agent: rpc:\n    command: {sys.executable} {PEER} {scenario}\n"
    return f"> agent: rpc:\n    command: `[{str(sys.executable)!r}, {str(PEER)!r}, {scenario!r}]`\n"


def run(source: str) -> tuple[Any, list[JsonRpcAdapter]]:
    runtime = compile_program(parse_program(source))
    adapters: list[JsonRpcAdapter] = []
    try:
        value = runtime.run_main()
        adapters = list(runtime._connection_adapters.values())  # type: ignore[arg-type]
        return value, adapters
    finally:
        asyncio.run(runtime.aclose())


def test_a_program_selects_a_bridge_and_the_runtime_closes_it() -> None:
    value, (adapter,) = run(agent("default", plain=True) + PROMPT)
    assert value == "echo: Say hello."
    assert adapter.backend == {"name": "peer", "version": "1"}
    assert adapter._connection._process.poll() is not None


def test_one_bridge_per_command() -> None:
    inner = "  " + agent("text-only").replace("\n    command", "\n      command")
    source = (
        agent("default")
        + "[a] << First.\n[b] << Second.\n"
        + f"@third() -> str:\n{inner}  [c] << Third.\n  = <c>\n\n"
        + "= `[a, b, third()]`\n"
    )
    value, adapters = run(source)
    assert value == ["echo: First.", "echo: Second.", "echo: Third."]
    assert sorted(adapter.capabilities.supports_structured_output for adapter in adapters) == [
        False,
        True,
    ]


def test_requires_is_checked_against_the_launched_bridge() -> None:
    source = "> requires: structured_output\n" + PROMPT
    assert run(agent("default") + source)[0] == "echo: Say hello."
    with pytest.raises(Exception, match="structured_output"):
        run(agent("text-only") + source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (agent("framework") + PROMPT, "is an agent-framework bridge; select it from Python"),
        ("> agent: rpc\n" + PROMPT, "> agent: rpc: needs a connection with 'command'"),
        (
            "> agent: rpc:\n    command: x\n    cwd: /tmp\n" + PROMPT,
            "RPC agent connection has unsupported field(s): cwd",
        ),
        ("> agent: rpc:\n    cwd: /tmp\n" + PROMPT, "unsupported field(s): cwd"),
    ],
)
def test_bad_selections_fail_before_the_call(source: str, message: str) -> None:
    with pytest.raises(Exception, match=message.replace("(", r"\(").replace(")", r"\)")):
        run(source)


def test_the_editor_accepts_the_rpc_connection_block() -> None:
    source = agent("default") + "> requires: structured_output\n" + PROMPT
    diagnostics = compute_diagnostics(source, None)
    assert [d for d in diagnostics if d.severity == 1] == []
