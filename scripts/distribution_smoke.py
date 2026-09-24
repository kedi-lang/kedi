"""Offline behavioral checks for installed release wheels, not editable sources."""

import importlib.metadata
import io
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import kedi
from kedi.lang.ast import AwaitStmt, BudgetDirectiveStmt, ShowStmt, TaskGroupStmt, TaskStmt
from kedi.lang.parser import parse_program


def main() -> None:
    assert Path(kedi.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    for name in ("kedi", "tree-sitter-kedi"):
        distribution = importlib.metadata.distribution(name)
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        assert not direct_url.get("dir_info", {}).get("editable"), name

    sources = (
        "> show: ready\n",
        "> budget:\n    request_limit: 5\n= ready\n",
        "> task [job]: reviewer:\n    >> The recommendation is [answer: str].\n"
        "> await [result]: job\n",
        "> task_group:\n    > await [result]: job\n    > process:\n        > show: done\n",
    )
    for source, expected_type in zip(
        sources, (ShowStmt, BudgetDirectiveStmt, TaskStmt, TaskGroupStmt)
    ):
        assert isinstance(parse_program(source).toplevel_body[0], expected_type), source
    assert isinstance(parse_program(sources[2]).toplevel_body[1], AwaitStmt)

    scripts = Path(sysconfig.get_path("scripts"))
    suffix = ".exe" if os.name == "nt" else ""
    entrypoints = {
        name: scripts / f"{name}{suffix}" for name in ("kedi", "kedi-lsp", "kedi-terminal-bench")
    }
    for executable in entrypoints.values():
        assert executable.is_file(), executable
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"

    cases = (
        ("kedi", ["--help"], "Usage:"),
        ("kedi", ["-c", "= `1 + 1`"], "2"),
        ("kedi", ["-c", "> show: visible\n= finished"], "visible"),
        ("kedi", ["-c", "> import: filesystem\n= ready"], "ready"),
        ("kedi", ["a2a", "--help"], "Usage:"),
        ("kedi", ["notebook", "--help"], "Usage:"),
        ("kedi-terminal-bench", ["--help"], "Usage:"),
    )
    for name, arguments, expected in cases:
        result = subprocess.run(
            [str(entrypoints[name]), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=60,
            check=True,
        )
        assert expected in result.stdout, (arguments, result.stdout, result.stderr)

    messages = (
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"processId": None, "rootUri": None, "capabilities": {}},
        },
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": None},
        {"jsonrpc": "2.0", "method": "exit", "params": None},
    )
    frames = []
    for message in messages:
        body = json.dumps(message).encode("utf-8")
        frames.append(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    result = subprocess.run(
        [str(entrypoints["kedi-lsp"])],
        input=b"".join(frames),
        capture_output=True,
        env=env,
        check=True,
        timeout=60,
    )
    stream = io.BytesIO(result.stdout)
    responses = {}
    while line := stream.readline():
        headers = {}
        while line != b"\r\n":
            key, value = line.decode("ascii").split(":", 1)
            headers[key.lower()] = value.strip()
            line = stream.readline()
            assert line, result.stdout
        response = json.loads(stream.read(int(headers["content-length"])))
        if "id" in response:
            responses[response["id"]] = response
    assert 1 in responses and "result" in responses[1], (responses, result.stderr)
    assert 2 in responses and "result" in responses[2], (responses, result.stderr)
    assert "capabilities" in responses[1]["result"], responses
    assert responses[2]["result"] is None, responses
    print("Installed wheel grammar, dependencies, all entrypoints, and LSP lifecycle passed.")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        for output in (error.stdout, error.stderr):
            if output:
                print(
                    output.decode("utf-8", errors="replace")
                    if isinstance(output, bytes)
                    else output,
                    file=sys.stderr,
                )
        raise
