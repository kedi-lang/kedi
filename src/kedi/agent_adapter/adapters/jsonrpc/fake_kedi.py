"""A fake Kedi: runs conformance fixtures against a bridge.

Usage::

    python testing/fake_kedi.py --bridge "node my_bridge.mjs" [--json] [fixtures/v1/*.json]

The bridge runs in test mode (``testModel``): every model request comes to the fake Kedi as
``model/complete``, answered from the fixture's ``model`` script, so no model endpoint is
needed. Every frame, both directions, is checked against ``schema/v1/protocol.json``; then
the fixture's ``expect`` is checked. A fixture whose ``capabilities`` the bridge lacks is
skipped, and reported as skipped.

Kedi ships a byte-identical copy beside its copies of the schema (``protocol.schema.json``)
and the fixtures (``fixtures/``): ``kedi adapter check --bridge`` runs it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

HERE = Path(__file__).parent
# The schema and the fixtures: beside this file in Kedi's copy, in this repository's tree here.
SCHEMA_FILE = HERE / "protocol.schema.json"
if not SCHEMA_FILE.exists():
    SCHEMA_FILE = HERE.parent / "schema" / "v1" / "protocol.json"
FIXTURES = HERE / "fixtures" if (HERE / "fixtures").is_dir() else HERE.parent / "fixtures" / "v1"
SCHEMA = json.loads(SCHEMA_FILE.read_text())
REGISTRY = Registry().with_resource(SCHEMA["$id"], Resource.from_contents(SCHEMA))
TIMEOUT = 20.0

RUN_DEFAULTS: dict[str, Any] = {
    "callKind": "invoke",
    "promptKind": "raw",
    "model": "test",
    "fallbackModels": [],
    "settings": {},
    "effort": None,
    "system": "You are being tested.",
    "instructionParts": [],
    "userContent": None,
    "output": None,
    "tools": [],
    "continuation": None,
    "session": None,
    "subagent": None,
    "hooks": [],
    "mcpServers": [],
    "admitResults": False,
}


def _validate(ref: str, value: Any) -> list[str]:
    validator = Draft202012Validator({"$ref": f"{SCHEMA['$id']}{ref}"}, registry=REGISTRY)
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
    return [f"{error.message} at {list(error.path)}" for error in errors[:3]]


@dataclass
class Outcome:
    """What running one fixture found."""

    name: str
    skipped: str | None = None
    problems: list[str] = field(default_factory=list)


class Bridge:
    """A bridge process and the frames it exchanged with the fake Kedi."""

    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.inbox: Queue[dict[str, Any]] = Queue()
        self.frames: list[tuple[str, dict[str, Any]]] = []
        self.stderr: list[str] = []
        self.next_id = 1
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        descriptor = self.process.stdout.fileno()
        buffered = b""
        while chunk := os.read(descriptor, 65536):
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                if line.strip():
                    message = json.loads(line)
                    self.frames.append(("bridge", message))
                    self.inbox.put(message)

    def _drain(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr.append(line.decode("utf-8", "replace").rstrip())

    def send(self, message: dict[str, Any]) -> None:
        message = {"jsonrpc": "2.0", **message}
        self.frames.append(("kedi", message))
        assert self.process.stdin is not None
        frame = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        self.process.stdin.write(frame.encode() + b"\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> int:
        request_id = self.next_id
        self.next_id += 1
        self.send({"id": request_id, "method": method, "params": params})
        return request_id

    def receive(self) -> dict[str, Any]:
        try:
            return self.inbox.get(timeout=TIMEOUT)
        except Empty as exc:
            tail = "\n  ".join(self.stderr[-5:])
            raise TimeoutError(f"the bridge sent nothing for {TIMEOUT:.0f}s\n  {tail}") from exc

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)


def run_fixture(command: list[str], fixture: dict[str, Any]) -> Outcome:
    outcome = Outcome(fixture["name"])
    bridge = Bridge(command)
    try:
        _exercise(bridge, fixture, outcome)
    except (TimeoutError, KeyError, ValueError) as exc:
        outcome.problems.append(f"{type(exc).__name__}: {exc}")
    finally:
        bridge.close()
    if outcome.skipped is None:
        outcome.problems += _schema_problems(bridge.frames)
    return outcome


def _exercise(bridge: Bridge, fixture: dict[str, Any], outcome: Outcome) -> None:
    hello = bridge.request(
        "initialize", {"protocolVersion": 1, "client": {"name": "fake-kedi"}, "testModel": True}
    )
    handshake = _answer_to(bridge, hello)
    capabilities = dict(handshake["result"]["capabilities"])
    # What the handshake declares beside its capabilities: compaction, with a threshold, and a
    # code mode of the backend's own.
    compaction = handshake["result"].get("compaction")
    capabilities["compaction"] = compaction is not None
    capabilities["compactionThreshold"] = bool(compaction and compaction.get("threshold"))
    capabilities["codemode"] = handshake["result"].get("codemode") is not None
    missing = [name for name in fixture.get("capabilities", []) if not capabilities.get(name)]
    if not capabilities.get("testModel"):
        missing.insert(0, "testModel")
    if missing:
        outcome.skipped = f"the bridge lacks {', '.join(missing)}"
        return
    params = {**RUN_DEFAULTS, **fixture.get("run", {}), "runId": "run-1"}
    params.setdefault("user", "Hello.")
    run_id = bridge.request("run", params)
    script = list(fixture.get("model", []))
    summaries = list(fixture.get("summaries", []))
    summarized: list[dict[str, Any]] = []
    requests_admitted = 0
    actions = {int(index): steps for index, steps in fixture.get("onModelRequest", {}).items()}
    budget = dict(fixture.get("budget", {}))
    tools = fixture.get("tools", {})
    seen: list[dict[str, Any]] = []
    while True:
        message = bridge.receive()
        method = message.get("method")
        if method is None:
            if message.get("id") == run_id:
                _check_expect(fixture.get("expect", {}), message, seen, bridge.frames, outcome)
                if fixture.get("expect", {}).get("compacted"):
                    _check_compaction(
                        fixture,
                        capabilities,
                        seen,
                        summarized,
                        requests_admitted,
                        bridge.frames,
                        outcome,
                    )
                return
            continue
        if method == "model/complete" and message["params"].get("purpose") == "compaction":
            summarized.append(message["params"])
            reply = dict(
                summaries.pop(0)
                if summaries
                else {"parts": [{"type": "text", "text": "A summary."}]}
            )
            usage = reply.pop("usage", None)
            result = {
                "message": {"role": "assistant", **reply},
                **({"usage": usage} if usage else {}),
            }
            bridge.send({"id": message["id"], "result": result})
        elif method == "model/complete":
            seen.append(message["params"])
            for action in actions.get(len(seen), []):
                _act(bridge, action, run_id)
            if message["params"].get("model") in fixture.get("failModels", []):
                down = {"code": -32002, "message": f"model {message['params']['model']} is down"}
                bridge.send({"id": message["id"], "error": down})
                continue
            reply = dict(script.pop(0) if script else {"parts": [{"type": "text", "text": "done"}]})
            usage = reply.pop("usage", None)
            result = {
                "message": {"role": "assistant", **reply},
                **({"usage": usage} if usage else {}),
            }
            bridge.send({"id": message["id"], "result": result})
        elif method == "budget/admit":
            dimension = message["params"]["dimension"]
            left = budget.get(dimension)
            if left is not None and left <= 0:
                denial = {"admitted": False, "dimension": dimension, "reason": "budget exhausted"}
                bridge.send({"id": message["id"], "result": denial})
                bridge.send({"method": "run/cancel", "params": {"runId": "run-1"}})
            else:
                if left is not None:
                    budget[dimension] = left - 1
                requests_admitted += dimension == "request"
                bridge.send({"id": message["id"], "result": {"admitted": True}})
        elif method == "tool/call":
            answer = tools.get(message["params"]["name"], {"error": "unknown tool"})
            result = (
                {"ok": True, "result": answer["result"]}
                if "result" in answer
                else {
                    "ok": False,
                    "error": answer["error"],
                }
            )
            bridge.send({"id": message["id"], "result": result})
        elif method == "history/process":
            keep = fixture.get("historyProcess", {}).get("keepLast")
            messages = message["params"]["messages"]
            kept = messages[-keep:] if keep else messages
            bridge.send({"id": message["id"], "result": {"messages": kept}})
        elif method == "run/input" or "id" not in message:
            continue  # notifications (run/usage, run/event) are only recorded
        else:
            error = {"code": -32601, "message": f"the fake Kedi does not answer {method}"}
            bridge.send({"id": message["id"], "error": error})


def _act(bridge: Bridge, action: dict[str, Any], run_id: int) -> None:
    del run_id
    method = action["send"]
    params = {"runId": "run-1", **{key: value for key, value in action.items() if key != "send"}}
    if method == "run/input":
        bridge.request(method, params)
    else:
        bridge.send({"method": method, "params": params})


def _answer_to(bridge: Bridge, request_id: int) -> dict[str, Any]:
    while True:
        message = bridge.receive()
        if message.get("id") == request_id and "method" not in message:
            return message


def _strings(value: Any) -> list[str]:
    """Every string inside ``value``."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def _contains(actual: Any, expected: Any) -> bool:
    """Whether ``actual`` holds everything ``expected`` names (objects partially, the rest equal)."""

    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _check_compaction(
    fixture: dict[str, Any],
    capabilities: dict[str, Any],
    seen: list[dict[str, Any]],
    summarized: list[dict[str, Any]],
    admitted: int,
    frames: list[tuple[str, dict[str, Any]]],
    outcome: Outcome,
) -> None:
    """A compacted run: a summary request reached Kedi, admitted, and no event reported it."""

    if not summarized:
        outcome.problems.append(
            "no summary request (model/complete with purpose compaction) reached Kedi"
        )
    if capabilities.get("governed") and admitted != len(seen) + len(summarized):
        outcome.problems.append(
            f"{admitted} requests admitted, for {len(seen)} turn and {len(summarized)} summary requests: each needs budget/admit"
        )
    texts = [
        frame.get("params", {}).get("text")
        for side, frame in frames
        if side == "bridge" and frame.get("method") == "run/event"
    ]
    for summary in fixture.get("summaries", []):
        said = json.dumps(summary["parts"])
        if any(isinstance(text, str) and text and text in said for text in texts):
            outcome.problems.append(
                "a run/event reported the summary request's text: a summary is no turn of the run"
            )


def _check_expect(
    expect: dict[str, Any],
    answer: dict[str, Any],
    seen: list[dict[str, Any]],
    frames: list[tuple[str, dict[str, Any]]],
    outcome: Outcome,
) -> None:
    if "result" in expect and not _contains(answer.get("result"), expect["result"]):
        outcome.problems.append(
            f"run answered {answer.get('result') or answer.get('error')}, expected {expect['result']}"
        )
    if "error" in expect and not _contains(answer.get("error"), expect["error"]):
        outcome.problems.append(
            f"run answered {answer.get('result') or answer.get('error')}, expected error {expect['error']}"
        )
    if expect.get("failed") and "error" not in answer:
        outcome.problems.append(f"run answered {answer.get('result')}, expected it to fail")
    data = (answer.get("error") or {}).get("data")
    for key in expect.get("errorDataKeys", []):
        if not isinstance(data, dict) or key not in data:
            outcome.problems.append(f"the run's error data has no {key!r}")
    for key in expect.get("resultKeys", []):
        if key not in (answer.get("result") or {}):
            outcome.problems.append(f"the run's result has no {key!r}")
    for expected in expect.get("answers", []):
        asked = {
            frame["id"]
            for side, frame in frames
            if side == "kedi" and frame.get("method") == expected["method"]
        }
        replies = [
            frame
            for side, frame in frames
            if side == "bridge" and frame.get("id") in asked and "method" not in frame
        ]
        if not any(_contains(reply.get("result"), expected["result"]) for reply in replies):
            outcome.problems.append(f"no {expected['method']} answer held {expected['result']}")
    if "modelRequests" in expect and len(seen) != expect["modelRequests"]:
        outcome.problems.append(f"{len(seen)} model requests, expected {expect['modelRequests']}")
    for check in expect.get("modelSaw", []):
        request = seen[check["request"] - 1] if len(seen) >= check["request"] else None
        if not any(check["contains"] in text for text in _strings(request)):
            outcome.problems.append(
                f"model request {check['request']} did not carry {check['contains']!r}"
            )
    for check in expect.get("modelMissed", []):
        request = seen[check["request"] - 1] if len(seen) >= check["request"] else None
        if any(check["contains"] in text for text in _strings(request)):
            outcome.problems.append(
                f"model request {check['request']} carried {check['contains']!r}"
            )
    for check in expect.get("modelTools", []):
        request = seen[check["request"] - 1] if len(seen) >= check["request"] else {}
        names = [tool.get("name") for tool in request.get("tools") or []]
        if len(names) < check.get("atLeast", 0):
            outcome.problems.append(
                f"model request {check['request']} offered {len(names)} tools, expected at least {check['atLeast']}"
            )
        for name in check.get("excludes", []):
            if name in names:
                outcome.problems.append(
                    f"model request {check['request']} offered {name!r} as a tool of its own"
                )
    events = [
        frame.get("params", {})
        for side, frame in frames
        if side == "bridge" and frame.get("method") == "run/event"
    ]
    for wanted in expect.get("events", []):
        if not any(_contains(event, wanted) for event in events):
            outcome.problems.append(f"no run/event held {wanted}, the bridge sent {events}")
    order = expect.get("order")
    if order:
        methods = [
            frame.get("method")
            for side, frame in frames
            if side == "bridge" and frame.get("method")
        ]
        position = 0
        for method in methods:
            if position < len(order) and method == order[position]:
                position += 1
        if position < len(order):
            outcome.problems.append(f"the bridge's requests {methods} do not follow {order}")
    for counted in expect.get("count", []):
        number = sum(
            1
            for side, frame in frames
            if side == "bridge" and frame.get("method") == counted["method"]
        )
        if number != counted["times"]:
            outcome.problems.append(
                f"{counted['method']} sent {number} times, expected {counted['times']}"
            )


def _schema_problems(frames: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Each frame that does not match its method's schema."""

    methods = SCHEMA["methods"]
    asked: dict[tuple[str, Any], str] = {}
    found: list[str] = []
    for side, frame in frames:
        method = frame.get("method")
        if method is not None:
            definition = methods.get(method)
            if definition is None:
                found.append(f"{side} sent {method}: not a protocol method")
                continue
            if not definition["direction"].startswith(side):
                found.append(f"{side} sent {method}: it goes the other way")
            if "id" in frame:
                asked[(side, frame["id"])] = method
            found += [
                f"{method} params: {problem}"
                for problem in _validate(definition["params"], frame.get("params", {}))
            ]
            continue
        asker = "kedi" if side == "bridge" else "bridge"
        method = asked.pop((asker, frame.get("id")), None)
        if method is None or "error" in frame:
            continue
        result = methods[method].get("result")
        if result is not None:
            found += [
                f"{method} result: {problem}" for problem in _validate(result, frame.get("result"))
            ]
    return found


Result = tuple[dict[str, Any], Outcome]


def check(command: list[str], paths: list[Path] | None = None) -> list[Result]:
    """Runs each fixture (all of them by default) against a fresh bridge process."""
    results = []
    for path in paths or sorted(FIXTURES.glob("*.json")):
        fixture = json.loads(path.read_text())
        results.append((fixture, run_fixture(command, fixture)))
    return results


def failed(results: list[Result]) -> bool:
    return any(outcome.problems and not outcome.skipped for _, outcome in results)


def render(results: list[Result]) -> str:
    """The results as text: a line per fixture, with each failure's problems and its fix."""
    lines = []
    for fixture, outcome in results:
        if outcome.skipped:
            lines.append(f"skip  {outcome.name}: {outcome.skipped}")
        elif outcome.problems:
            lines.append(f"FAIL  {outcome.name} [{fixture['code']}]: {fixture['failure']}")
            lines += [f"      {problem}" for problem in outcome.problems]
            lines.append(f"      help: {fixture['help']}")
        else:
            lines.append(f"pass  {outcome.name}")
    return "\n".join(lines)


def as_json(results: list[Result]) -> dict[str, Any]:
    """The results for machines (``--json``)."""
    fixtures: list[dict[str, Any]] = []
    for fixture, outcome in results:
        if outcome.skipped:
            fixtures.append({"name": outcome.name, "status": "skip", "reason": outcome.skipped})
        elif outcome.problems:
            fixtures.append(
                {
                    "name": outcome.name,
                    "status": "fail",
                    "code": fixture["code"],
                    "failure": fixture["failure"],
                    "problems": outcome.problems,
                    "help": fixture["help"],
                    "spec": fixture.get("spec"),
                }
            )
        else:
            fixtures.append({"name": outcome.name, "status": "pass"})
    return {"ok": not failed(results), "fixtures": fixtures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run conformance fixtures against a bridge.")
    parser.add_argument("--bridge", required=True, help="the command that starts the bridge")
    parser.add_argument("--json", action="store_true", help="report as JSON")
    parser.add_argument("fixtures", nargs="*", type=Path, help=f"default: all of {FIXTURES}")
    args = parser.parse_args()
    results = check(shlex.split(args.bridge), args.fixtures)
    print(json.dumps(as_json(results), indent=2) if args.json else render(results))
    return 1 if failed(results) else 0


if __name__ == "__main__":
    sys.exit(main())
