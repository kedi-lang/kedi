from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path

import pytest

from kedi.debugging import DebugEvent, observe_execution
from kedi.lang import compile_program, parse_program
from kedi_debugger.controller import Controller
from tests.mock_adapter import MockAdapter


class Session:
    def __init__(self, tmp_path: Path, source: str, **config):
        path = tmp_path / "debug.kedi"
        path.write_text(source, encoding="utf-8")
        self.events = queue.Queue()
        self.controller = Controller({"program": str(path), **config}, self.events.put)
        self.adapter = MockAdapter()
        self.runtime = compile_program(
            parse_program(source, source_path=str(path)), adapter=self.adapter
        )
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self.run)
        self.path = str(path)

    def run(self):
        try:
            with observe_execution(self.controller):
                self.result = self.runtime.run_main()
        except BaseException as exc:
            self.error = exc
        finally:
            self.controller.finished = True

    def stopped(self):
        while True:
            event = self.events.get(timeout=5)
            if event.get("event") == "stopped":
                return event["body"]

    def stack(self, lane):
        return self.controller.request("stackTrace", {"threadId": lane})["stackFrames"]

    def finish(self):
        self.controller.close()
        if self.thread.ident is not None:
            self.thread.join(5)
            assert not self.thread.is_alive()
        asyncio.run(self.runtime.aclose())


@pytest.fixture
def session(tmp_path):
    sessions = []

    def create(source, **config):
        result = Session(tmp_path, source, **config)
        sessions.append(result)
        return result

    yield create
    for item in sessions:
        item.finish()


def test_entry_step_into_over_out_and_readonly_values(session):
    run = session(
        "@double(value: int) -> int:\n  [result: int] = `value * 2`\n  = `result`\n[x: int] = `3`\n[y: int] = `double(x)`\n= `y`\n"
    )
    run.thread.start()
    stop = run.stopped()
    lane = stop["threadId"]
    # Top-level procedure registration may itself be a visible source boundary.
    while run.stack(lane)[0]["line"] != 5:
        run.controller.request("next", {"threadId": lane})
        run.stopped()
    previous = run.stack(lane)[0]["id"]
    result = run.controller.request("evaluate", {"frameId": previous, "expression": "x"})
    assert result["result"] == "3"
    with pytest.raises(ValueError):
        run.controller.request("evaluate", {"frameId": previous, "expression": "print(x)"})
    run.controller.request("stepIn", {"threadId": lane})
    run.stopped()
    stack = run.stack(lane)
    assert [frame["name"] for frame in stack] == ["double", "<program>"]
    assert stack[0]["line"] == 2
    with pytest.raises(ValueError, match="Expired"):
        run.controller.request("scopes", {"frameId": previous})
    run.controller.request("stepOut", {"threadId": lane})
    run.stopped()
    assert run.stack(lane)[0]["line"] == 6
    run.controller.request("continue", {"threadId": lane})
    run.thread.join(5)
    assert run.error is None
    assert run.result == 6
    assert run.stack(lane)  # A bounded final snapshot remains available.
    with pytest.raises(ValueError, match="finished"):
        run.controller.request("next", {"threadId": lane})


@pytest.mark.parametrize("command", ["continue", "next", "stepIn", "stepOut", "pause"])
def test_control_acknowledges_before_execution_changes(session, command):
    run = session("[value: int] = `1`\n= done\n")
    run.thread.start()
    lane = run.stopped()["threadId"]
    frames = run.stack(lane)
    acknowledged = []

    def acknowledge(body):
        assert run.stack(lane) == frames
        assert all(item.paused for item in run.controller.lanes.values())
        assert run.events.empty()
        acknowledged.append(body)

    result = run.controller.request(command, {"threadId": lane}, acknowledge=acknowledge)
    assert acknowledged == [result]


def test_invalid_control_does_not_acknowledge_success(session):
    run = session("= done\n")
    run.thread.start()
    lane = run.stopped()["threadId"]
    acknowledged = []
    with pytest.raises(ValueError, match="singleThread"):
        run.controller.request(
            "next", {"threadId": lane, "singleThread": "false"}, acknowledge=acknowledged.append
        )
    assert acknowledged == []
    assert run.stack(lane)


def test_failed_acknowledgement_does_not_resume_execution(session):
    run = session("= done\n")
    run.thread.start()
    lane = run.stopped()["threadId"]

    def disconnected(body):
        raise BrokenPipeError("Disconnected test client")

    with pytest.raises(BrokenPipeError):
        run.controller.request("next", {"threadId": lane}, acknowledge=disconnected)
    assert all(item.paused for item in run.controller.lanes.values())
    assert run.stack(lane)


def test_loop_breakpoints_repeat_and_replacement_clears(session):
    run = session(
        "> loop [item]: `range(3)`:\n  [value: int] = `item`\n= done\n", stopOnEntry=False
    )
    result = run.controller.request(
        "setBreakpoints", {"source": {"path": run.path}, "breakpoints": [{"line": 2}, {"line": 99}]}
    )
    assert [b["verified"] for b in result["breakpoints"]] == [True, False]
    run.thread.start()
    for _ in range(3):
        stop = run.stopped()
        assert stop["reason"] == "breakpoint"
        run.controller.request("continue", {"threadId": stop["threadId"]})
    run.thread.join(5)
    assert run.result == "done"
    run.controller.request("setBreakpoints", {"source": {"path": run.path}, "breakpoints": []})
    assert run.controller.breakpoints[run.path] == set()


def test_exception_location_and_info(session):
    run = session("[value: int] = `1 / 0`\n", stopOnEntry=False)
    run.thread.start()
    stop = run.stopped()
    assert stop["reason"] == "exception"
    lane = stop["threadId"]
    assert run.stack(lane)[0]["line"] == 1
    assert (
        run.controller.request("exceptionInfo", {"threadId": lane})["exceptionId"]
        == "ZeroDivisionError"
    )
    run.controller.request("continue", {"threadId": lane})
    run.thread.join(5)
    assert run.error is not None
    assert (
        run.controller.request("exceptionInfo", {"threadId": lane})["exceptionId"]
        == "ZeroDivisionError"
    )


def test_final_snapshot_observes_drained_not_forced_promise(session):
    run = session(">> The answer is [answer].\n= done\n", stopOnEntry=False)
    run.thread.start()
    run.thread.join(5)
    assert not run.thread.is_alive()
    assert run.error is None
    assert run.controller.request("threads", {}) == {"threads": []}
    lane = next(iter(run.controller.lanes.values())).id
    frame = run.stack(lane)[0]["id"]
    value = run.controller.request("evaluate", {"frameId": frame, "expression": "answer"})
    assert "pending" not in value["result"]
    assert len(run.adapter.request_calls) == 1


def test_source_edits_cannot_resume_a_stale_snapshot(session):
    source = "[value: int] = `7`\n= done\n"
    run = session(source)
    run.thread.start()
    lane = run.stopped()["threadId"]
    Path(run.path).write_text(source.replace("7", "9"), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)changed"):
        run.controller.request("continue", {"threadId": lane})
    assert any(
        event.get("event") == "output" and "restart required" in event["body"]["output"]
        for event in list(run.events.queue)
    )
    assert run.controller._thread(lane).paused
    Path(run.path).write_text(source, encoding="utf-8")
    run.controller.request("continue", {"threadId": lane})
    run.thread.join(5)
    assert run.error is None


def test_model_filters_preserve_call_count_and_prompt(session):
    run = session(">> The answer is [answer].\n= done\n", stopOnEntry=False)
    run.controller.request("setExceptionBreakpoints", {"filters": ["model_input", "model_result"]})
    run.thread.start()
    before = run.stopped()
    assert before["description"] == "model_input"
    assert not run.adapter.request_calls
    run.controller.request("continue", {"threadId": before["threadId"]})
    after = run.stopped()
    assert after["description"] == "model_result"
    assert len(run.adapter.request_calls) == 1
    scopes = run.controller.request("scopes", {"frameId": run.stack(after["threadId"])[0]["id"]})[
        "scopes"
    ]
    assert {s["name"] for s in scopes} >= {
        "Locals",
        "Globals",
        "request",
        "model_input",
        "model_result",
    }
    run.controller.request("continue", {"threadId": after["threadId"]})
    run.thread.join(5)
    assert run.error is None
    assert len(run.adapter.request_calls) == 1


def test_snapshot_stays_stable_and_scopes_are_bounded(session):
    run = session("[values: list[int]] = `[1, 2]`\n= done\n", stopOnEntry=False)
    run.controller.request(
        "setBreakpoints", {"source": {"path": run.path}, "breakpoints": [{"line": 2}]}
    )
    run.thread.start()
    stop = run.stopped()
    frame = run.stack(stop["threadId"])[0]["id"]
    first = run.controller.request("evaluate", {"frameId": frame, "expression": "values[0]"})
    assert first["result"] == "1"
    run.runtime._globals_env["values"].append(99)
    with pytest.raises(ValueError, match="bounded snapshot"):
        run.controller.request("evaluate", {"frameId": frame, "expression": "values[2]"})
    run.controller.request("continue", {"threadId": stop["threadId"]})
    with pytest.raises(ValueError, match="current stopped frame"):
        run.controller.request("evaluate", {"frameId": frame, "expression": "values"})


def test_zero_based_coordinates_and_uri_sources(session):
    run = session(
        "[value: int] = `7`\n= <value>\n",
        __initialize={"linesStartAt1": False, "columnsStartAt1": False, "pathFormat": "uri"},
    )
    run.thread.start()
    lane = run.stopped()["threadId"]
    frame = run.stack(lane)[0]
    assert (frame["line"], frame["column"]) == (0, 0)
    assert frame["source"]["path"].startswith("file:")
    assert "value" in run.controller.request("source", {"source": frame["source"]})["content"]
    run.controller.request("continue", {"threadId": lane})


def test_request_rejects_unsupported_features(session):
    run = session("= done\n")
    for command in ("attach", "setVariable", "restartFrame", "stepBack"):
        with pytest.raises(ValueError, match="Unsupported"):
            run.controller.request(command, {})


def test_lexical_closure_and_global_watch_use_snapshot_values(session):
    run = session(
        "[global_value: int] = `4`\n"
        "@outer(value: int) -> int:\n"
        "  @inner() -> int:\n"
        "    = `value + global_value`\n"
        "  = `inner()`\n"
        "= `outer(3)`\n",
        stopOnEntry=False,
    )
    run.controller.request(
        "setBreakpoints", {"source": {"path": run.path}, "breakpoints": [{"line": 4}]}
    )
    run.thread.start()
    lane = run.stopped()["threadId"]
    frame = run.stack(lane)[0]["id"]
    assert (
        run.controller.request("evaluate", {"frameId": frame, "expression": "value"})["result"]
        == "3"
    )
    assert (
        run.controller.request("evaluate", {"frameId": frame, "expression": "global_value"})[
            "result"
        ]
        == "4"
    )
    run.controller.request("continue", {"threadId": lane})
    run.thread.join(5)
    assert run.error is None
    assert run.result == 7


def test_actual_compiled_source_must_match_launch_snapshot(session):
    run = session("= original\n", stopOnEntry=False)
    with observe_execution(run.controller), pytest.raises(ValueError, match="(?i)changed"):
        compile_program(
            parse_program("= replacement\n", source_path=run.path), adapter=MockAdapter()
        )


def test_exception_inspection_does_not_invoke_metaclass(session):
    run = session("= done\n", stopOnEntry=False)

    class Meta(type):
        def __getattribute__(self, name):
            raise AssertionError("Inspector executed a metaclass")

    class Failure(Exception, metaclass=Meta):
        pass

    lane = run.controller._lane()
    event = DebugEvent("exception", run.runtime, (), {}, None, False, Failure("failed"))
    run.controller._snapshot(lane, event)
    assert (
        run.controller.request("exceptionInfo", {"threadId": lane.id})["exceptionId"] == "Failure"
    )


@pytest.mark.parametrize("command", ["next", "stepIn", "stepOut"])
@pytest.mark.parametrize("single_thread", [False, True])
def test_stepping_resumes_dependency_lanes_unless_single_thread_requested(
    session, command, single_thread
):
    run = session(">> The answer is [answer].\n= `answer`\n", stopOnEntry=False)
    run.controller.request("setExceptionBreakpoints", {"filters": ["model_input"]})
    run.controller.request(
        "setBreakpoints", {"source": {"path": run.path}, "breakpoints": [{"line": 2}]}
    )
    run.thread.start()
    stops = [run.stopped(), run.stopped()]
    main_lane = next(stop["threadId"] for stop in stops if stop["description"] == "statement")
    model_lane = next(stop["threadId"] for stop in stops if stop["description"] == "model_input")
    arguments = {"threadId": main_lane}
    if single_thread:
        arguments["singleThread"] = True
    run.controller.request(command, arguments)
    if single_thread:
        assert run.controller._thread(model_lane).paused
        run.controller.request("continue", {"threadId": model_lane, "singleThread": True})
    run.thread.join(5)
    assert not run.thread.is_alive()
    assert run.error is None
    assert len(run.adapter.request_calls) == 1


@pytest.mark.parametrize("value", [True, None, -1, 1.5, "1", []])
def test_invalid_inspection_arguments_preserve_current_stop(session, value):
    run = session("[value: int] = `7`\n= done\n")
    run.thread.start()
    lane = run.stopped()["threadId"]
    frame = run.stack(lane)[0]["id"]
    for command, args in (
        ("scopes", {"frameId": value}),
        ("variables", {"variablesReference": value}),
        ("stackTrace", {"threadId": lane, "startFrame": value}),
        ("stackTrace", {"threadId": value}),
    ):
        with pytest.raises(ValueError, match="integer"):
            run.controller.request(command, args)
    assert run.controller.request("scopes", {"frameId": frame})["scopes"]
    run.controller.request("continue", {"threadId": lane})
    run.thread.join(5)
    assert run.error is None


def test_ambiguous_snapshot_dictionary_keys_are_not_misreported(session):
    run = session("[values: dict] = `{1: 'integer', '1': 'string'}`\n= done\n", stopOnEntry=False)
    run.controller.request(
        "setBreakpoints", {"source": {"path": run.path}, "breakpoints": [{"line": 2}]}
    )
    run.thread.start()
    lane = run.stopped()["threadId"]
    frame = run.stack(lane)[0]["id"]
    for expression in ("values[1]", "values['1']"):
        with pytest.raises(ValueError, match="Ambiguous"):
            run.controller.request("evaluate", {"frameId": frame, "expression": expression})
    run.controller.request("continue", {"threadId": lane})


def test_parallel_provider_exception_stops_without_reissuing_request(session, monkeypatch):
    run = session(">> The answer is [answer].\n= done\n", stopOnEntry=False)
    calls = []

    def fail(**kwargs):
        calls.append(1)
        raise RuntimeError("Synthetic provider failure")

    monkeypatch.setattr(run.adapter, "produce_sync", fail)
    run.thread.start()
    stop = run.stopped()
    assert stop["reason"] == "exception"
    lane = stop["threadId"]
    assert run.stack(lane)[0]["line"] == 1
    assert (
        run.controller.request("exceptionInfo", {"threadId": lane})["exceptionId"] == "RuntimeError"
    )
    run.controller.request("continue", {"threadId": lane})
    run.thread.join(5)
    assert not run.thread.is_alive()
    assert run.error is not None
    assert calls == [1]


def test_watch_includes_python_prelude_values(session):
    run = session("```\nprelude_value = 42\n```\n= done\n", stopOnEntry=False)
    run.controller.request(
        "setBreakpoints", {"source": {"path": run.path}, "breakpoints": [{"line": 4}]}
    )
    run.thread.start()
    lane = run.stopped()["threadId"]
    frame = run.stack(lane)[0]["id"]
    assert (
        run.controller.request("evaluate", {"frameId": frame, "expression": "prelude_value"})[
            "result"
        ]
        == "42"
    )


def test_wait_lifecycle_has_stable_thread_names_and_balanced_progress(session):
    run = session("= done\n", __initialize={"supportsProgressReporting": True})
    event = DebugEvent("waiting", run.runtime, (), {}, None, False, {"capture": "answer"})
    run.controller(event)
    before = run.controller.request("threads", {})
    run.controller(DebugEvent("wait_complete", run.runtime, (), {}, None, False))
    after = run.controller.request("threads", {})
    assert before == after
    assert "waiting" not in after["threads"][0]["name"]
    run.controller.finish()
    events = list(run.events.queue)
    assert [e["event"] for e in events] == ["thread", "progressStart", "progressEnd", "thread"]
    assert events[1]["body"]["progressId"] == events[2]["body"]["progressId"]
    assert events[-1]["body"]["reason"] == "exited"
    assert run.controller.request("threads", {}) == {"threads": []}


def test_statement_step_does_not_stop_at_unselected_model_event(session):
    run = session("= done\n", stopOnEntry=False)
    lane = run.controller._lane()
    lane.step = "stepIn"
    run.controller(DebugEvent("model_result", run.runtime, (), {}, None, True, {"answer": 42}))
    assert not lane.paused
    assert lane.step == "stepIn"


def test_continue_all_reports_all_threads_even_with_one_lane(session):
    run = session("= done\n")
    run.thread.start()
    lane = run.stopped()["threadId"]
    assert run.controller.request("continue", {"threadId": lane}) == {"allThreadsContinued": True}


def test_scalar_observation_scope_is_inspectable(session):
    run = session("= done\n", stopOnEntry=False)
    run.controller(DebugEvent("model_result", run.runtime, (), {}, None, False, "Finished"))
    lane = run.controller._lane()
    run.controller._snapshot(lane, DebugEvent("statement", run.runtime, (), {}, None, False))
    scopes = run.controller.request("scopes", {"frameId": lane.frames[0]["id"]})["scopes"]
    result = next(s for s in scopes if s["name"] == "model_result")
    values = run.controller.request(
        "variables", {"variablesReference": result["variablesReference"]}
    )
    assert values["variables"][0]["value"] == "Finished"


def test_scope_truncation_is_visible_without_hiding_retained_bindings(session):
    run = session("= done\n", stopOnEntry=False)
    lane = run.controller._lane()
    env = {f"value_{index}": index for index in range(200)}
    run.controller._snapshot(lane, DebugEvent("statement", run.runtime, (), env, None, False))
    scopes = run.controller.request("scopes", {"frameId": lane.frames[0]["id"]})["scopes"]
    local = next(scope for scope in scopes if scope["name"] == "Locals")
    values = run.controller.request(
        "variables", {"variablesReference": local["variablesReference"]}
    )["variables"]
    assert next(value for value in values if value["name"] == "value_0")["value"] == "0"
    assert any("truncated" in value["value"] for value in values)
    assert len(values) < len(env)
