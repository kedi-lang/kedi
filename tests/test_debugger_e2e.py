from __future__ import annotations

import os
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kedi_debugger.protocol import read_message, write_message


class Client:
    def __init__(self, extra_env=None):
        environment = os.environ.copy()
        environment.update(extra_env or {})
        root = Path(__file__).resolve().parents[1]
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(root / "debugger/src"), str(root / "src"), environment.get("PYTHONPATH", "")]
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", "kedi_debugger", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        self.messages = queue.Queue()
        self.history = []
        self.seq = 0
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        try:
            while (message := read_message(self.process.stdout)) is not None:
                self.messages.put(message)
        except Exception as exc:
            self.messages.put({"error": str(exc)})

    def send(self, command, **arguments):
        self.seq += 1
        write_message(
            self.process.stdin,
            {"seq": self.seq, "type": "request", "command": command, "arguments": arguments},
        )
        return self.seq

    def wait(self, predicate):
        while True:
            message = self.messages.get(timeout=15)
            self.history.append(message)
            assert "error" not in message, message
            if predicate(message):
                return message

    def response(self, seq):
        return self.wait(lambda msg: msg.get("request_seq") == seq)

    def event(self, name):
        return self.wait(lambda msg: msg.get("event") == name)

    def request(self, command, **arguments):
        result = self.response(self.send(command, **arguments))
        assert result["success"], result
        return result.get("body", {})

    def launch(self, path, **kwargs):
        self.request("initialize", adapterID="kedi")
        launch_seq = self.send("launch", program=str(path), **kwargs)
        self.event("initialized")
        return launch_seq

    def close(self):
        if self.process.poll() is None:
            try:
                self.request("disconnect")
            finally:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.reader.join(5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()


@pytest.fixture
def client():
    value = Client()
    yield value
    value.close()


def test_real_dap_worker_steps_inspects_and_preserves_raw_stdout(client, tmp_path):
    program = tmp_path / "session.kedi"
    program.write_text(
        "[value: int] = `41`\n> show: The value is <value>.\n`__import__('os').write(1, b'raw-output\\n')`\n= `value + 1`\n",
        encoding="utf-8",
    )
    launch = client.launch(program)
    client.request("configurationDone")
    assert client.response(launch)["success"]
    stop = client.event("stopped")
    lane = stop["body"]["threadId"]
    start = len(client.history)
    step_seq = client.send("next", threadId=lane)
    assert client.response(step_seq)["success"]
    client.event("stopped")
    step_messages = client.history[start:]
    assert step_messages[0].get("request_seq") == step_seq
    assert [message.get("event") for message in step_messages[1:]] == [
        "continued",
        "stopped",
    ]
    stack = client.request("stackTrace", threadId=lane)["stackFrames"]
    assert stack[0]["line"] == 2
    assert client.request("evaluate", frameId=stack[0]["id"], expression="value")["result"] == "41"
    start = len(client.history)
    continue_seq = client.send("continue", threadId=lane)
    assert client.response(continue_seq)["success"]
    client.event("terminated")
    assert client.history[start].get("request_seq") == continue_seq
    # Output arrives on an independent pipe; an extra request drains any preceding events.
    client.request("threads")
    output = "".join(msg.get("body", {}).get("output", "") for msg in client.history)
    assert "The value is 41." in output
    assert "raw-output" in output
    assert "42" in output
    assert len([m for m in client.history if m.get("event") == "terminated"]) == 1


def test_real_dap_worker_exposes_token_usage_without_exposing_access_tokens(client, tmp_path):
    program = tmp_path / "usage.kedi"
    program.write_text(
        '[usage: dict[str, int]] = `{"input_tokens": 4096, "output_tokens": 32, '
        '"cache_read_tokens": 3072, "reasoning_tokens": 16}`\n'
        "[access_token] = synthetic-debugger-access\n"
        "= done\n",
        encoding="utf-8",
    )
    launch = client.launch(program, stopOnEntry=False, env={"MAX_TOKENS": "4096"})
    client.request("setBreakpoints", source={"path": str(program)}, breakpoints=[{"line": 3}])
    client.request("configurationDone")
    assert client.response(launch)["success"]
    lane = client.event("stopped")["body"]["threadId"]
    frame = client.request("stackTrace", threadId=lane)["stackFrames"][0]["id"]
    usage = client.request("evaluate", frameId=frame, expression="usage")
    variables = client.request("variables", variablesReference=usage["variablesReference"])
    assert {item["name"]: item["value"] for item in variables["variables"]} == {
        "input_tokens": "4096",
        "output_tokens": "32",
        "cache_read_tokens": "3072",
        "reasoning_tokens": "16",
    }
    assert client.request("evaluate", frameId=frame, expression="access_token")["result"] == (
        "<redacted>"
    )
    client.request("continue", threadId=lane)
    client.event("terminated")


def test_worker_protocol_survives_blocked_python_and_terminate(client, tmp_path):
    program = tmp_path / "blocked.kedi"
    program.write_text("`__import__('time').sleep(60)`\n", encoding="utf-8")
    launch = client.launch(program, stopOnEntry=False)
    client.request("configurationDone")
    assert client.response(launch)["success"]
    # The interpreter may be blocked, but its control reader and supervisor are independent.
    client.request("threads")
    client.request("terminate")
    client.event("terminated")


@pytest.mark.parametrize("command", ["terminate", "disconnect"])
def test_stopping_paused_worker_does_not_report_program_failure(client, tmp_path, command):
    program = tmp_path / "paused.kedi"
    program.write_text("[value: int] = `42`\n= <value>\n", encoding="utf-8")
    launch = client.launch(program)
    client.request("configurationDone")
    assert client.response(launch)["success"]
    client.event("stopped")
    client.request(command)
    client.event("terminated")
    if command == "disconnect":
        client.process.wait(timeout=10)
    client.close()
    messages = [*client.history, *list(client.messages.queue)]
    output = "".join(
        message.get("body", {}).get("output", "")
        for message in messages
        if message.get("event") == "output"
    )
    assert "Kedi debuggee failed" not in output
    assert "Exception in thread" not in output
    assert "Traceback" not in output
    assert sum(message.get("event") == "exited" for message in messages) == 1
    assert sum(message.get("event") == "terminated" for message in messages) == 1


@pytest.mark.parametrize("command", ["terminate", "disconnect"])
def test_editor_restart_sequence_reloads_saved_source_and_environment(tmp_path, command):
    program = tmp_path / "restart.kedi"
    for version in range(3):
        program.write_text(
            f"[version: int] = `{version}`\n"
            "[setting: str] = `__import__('os').environ['RUN_SETTING']`\n= <version>\n",
            encoding="utf-8",
        )
        (tmp_path / ".env").write_text(f"RUN_SETTING=revision-{version}\n", encoding="utf-8")
        session = Client()
        try:
            launch = session.launch(program, stopOnEntry=False)
            session.request(
                "setBreakpoints", source={"path": str(program)}, breakpoints=[{"line": 3}]
            )
            session.request("configurationDone")
            assert session.response(launch)["success"]
            lane = session.event("stopped")["body"]["threadId"]
            frame = session.request("stackTrace", threadId=lane)["stackFrames"][0]["id"]
            assert session.request("evaluate", frameId=frame, expression="version")[
                "result"
            ] == str(version)
            assert (
                session.request("evaluate", frameId=frame, expression="setting")["result"]
                == f"revision-{version}"
            )
            session.request(command, restart=True)
            session.event("terminated")
            if command == "disconnect":
                session.process.wait(timeout=10)
        finally:
            session.close()


def test_real_worker_completed_capture_is_visible_and_lifecycle_is_clean(client, tmp_path):
    program = tmp_path / "captures.kedi"
    program.write_text(
        "```\n"
        "from kedi import app\n"
        "from kedi.agent_adapter.adapters import PydanticAdapter\n"
        "from pydantic_ai.models.test import TestModel\n"
        "app._build_runtime_adapter = lambda *a, **kw: PydanticAdapter(\n"
        "    model=TestModel(custom_output_args={'answer': 'ready'}))\n"
        "```\n"
        ">> The status is [answer: str].\n"
        "> show: <answer>\n"
        "= done\n",
        encoding="utf-8",
    )
    client.request("initialize", adapterID="kedi", supportsProgressReporting=True)
    launch = client.send("launch", program=str(program), stopOnEntry=False)
    client.event("initialized")
    client.request("setBreakpoints", source={"path": str(program)}, breakpoints=[{"line": 10}])
    client.request("configurationDone")
    assert client.response(launch)["success"]
    lane = client.event("stopped")["body"]["threadId"]
    frame = client.request("stackTrace", threadId=lane)["stackFrames"][0]["id"]
    assert client.request("evaluate", frameId=frame, expression="answer")["result"] == "ready"
    client.request("continue", threadId=lane)
    assert client.event("exited")["body"]["exitCode"] == 0
    client.event("terminated")
    assert client.request("threads") == {"threads": []}
    messages = client.history
    starts = {
        m["body"]["threadId"]
        for m in messages
        if m.get("event") == "thread" and m["body"]["reason"] == "started"
    }
    exits = {
        m["body"]["threadId"]
        for m in messages
        if m.get("event") == "thread" and m["body"]["reason"] == "exited"
    }
    assert starts and starts == exits
    progress_starts = {
        m["body"]["progressId"] for m in messages if m.get("event") == "progressStart"
    }
    progress_ends = {m["body"]["progressId"] for m in messages if m.get("event") == "progressEnd"}
    assert progress_starts == progress_ends


def test_forced_stop_ends_capture_progress_and_thread_activity(client, tmp_path):
    program = tmp_path / "waiting.kedi"
    program.write_text(
        "[ready: bool] = `True`\n"
        "```\n"
        "from kedi.debugging import waiting_for_capture\n"
        "import time\n"
        "with waiting_for_capture('answer'):\n"
        "    time.sleep(60)\n"
        "```\n",
        encoding="utf-8",
    )
    client.request("initialize", adapterID="kedi", supportsProgressReporting=True)
    launch = client.send("launch", program=str(program), stopOnEntry=False)
    client.event("initialized")
    client.request("configurationDone")
    assert client.response(launch)["success"]
    progress = client.event("progressStart")["body"]["progressId"]
    client.request("terminate")
    client.event("terminated")
    assert any(
        m.get("event") == "progressEnd" and m["body"]["progressId"] == progress
        for m in client.history
    )
    assert any(
        m.get("event") == "thread" and m["body"]["reason"] == "exited" for m in client.history
    )


def test_real_worker_steps_through_procedure_with_typed_model_capture(client, tmp_path):
    program = tmp_path / "typed-capture.kedi"
    lines = [
        "```",
        "from kedi import app",
        "from kedi.agent_adapter.adapters import PydanticAdapter",
        "from pydantic_ai.models.test import TestModel",
        "app._build_runtime_adapter = lambda *a, **kw: PydanticAdapter(",
        "    model=TestModel(custom_output_args={",
        "        'report': {'city': 'Stockholm', 'population': 42}}))",
        "```",
        "~Report(city: str, population: int)",
        "@summarize(value: Report) -> str:",
        "  = <`value.city`> with <`value.population`> residents.",
        ">> The city report is [report: Report].",
        "[summary: str] = `summarize(report)`",
        "> show: <summary>",
        "= done",
    ]
    program.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launch = client.launch(program, stopOnEntry=False)
    client.request("setExceptionBreakpoints", filters=[])
    client.request(
        "setBreakpoints",
        source={"path": str(program)},
        breakpoints=[{"line": lines.index("[summary: str] = `summarize(report)`") + 1}],
    )
    client.request("configurationDone")
    assert client.response(launch)["success"]
    lane = client.event("stopped")["body"]["threadId"]
    client.request("stepIn", threadId=lane)
    stop = client.event("stopped")
    assert stop["body"]["reason"] == "step"
    assert stop["body"]["description"] == "statement"
    frame = client.request("stackTrace", threadId=lane)["stackFrames"][0]
    assert frame["name"] == "summarize"
    scopes = client.request("scopes", frameId=frame["id"])["scopes"]
    for scope in scopes:
        assert scope["variablesReference"] > 0
        client.request("variables", variablesReference=scope["variablesReference"])
    assert (
        client.request("evaluate", frameId=frame["id"], expression="value.city")["result"]
        == "Stockholm"
    )
    assert (
        client.request("evaluate", frameId=frame["id"], expression="value.population")["result"]
        == "42"
    )
    client.request("stepOut", threadId=lane)
    assert client.event("stopped")["body"]["reason"] == "step"
    parent = client.request("stackTrace", threadId=lane)["stackFrames"][0]
    assert parent["line"] == lines.index("> show: <summary>") + 1
    assert (
        client.request("evaluate", frameId=parent["id"], expression="summary")["result"]
        == "Stockholm with 42 residents."
    )
    assert not client.response(client.send("scopes", frameId=frame["id"]))["success"]
    client.request("continue", threadId=lane)
    assert client.event("exited")["body"]["exitCode"] == 0
    client.event("terminated")
    assert client.request("threads") == {"threads": []}


@pytest.mark.parametrize("exit_value,expected", [("None", 0), ("0", 0), ("7", 7), ('"Stopped"', 1)])
def test_program_system_exit_preserves_exit_status(client, tmp_path, exit_value, expected):
    program = tmp_path / "exit.kedi"
    program.write_text(
        f"`__import__('sys').exit({exit_value})`\n> show: unreachable\n", encoding="utf-8"
    )
    launch = client.launch(program, stopOnEntry=False)
    client.request("configurationDone")
    assert client.response(launch)["success"]
    assert client.event("exited")["body"]["exitCode"] == expected
    client.event("terminated")
    client.close()
    messages = [*client.history, *list(client.messages.queue)]
    output = "".join(
        message.get("body", {}).get("output", "")
        for message in messages
        if message.get("event") == "output"
    )
    assert "unreachable" not in output
    assert "Kedi debuggee failed" not in output
    if expected == 1:
        assert "Stopped" in output


def test_breakpoint_source_exception_and_no_eval_execution(client, tmp_path):
    program = tmp_path / "exception.kedi"
    program.write_text("[value: int] = `8`\n= `value / 0`\n", encoding="utf-8")
    launch = client.launch(program, stopOnEntry=False)
    configured = client.request(
        "setBreakpoints", source={"path": str(program)}, breakpoints=[{"line": 2}]
    )
    assert configured["breakpoints"][0]["verified"]
    client.request("configurationDone")
    client.response(launch)
    lane = client.event("stopped")["body"]["threadId"]
    frame = client.request("stackTrace", threadId=lane)["stackFrames"][0]
    result = client.response(
        client.send("evaluate", frameId=frame["id"], expression="__import__('os')")
    )
    assert not result["success"]
    client.request("continue", threadId=lane)
    assert client.event("stopped")["body"]["reason"] == "exception"
    assert client.request("exceptionInfo", threadId=lane)["exceptionId"] == "ZeroDivisionError"
    client.request("continue", threadId=lane)
    client.event("terminated")


def test_changed_source_before_configuration_is_rejected(client, tmp_path):
    program = tmp_path / "changed.kedi"
    program.write_text("= original\n", encoding="utf-8")
    client.launch(program)
    program.write_text("= changed\n", encoding="utf-8")
    response = client.response(client.send("configurationDone"))
    assert not response["success"]
    assert "changed" in response["message"].lower()
    assert not any(message.get("event") == "stopped" for message in client.history)


def test_source_response_masks_worker_secret_without_losing_lines(client, tmp_path):
    secret = "synthetic-debugger-token-4820"
    program = tmp_path / "source.kedi"
    text = f"# {secret}\n= done\n"
    program.write_text(text, encoding="utf-8")
    client.launch(program, env={"EXAMPLE_API_KEY": secret})
    result = client.request("source", source={"path": str(program)})["content"]
    assert secret not in result
    assert len(result) == len(text)
    assert result.splitlines()[1] == "= done"


@pytest.mark.parametrize("model_variable", ["KEDI_ADAPTER_MODEL", "MODEL_NAME"])
@pytest.mark.parametrize("override", [False, True])
def test_dotenv_configures_model_before_first_request(
    tmp_path, monkeypatch, model_variable, override
):
    for name in (
        "PYTHON_DOTENV_DISABLED",
        "KEDI_ADAPTER",
        "KEDI_ADAPTER_MODEL",
        "MODEL_NAME",
        "GROQ_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        f"KEDI_ADAPTER=pydantic\n{model_variable}=openai:from-project\n"
        "GROQ_API_KEY=synthetic-project-credential\n",
        encoding="utf-8",
    )
    expected = "openai:launch-override" if override else "openai:from-project"
    program = tmp_path / "model.kedi"
    program.write_text(
        "```\n"
        "import os\n"
        "import kedi.app as app\n"
        "from kedi.agent_adapter.adapters import PydanticAdapter\n"
        "from kedi.settings import settings\n"
        "from pydantic_ai.models.test import TestModel\n"
        "assert os.environ['GROQ_API_KEY'] == 'synthetic-project-credential'\n"
        "assert settings.groq_api_key == 'synthetic-project-credential'\n"
        "def offline_adapter(adapter, model, **kwargs):\n"
        "    assert adapter == 'pydantic'\n"
        f"    assert model == {expected!r}\n"
        "    return PydanticAdapter(model=TestModel(custom_output_args={'answer': 'ready'}))\n"
        "app._build_runtime_adapter = offline_adapter\n"
        "```\n"
        ">> The environment check is [answer].\n"
        "= <answer>\n",
        encoding="utf-8",
    )
    client = Client()
    try:
        kwargs = {"model": expected} if override else {}
        launch = client.launch(program, stopOnEntry=False, **kwargs)
        client.request("setExceptionBreakpoints", filters=[])
        client.request("configurationDone")
        assert client.response(launch)["success"]
        code = client.event("exited")["body"]["exitCode"]
        client.event("terminated")
        client.request("threads")
        output = "".join(msg.get("body", {}).get("output", "") for msg in client.history)
        assert code == 0, output
        assert "ready" in output
    finally:
        client.close()


def test_dotenv_secrets_are_redacted_and_removed_keys_stay_absent(tmp_path, monkeypatch):
    for name in ("PYTHON_DOTENV_DISABLED", "DEBUG_TEST_API_KEY", "DEBUG_REMOVE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    secret = "synthetic-file-only-secret-6841"
    (tmp_path / ".env").write_text(
        f"DEBUG_TEST_API_KEY={secret}\nDEBUG_REMOVE_API_KEY=must-not-return\n", encoding="utf-8"
    )
    program = tmp_path / "secrets.kedi"
    program.write_text(
        f"# {secret}\n"
        "```\n"
        "import os\n"
        "import dotenv\n"
        "dotenv.load_dotenv()\n"
        "assert 'DEBUG_REMOVE_API_KEY' not in os.environ\n"
        "print(os.environ['DEBUG_TEST_API_KEY'])\n"
        "```\n"
        "= dotenv-complete\n",
        encoding="utf-8",
    )
    client = Client()
    try:
        launch = client.launch(program, stopOnEntry=False, env={"DEBUG_REMOVE_API_KEY": None})
        source = client.request("source", source={"path": str(program)})["content"]
        assert secret not in source
        client.request("setExceptionBreakpoints", filters=[])
        client.request("configurationDone")
        assert client.response(launch)["success"]
        assert client.event("exited")["body"]["exitCode"] == 0
        client.event("terminated")
        client.request("threads")
        output = "".join(msg.get("body", {}).get("output", "") for msg in client.history)
        assert "[REDACTED]" in output
        assert "dotenv-complete" in output
        assert secret not in json.dumps(client.history)
    finally:
        client.close()


@pytest.mark.parametrize("path", [None, False, 7, [], {}, "", "bad\x00path"])
def test_invalid_source_request_does_not_kill_debuggee(client, tmp_path, path):
    program = tmp_path / "source.kedi"
    program.write_text("= done\n", encoding="utf-8")
    client.launch(program)
    response = client.response(client.send("source", source={"path": path}))
    assert not response["success"]
    assert "Source path" in response["message"]
    assert client.request("threads") == {"threads": []}
    assert not any(message.get("event") == "terminated" for message in client.history)


def test_imported_procedure_breakpoint_uses_its_real_source_and_scope(client, tmp_path):
    helper = tmp_path / "helper.kedi"
    helper.write_text(
        "@double(value: int) -> int:\n  = `value * 2`\n> export:\n  double\n", encoding="utf-8"
    )
    program = tmp_path / "main.kedi"
    program.write_text("> import: helper\n= `double(3)`\n", encoding="utf-8")
    launch = client.launch(program, stopOnEntry=False)
    configured = client.request(
        "setBreakpoints", source={"path": str(helper)}, breakpoints=[{"line": 2}]
    )
    assert configured["breakpoints"][0]["verified"]
    client.request("configurationDone")
    client.response(launch)
    lane = client.event("stopped")["body"]["threadId"]
    frame = client.request("stackTrace", threadId=lane)["stackFrames"][0]
    assert frame["source"]["path"] == str(helper)
    assert frame["line"] == 2
    assert client.request("evaluate", frameId=frame["id"], expression="value")["result"] == "3"
    client.request("continue", threadId=lane)
    client.event("terminated")


def test_runtime_import_failure_still_reports_exit_and_termination(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        "import builtins, sys\n"
        "if 'kedi_debugger.worker' in sys.orig_argv:\n"
        "    original = builtins.__import__\n"
        "    def guarded(name, *args, **kwargs):\n"
        "        if name == 'kedi.app':\n"
        "            raise ImportError('Synthetic missing runtime dependency')\n"
        "        return original(name, *args, **kwargs)\n"
        "    builtins.__import__ = guarded\n",
        encoding="utf-8",
    )
    program = tmp_path / "main.kedi"
    program.write_text("= done\n", encoding="utf-8")
    client = Client({"PYTHONPATH": str(tmp_path)})
    try:
        launch = client.launch(program, stopOnEntry=False)
        client.request("configurationDone")
        client.response(launch)
        assert client.event("exited")["body"]["exitCode"] == 1
        client.event("terminated")
        client.request("threads")
        output = "".join(message.get("body", {}).get("output", "") for message in client.history)
        assert "ImportError" in output
    finally:
        client.close()
