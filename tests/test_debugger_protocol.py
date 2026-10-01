from __future__ import annotations

import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO
from unittest.mock import Mock

import pytest

DEBUGGER_SRC = Path(__file__).resolve().parents[1] / "debugger" / "src"
sys.path.insert(0, str(DEBUGGER_SRC))

from kedi_debugger.protocol import (  # noqa: E402
    MAX_HEADER_BYTES,
    MAX_MESSAGE_BYTES,
    ProtocolError,
    cancel_windows_read,
    read_message,
    write_message,
)
from kedi_debugger.server import DebugAdapter, _Redactor, validate_launch  # noqa: E402


def test_windows_input_cancellation_retries_the_read_start_race(monkeypatch) -> None:
    import ctypes

    kernel = Mock()
    kernel.OpenThread.return_value = 1 << 40
    kernel.CancelSynchronousIo.side_effect = [False, True]
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 1168, raising=False)
    reader = Mock(spec=threading.Thread)
    reader.native_id = 123
    reader.is_alive.return_value = True

    def joined(timeout):
        if reader.join.call_count == 2:
            reader.is_alive.return_value = False

    reader.join.side_effect = joined
    cancel_windows_read(reader, 1)
    kernel.OpenThread.assert_called_once_with(0x0001, False, 123)
    assert kernel.CancelSynchronousIo.call_count == 2
    kernel.CloseHandle.assert_called_once_with(1 << 40)


def test_windows_input_cancellation_closes_thread_handle_on_error(monkeypatch) -> None:
    import ctypes

    kernel = Mock()
    kernel.OpenThread.return_value = 42
    kernel.CancelSynchronousIo.return_value = False
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda code: OSError(code, "denied"), raising=False)
    reader = Mock(spec=threading.Thread)
    reader.native_id = 123
    reader.is_alive.return_value = True
    with pytest.raises(OSError, match="denied"):
        cancel_windows_read(reader, 1)
    kernel.CloseHandle.assert_called_once_with(42)


def frame(message: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    write_message(output, message)
    return output.getvalue()


class FragmentedInput(io.BytesIO):
    def read(self, size: int | None = -1) -> bytes:
        return super().read(min(size if size is not None else 1, 3))


class ShortWriter(io.BytesIO):
    def write(self, data: Any) -> int:
        return super().write(data[:3])


def test_fragmented_unicode_frames_and_clean_eof() -> None:
    message = {"type": "event", "body": {"text": "\u0130stanbul \u732b \U0001f642"}}
    payload = frame(message)
    header, body = payload.split(b"\r\n\r\n")
    assert int(header.split(b":")[1]) == len(body)
    stream = FragmentedInput(payload + frame({"seq": 2}))
    assert read_message(stream) == message
    assert read_message(stream) == {"seq": 2}
    assert read_message(stream) is None
    writer = ShortWriter()
    write_message(writer, message)
    assert writer.getvalue() == payload


@pytest.mark.parametrize(
    "payload",
    [
        b"Content-Length: 3\r\n",
        b"Content-Length: 3\r\n\r\n{}",
        b"X: y\r\n\r\n{}",
        b"Content-Length: -1\r\n\r\n",
        b"Content-Length: 0\r\n\r\n",
        b"Content-Length: +2\r\n\r\n{}",
        b"Content-Length: 2\r\ncontent-length: 2\r\n\r\n{}",
        b"Content-Length: 999999999999999999999999\r\n\r\n",
        f"Content-Length: {MAX_MESSAGE_BYTES + 1}\r\n\r\n".encode(),
        b"X" * (MAX_HEADER_BYTES + 1),
        b"Bad header\r\nContent-Length: 2\r\n\r\n{}",
        b"X: \xff\r\nContent-Length: 2\r\n\r\n{}",
        b"Content-Length: 1\r\n\r\n\xff",
        b"Content-Length: 1\r\n\r\n{",
        b"Content-Length: 2\r\n\r\n[]",
        b'Content-Length: 9\r\n\r\n{"x":NaN}',
    ],
)
def test_invalid_frames_are_rejected_without_echoing_input(payload: bytes) -> None:
    with pytest.raises(ProtocolError) as error:
        read_message(io.BytesIO(payload))
    assert len(str(error.value)) < 100


def test_header_extensions_and_exact_body_limit() -> None:
    assert read_message(io.BytesIO(b"X-Test: ok\r\ncontent-length: 2\r\n\r\n{}")) == {}
    message = {"x": "x" * (MAX_MESSAGE_BYTES - 8)}
    assert read_message(io.BytesIO(frame(message))) == message
    with pytest.raises(ProtocolError, match="exceeds"):
        frame({"x": message["x"] + "x"})


@pytest.mark.parametrize("message", [{"x": float("nan")}, {"x": object()}, {"x": "\ud800"}])
def test_invalid_outgoing_json(message: dict[str, Any]) -> None:
    with pytest.raises(ProtocolError):
        frame(message)


def test_nonobject_and_nonwritable_output_are_rejected() -> None:
    invalid: Any = []
    with pytest.raises(ProtocolError, match="JSON object"):
        write_message(io.BytesIO(), invalid)

    class StalledWriter(io.BytesIO):
        def write(self, data: Any) -> int:
            return 0

    with pytest.raises(OSError, match="no write progress"):
        write_message(StalledWriter(), {"type": "event"})


def test_deep_json_is_a_protocol_error() -> None:
    body = b'{"x":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}"
    with pytest.raises(ProtocolError):
        read_message(io.BytesIO(f"Content-Length: {len(body)}\r\n\r\n".encode() + body))


@pytest.mark.parametrize(
    "body",
    [
        b'{"value":1e999}',
        b'{"value":-1e999}',
        b'{"nested":[{"value":"\\ud800"}]}',
        b'{"\\udfff":"value"}',
    ],
)
def test_decoded_json_must_remain_serializable(body: bytes) -> None:
    with pytest.raises(ProtocolError):
        read_message(io.BytesIO(f"Content-Length: {len(body)}\r\n\r\n".encode() + body))


def test_paired_surrogates_and_finite_exponents_are_valid() -> None:
    body = b'{"value":"\\ud83d\\ude42", "number":1e300}'
    message = read_message(io.BytesIO(f"Content-Length: {len(body)}\r\n\r\n".encode() + body))
    assert message == {"value": "\U0001f642", "number": 1e300}


@pytest.fixture
def program(tmp_path: Path) -> Path:
    path = tmp_path / "program.kedi"
    path.write_text("[answer: int] = `42`\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "arguments",
    [
        {"program": 7},
        {"program": "missing.kedi"},
        {"program": "bad\0.kedi"},
        {"cwd": []},
        {"cwd": ""},
        {"args": "--name=x"},
        {"args": [3]},
        {"args": ["nul\0"]},
        {"env": []},
        {"env": {"": "value"}},
        {"env": {"BAD=KEY": "value"}},
        {"env": {"KEY": 4}},
        {"env": {"KEY": "nul\0"}},
        {"stopOnEntry": 1},
        {"adapter": False},
        {"model": ""},
        {"python": "/other/python"},
    ],
)
def test_launch_rejects_invalid_process_arguments(program: Path, arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_launch({"program": str(program), **arguments})


def test_launch_paths_args_and_environment(program: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEDI_DEBUG_TEST_REMOVE", "remove")
    config, environment = validate_launch(
        {
            "program": program.name,
            "cwd": str(program.parent),
            "args": ["--name=\u732b", "argument with spaces"],
            "env": {"KEDI_DEBUG_TEST_REMOVE": None, "KEDI_DEBUG_TEST_KEEP": "keep"},
        }
    )
    assert config["program"] == str(program)
    assert config["cwd"] == str(program.parent)
    assert config["args"] == ["--name=\u732b", "argument with spaces"]
    assert environment["KEDI_DEBUG_TEST_KEEP"] == "keep"
    assert "KEDI_DEBUG_TEST_REMOVE" not in environment
    assert os.environ["KEDI_DEBUG_TEST_REMOVE"] == "remove"
    with pytest.raises(ValueError, match="directory"):
        validate_launch({"program": str(program), "cwd": str(program.parent / "missing")})


class Client:
    def __init__(self, reader: BinaryIO, writer: BinaryIO) -> None:
        self.writer = writer
        self.messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.received: list[dict[str, Any]] = []
        self.seq = 0

        def receive() -> None:
            try:
                while message := read_message(reader):
                    self.messages.put(message)
            finally:
                self.messages.put(None)

        self.thread = threading.Thread(target=receive, daemon=True)
        self.thread.start()

    def send(self, command: str, **arguments: Any) -> int:
        self.seq += 1
        write_message(
            self.writer,
            {"seq": self.seq, "type": "request", "command": command, "arguments": arguments},
        )
        return self.seq

    def receive(self) -> dict[str, Any]:
        message = self.messages.get(timeout=5)
        assert message is not None, "Adapter closed before expected response"
        self.received.append(message)
        return message

    def until(self, *, event: str = "", response: int = -1) -> dict[str, Any]:
        for _ in range(200):
            message = self.receive()
            if (event and message.get("event") == event) or message.get("request_seq") == response:
                return message
        raise AssertionError("Expected message not received")


@contextmanager
def adapter_client() -> Iterator[tuple[Client, DebugAdapter, threading.Thread]]:
    request_read, request_write = os.pipe()
    output_read, output_write = os.pipe()
    with (
        os.fdopen(request_read, "rb", buffering=0) as reader,
        os.fdopen(request_write, "wb", buffering=0) as writer,
        os.fdopen(output_read, "rb", buffering=0) as client_reader,
        os.fdopen(output_write, "wb", buffering=0) as server_writer,
    ):
        adapter = DebugAdapter(reader, server_writer)
        thread = threading.Thread(target=adapter.run, daemon=True)
        thread.start()
        client = Client(client_reader, writer)
        try:
            yield client, adapter, thread
        finally:
            writer.close()
            thread.join(timeout=5)
            assert not thread.is_alive(), "Adapter leaked after client EOF"
            server_writer.close()
            client.thread.join(timeout=1)


FAKE_WORKER = r"""
import os
import signal
import subprocess
import sys
import time
from kedi_debugger.protocol import read_message, write_message

protocol = os.fdopen(os.dup(1), "wb", buffering=0)
control = os.fdopen(os.dup(0), "rb", buffering=0)
os.dup2(2, 1)
null = os.open(os.devnull, os.O_RDONLY)
os.dup2(null, 0)
os.close(null)

launch = read_message(control)
config = launch["arguments"]
mode = config.get("testMode", "normal")
def send(message):
    write_message(protocol, message)
def response(request, body=None):
    send({"type": "response", "request_seq": request["seq"], "command": request["command"],
          "success": True, "body": body or {}})

if mode == "crash":
    os.write(2, b"startup failed\n")
    sys.exit(7)
if mode == "refuse":
    send({"type": "response", "request_seq": launch["seq"], "command": "__launch",
          "success": False, "message": "runtime missing"})
    sys.exit(1)
if mode == "silent":
    time.sleep(60)
if mode == "close_control":
    control.close()
response(launch)
if mode == "bad_worker":
    send(config["workerMessage"])
if mode == "bad_framing":
    os.write(protocol.fileno(), b"Content-Length: 1\r\n\r\n{")
if mode == "close_control":
    time.sleep(60)
if mode == "output":
    secret = os.environ["KEDI_TEST_SECRET"]
    os.write(1, b"raw stdout remains isolated\n")
    os.write(2, ("prefix:" + secret[:5]).encode())
    time.sleep(0.1)
    os.write(2, (secret[5:] + ":suffix\n").encode())
    sys.exit(0)
if mode == "small_output":
    os.write(1, b"ok\n")
if mode == "tree":
    descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    send({"type": "event", "event": "output", "body": {"output": str(descendant.pid)}})
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
while request := read_message(control):
    command = request["command"]
    if command == "configurationDone":
        if mode == "config_failure":
            mode = "normal"
            send({"type": "response", "request_seq": request["seq"], "command": command,
                  "success": False, "message": "configuration rejected"})
            continue
        response(request)
        send({"type": "event", "event": "stopped", "body": {"reason": "entry"}})
    elif command == "threads":
        response(request, {"threads": [{"id": os.getpid(), "name": "worker"}]})
    elif command == "evaluate":
        if mode == "block":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            send({"type": "event", "event": "output", "body": {"output": "blocking"}})
            time.sleep(60)
        response(request, {"result": "42", "variablesReference": 0})
    elif command == "continue":
        response(request)
        for _ in range(2):
            send({"type": "event", "event": "exited", "body": {"exitCode": 0}})
            send({"type": "event", "event": "terminated"})
    else:
        response(request)
"""


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    original = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def start(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        if command == [sys.executable, "-m", "kedi_debugger.worker"]:
            command = [sys.executable, "-u", "-c", FAKE_WORKER]
            kwargs["env"]["PYTHONPATH"] = str(DEBUGGER_SRC)
        process = original(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start)
    return processes


def configure(client: Client, program: Path, **arguments: Any) -> None:
    initialize = client.send("initialize", linesStartAt1=False)
    assert client.until(response=initialize)["success"]
    launch = client.send("launch", program=str(program), **arguments)
    client.until(event="initialized")
    assert not any(message.get("request_seq") == launch for message in client.received)
    config = client.send("configurationDone")
    assert client.until(response=config)["success"]
    assert client.receive()["request_seq"] == launch
    client.until(event="stopped")


def test_initialize_and_disconnect_without_runtime_dependencies() -> None:
    environment = {**os.environ, "PYTHONPATH": str(DEBUGGER_SRC)}
    initialize = {"seq": 1, "type": "request", "command": "initialize", "arguments": {}}
    disconnect = {"seq": 2, "type": "request", "command": "disconnect"}
    result = subprocess.run(
        [sys.executable, "-S", "-m", "kedi_debugger", "--stdio"],
        input=frame(initialize) + frame(disconnect),
        capture_output=True,
        env=environment,
        timeout=5,
        check=True,
    )
    stream = io.BytesIO(result.stdout)
    first = read_message(stream)
    assert first is not None and first["body"]["supportsConfigurationDoneRequest"]
    assert "supportsSetVariable" not in first["body"]
    second = read_message(stream)
    assert second is not None and second["request_seq"] == 2 and second["success"]
    assert read_message(stream) is None
    assert result.stderr == b""


def test_configuration_order_and_postrun_inspection(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    with adapter_client() as (client, _, thread):
        configure(client, program)
        continued = client.send("continue", threadId=1)
        assert client.until(response=continued)["success"]
        client.until(event="terminated")
        evaluate = client.send("evaluate", expression="answer")
        assert client.until(response=evaluate)["body"]["result"] == "42"
        disconnect = client.send("disconnect")
        assert client.until(response=disconnect)["success"]
        thread.join(3)
        assert not thread.is_alive()
        assert fake_worker[0].poll() is not None
        assert sum(message.get("event") == "exited" for message in client.received) == 1
        assert sum(message.get("event") == "terminated" for message in client.received) == 1
        seqs = [message["seq"] for message in client.received]
        assert seqs == list(range(1, len(seqs) + 1))


@pytest.mark.parametrize("close", ["eof", "disconnect", "terminate"])
def test_blocked_worker_is_killed_on_client_loss_or_stop(
    program: Path, fake_worker: list[subprocess.Popen[bytes]], close: str
) -> None:
    with adapter_client() as (client, _, thread):
        configure(client, program, testMode="block")
        client.send("evaluate", expression="blocked")
        client.until(event="output")
        if close == "eof":
            client.writer.close()
        else:
            seq = client.send(close)
            assert client.until(response=seq)["success"]
            client.until(event="terminated")
            if close == "terminate":
                client.send("disconnect")
        thread.join(3)
        assert not thread.is_alive()
        assert fake_worker[0].poll() is not None


def test_stderr_raw_stdout_isolation_and_split_secret_redaction(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    secret = "secret-value-across-read-boundary"
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        client.send(
            "launch", program=str(program), testMode="output", env={"KEDI_TEST_SECRET": secret}
        )
        client.until(event="terminated")
        output = "".join(
            message["body"]["output"]
            for message in client.received
            if message.get("event") == "output"
        )
        assert "raw stdout remains isolated" in output
        assert "prefix:[REDACTED]:suffix" in output
        assert secret not in json.dumps(client.received)


def test_short_console_output_is_not_delayed_by_long_environment_secret(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        client.send(
            "launch",
            program=str(program),
            testMode="small_output",
            env={"KEDI_TEST_SECRET": "very-long-secret" * 100},
        )
        output = client.until(event="output")
        assert output["body"] == {"category": "console", "output": "ok\n"}
        assert fake_worker[0].poll() is None


def test_streaming_redaction_preserves_only_potential_secret_suffixes() -> None:
    assert _Redactor([{}]).feed("ordinary output\n") == "ordinary output\n"
    redactor = _Redactor([{"API_TOKEN": "abcdef"}])
    assert redactor.feed("ordinary output\n") == "ordinary output\n"
    assert redactor.feed("value: abc") == "value: "
    assert redactor.feed("def\n") == "[REDACTED]\n"
    assert redactor.feed("other: ab") == "other: "
    assert redactor.feed("X\n") == "abX\n"
    assert redactor.feed("ab") == ""
    assert redactor.feed("", final=True) == "ab"


@pytest.mark.parametrize(
    "secrets",
    [
        {"FIRST_TOKEN": "abc", "SECOND_TOKEN": "abcdef"},
        {"FIRST_TOKEN": "abc", "SECOND_TOKEN": "cdef"},
        {"FIRST_TOKEN": "abc", "SECOND_TOKEN": "bcde", "THIRD_TOKEN": "ef"},
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 3, 6, 20])
def test_streaming_redaction_covers_overlaps_at_every_chunk_boundary(
    secrets: dict[str, str], chunk_size: int
) -> None:
    redactor = _Redactor([secrets])
    text = "start:abcdef:stop\n"
    result = "".join(
        redactor.feed(text[start : start + chunk_size]) for start in range(0, len(text), chunk_size)
    ) + redactor.feed("", final=True)
    assert result == "start:[REDACTED]:stop\n"


@pytest.mark.parametrize("name", ["PASSWD", "PASSPHRASE", "DATABASE_URL", "DSN"])
def test_console_redaction_recognizes_credential_environment_names(name: str) -> None:
    redactor = _Redactor([{name: "private-credential"}])
    assert redactor.feed("private-credential", final=True) == "[REDACTED]"


def test_console_redaction_matches_non_utf8_environment_bytes() -> None:
    secret = b"\xff-private-credential"
    redactor = _Redactor([{"API_TOKEN": secret.decode("utf-8", "surrogateescape")}])
    assert redactor.feed(secret.decode("utf-8", "replace"), final=True) == "[REDACTED]"


def test_console_redaction_does_not_treat_token_limits_as_credentials() -> None:
    redactor = _Redactor(
        [
            {
                "MAX_TOKENS": "4096",
                "MAX_OUTPUT_TOKENS": "2048",
                "ACCESS_TOKEN": "private-credential",
                "TOKEN_USAGE_SECRET": "another-credential",
            }
        ]
    )
    assert redactor.feed("4096 / 2048 private-credential another-credential", final=True) == (
        "4096 / 2048 [REDACTED] [REDACTED]"
    )


@pytest.mark.parametrize(
    "environment",
    [
        {"API_TOKEN": "x" * 8193},
        {f"TOKEN_{index}": str(index) * 8192 for index in range(9)},
        {f"TOKEN_{index}": str(index) for index in range(257)},
    ],
    ids=["length", "total-size", "count"],
)
def test_console_secret_scan_overflow_fails_closed(environment: dict[str, str]) -> None:
    redactor = _Redactor([environment])
    assert redactor.feed("do-not-disclose\n") == "[REDACTED]"
    assert redactor.feed("", final=True) == ""


@pytest.mark.parametrize("mode", ["crash", "refuse"])
def test_startup_failure_resolves_launch_and_terminal_events(
    program: Path, fake_worker: list[subprocess.Popen[bytes]], mode: str
) -> None:
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        launch = client.send("launch", program=str(program), testMode=mode)
        client.until(event="terminated")
        response = next(
            message for message in client.received if message.get("request_seq") == launch
        )
        assert not response["success"]
        assert sum(message.get("event") == "terminated" for message in client.received) == 1


def test_invalid_requests_and_configuration_states(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    with adapter_client() as (client, _, _):
        seq = client.send("threads")
        assert not client.until(response=seq)["success"]
        seq = client.send("initialize", linesStartAt1="bad")
        assert not client.until(response=seq)["success"]
        configure(client, program)
        for command, arguments in [
            ("initialize", {}),
            ("launch", {"program": str(program)}),
            ("attach", {}),
            ("configurationDone", {}),
            ("disconnect", {"terminateDebuggee": False}),
        ]:
            seq = client.send(command, **arguments)
            assert not client.until(response=seq)["success"]


@pytest.mark.parametrize("command", ["terminate", "disconnect"])
@pytest.mark.parametrize(
    "arguments",
    [{"restart": 0}, {"restart": None}, {"terminateDebuggee": 0}, {"terminateDebuggee": "false"}],
)
def test_invalid_shutdown_flags_do_not_terminate_the_worker(
    program: Path,
    fake_worker: list[subprocess.Popen[bytes]],
    command: str,
    arguments: dict[str, Any],
) -> None:
    with adapter_client() as (client, _, _):
        configure(client, program)
        response = client.until(response=client.send(command, **arguments))
        assert not response["success"]
        assert "boolean" in response["message"]
        assert fake_worker[0].poll() is None
        seq = client.send("threads")
        assert client.until(response=seq)["success"]


def test_worker_initialization_timeout_is_bounded(
    program: Path, fake_worker: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kedi_debugger.server._START_TIMEOUT", 0.1)
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        launch = client.send("launch", program=str(program), testMode="silent")
        assert "timed out" in client.until(response=launch)["message"]
        client.until(event="terminated")
        assert fake_worker[0].poll() is not None


@pytest.mark.parametrize(
    "message",
    [
        {"type": "response", "command": "threads", "request_seq": True, "success": True},
        {"type": "response", "command": "threads", "request_seq": 999, "success": True},
        {"type": "response", "command": "launch", "request_seq": 2, "success": True},
        {"type": "response", "command": "__launch", "request_seq": 2, "success": True},
        {"type": "event", "event": "initialized"},
        {"type": "request", "command": "unknown"},
    ],
)
def test_bad_worker_messages_fail_pending_launch_and_clean_up(
    program: Path, fake_worker: list[subprocess.Popen[bytes]], message: dict[str, Any]
) -> None:
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        launch = client.send(
            "launch", program=str(program), testMode="bad_worker", workerMessage=message
        )
        assert not client.until(response=launch)["success"]
        client.until(event="terminated")
        assert fake_worker[0].poll() is not None


@pytest.mark.parametrize("mode", ["bad_framing", "close_control"])
def test_worker_pipe_failures_do_not_hang(
    program: Path, fake_worker: list[subprocess.Popen[bytes]], mode: str
) -> None:
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        client.send("launch", program=str(program), testMode=mode)
        client.until(event="initialized")
        if mode == "close_control":
            client.send("configurationDone")
        client.until(event="terminated")
        assert fake_worker[0].poll() is not None


def test_failed_configuration_can_be_retried_before_launch_completes(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        launch = client.send("launch", program=str(program), testMode="config_failure")
        client.until(event="initialized")
        config = client.send("configurationDone")
        assert not client.until(response=config)["success"]
        assert not any(message.get("request_seq") == launch for message in client.received)
        config = client.send("configurationDone")
        assert client.until(response=config)["success"]
        assert client.until(response=launch)["success"]


def test_full_worker_pipe_does_not_block_other_requests_or_disconnect(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    with adapter_client() as (client, _, thread):
        configure(client, program, testMode="block")
        client.send("evaluate", expression="blocking")
        client.until(event="output")
        for _ in range(32):
            seq = client.send("evaluate", expression="x" * 65536)
        assert "pending" in client.until(response=seq)["message"]
        seq = client.send("disconnect")
        assert client.until(response=seq)["success"]
        client.until(event="terminated")
        thread.join(3)
        assert not thread.is_alive()
        assert fake_worker[0].poll() is not None


@pytest.mark.parametrize("payload", [b"Content-Length: -1\r\n\r\n", frame({"type": "event"})])
def test_bad_client_protocol_stops_an_owned_worker(
    program: Path, fake_worker: list[subprocess.Popen[bytes]], payload: bytes
) -> None:
    with adapter_client() as (client, _, thread):
        configure(client, program)
        client.writer.write(payload)
        client.writer.flush()
        thread.join(3)
        assert not thread.is_alive()
        assert fake_worker[0].poll() is not None


def test_invalid_request_arguments_and_sequences_are_correlated() -> None:
    with adapter_client() as (client, _, _):
        first = client.send("initialize", pathFormat="invalid")
        assert not client.until(response=first)["success"]
        write_message(client.writer, {"seq": first, "type": "request", "command": "initialize"})
        assert not client.receive()["success"]
        client.seq += 1
        write_message(
            client.writer,
            {"seq": client.seq, "type": "request", "command": "initialize", "arguments": []},
        )
        assert not client.receive()["success"]
        seq = client.send("initialize")
        assert client.until(response=seq)["success"]
        seq = client.send("threads")
        assert "No active" in client.until(response=seq)["message"]
        seq = client.send("launch", program="does-not-exist.kedi")
        assert not client.until(response=seq)["success"]


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_spawn_failure_is_actionable_without_echoing_environment(
    program: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error("private-secret-value")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        seq = client.send("launch", program=str(program))
        response = client.until(response=seq)
        assert response["message"] == "Unable to start the selected Python"
        assert "private-secret-value" not in json.dumps(response)


@pytest.mark.parametrize("field", ["program", "cwd"])
def test_failed_path_expansion_is_a_correlated_private_launch_error(
    program: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    expanduser = Path.expanduser

    def fail(path: Path) -> Path:
        if str(path).startswith("~"):
            raise RuntimeError("private-home-directory")
        return expanduser(path)

    monkeypatch.setattr(Path, "expanduser", fail)
    with adapter_client() as (client, _, _):
        client.send("initialize")
        client.receive()
        arguments = {"program": str(program), field: "~missing-user/program.kedi"}
        response = client.until(response=client.send("launch", **arguments))
        assert not response["success"]
        assert "private-home-directory" not in json.dumps(response)
        assert "missing-user" not in json.dumps(response)
        seq = client.send("threads")
        assert "No active" in client.until(response=seq)["message"]


def test_closing_client_output_terminates_owned_worker(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    class ClosedOutput(io.BytesIO):
        def write(self, data: Any) -> int:
            raise BrokenPipeError("Client closed its reader")

    with adapter_client() as (client, adapter, thread):
        configure(client, program)
        adapter._writer = ClosedOutput()
        client.send("threads")
        thread.join(3)
        assert not thread.is_alive()
        assert fake_worker[0].poll() is not None


def test_nonconsuming_client_has_bounded_backpressure_and_cleanup(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    released = threading.Event()
    started = threading.Event()

    class BlockedOutput(io.BytesIO):
        def write(self, data: Any) -> int:
            started.set()
            assert released.wait(timeout=5)
            return super().write(data)

    try:
        with adapter_client() as (client, adapter, thread):
            configure(client, program)
            adapter._writer = BlockedOutput()
            client.send("attach")
            assert started.wait(timeout=2)
            for _ in range(40):
                client.send("attach")
            thread.join(3)
            assert not thread.is_alive()
            assert fake_worker[0].poll() is not None
            released.set()
    finally:
        released.set()


@pytest.mark.parametrize("result", ["success", "already_exited", "failure", "nonzero"])
def test_windows_cleanup_only_targets_the_owned_process_tree(
    monkeypatch: pytest.MonkeyPatch, result: str
) -> None:
    from types import SimpleNamespace

    import kedi_debugger.server as server

    class OwnedProcess:
        pid = 123456
        returncode: int | None = 0 if result == "already_exited" else None
        kills = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

        def kill(self) -> None:
            self.kills += 1
            self.returncode = -9

    process = OwnedProcess()
    commands: list[list[str]] = []

    def taskkill(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        assert kwargs["timeout"] == 2
        if result == "failure":
            raise OSError("taskkill unavailable")
        if result == "success":
            process.returncode = 0
        return subprocess.CompletedProcess(command, 0 if result == "success" else 1)

    monkeypatch.setattr(server, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(subprocess, "run", taskkill)
    adapter = DebugAdapter(io.BytesIO(), io.BytesIO())
    adapter._process = process  # type: ignore[assignment]
    adapter._stop_worker()
    adapter._stop_worker()
    expected = [] if result == "already_exited" else [["taskkill", "/PID", "123456", "/T", "/F"]]
    assert commands == expected
    assert process.returncode is not None
    assert process.kills == (1 if result in ("failure", "nonzero") else 0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group ownership")
def test_owned_process_group_is_killed_but_unrelated_process_survives(
    program: Path, fake_worker: list[subprocess.Popen[bytes]]
) -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with adapter_client() as (client, _, thread):
            configure(client, program, testMode="tree")
            descendant = next(
                int(message["body"]["output"])
                for message in client.received
                if message.get("event") == "output"
            )
            client.send("disconnect")
            client.until(event="terminated")
            thread.join(3)
            assert not thread.is_alive()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(descendant, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                # Linux can retain an orphan zombie until init reaps it.
                stat = Path(f"/proc/{descendant}/stat")
                assert stat.exists() and stat.read_text().split()[2] == "Z"
            assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=3)
