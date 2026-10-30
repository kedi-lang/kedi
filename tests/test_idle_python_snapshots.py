from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from click.testing import CliRunner

import kedi.cli as cli_module
from kedi.idle import IdleConsole
from kedi.interactive_session import InteractiveSession


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--snapshot-format", "python"], "require --idle"),
        (["--snapshot-format", "portable"], "require --idle"),
        (["--trust-snapshot"], "requires --idle --load"),
        (["--idle", "--trust-snapshot"], "requires --idle --load"),
        (["--load", "state", "--trust-snapshot"], "requires --idle --load"),
        (["--idle", "--snapshot-format", "unknown"], "Invalid value"),
    ],
)
def test_cli_rejects_invalid_snapshot_options(arguments: list[str], message: str) -> None:
    result = CliRunner().invoke(cli_module.main, arguments)
    assert result.exit_code == 2
    assert message in result.output


def _rich_snapshot(path: Path) -> None:
    with InteractiveSession() as session:
        session.execute("[value: Any] = `(lambda offset: lambda: offset + 2)(40)`")
        session.dump(path, format="python")


def test_python_dump_prints_explicit_trust_and_warning(tmp_path: Path) -> None:
    pytest.importorskip("cloudpickle")
    output, errors = StringIO(), StringIO()
    path = tmp_path / "session with spaces.state"
    with InteractiveSession() as session:
        session.execute("[value: Any] = `lambda: 42`")
        console = IdleConsole(
            session, stdout=output, stderr=errors, session_path=path, snapshot_format="python"
        )
        console.push(":dump")
    assert path.is_file()
    assert "--trust-snapshot" in output.getvalue()
    assert "execute arbitrary Python code" in output.getvalue()
    assert "credentials" in output.getvalue()
    assert errors.getvalue() == ""


@pytest.mark.parametrize("format", [None, "portable"])
def test_loaded_python_record_retains_mode_unless_explicitly_overridden(
    tmp_path: Path, format: str | None
) -> None:
    pytest.importorskip("cloudpickle")
    path = tmp_path / "rich.state"
    _rich_snapshot(path)
    original = path.read_bytes()
    arguments = [
        sys.executable,
        "-m",
        "kedi",
        "--idle",
        "--load",
        str(path),
        "--trust-snapshot",
        "--record",
        "--no-color",
    ]
    if format is not None:
        arguments.extend(["--snapshot-format", format])
    environment = dict(os.environ, NO_COLOR="1")
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        arguments,
        input=":exit\n",
        text=True,
        capture_output=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    if format == "portable":
        assert "Session dump failed" in result.stderr
        assert path.read_bytes() == original
    else:
        assert "--trust-snapshot" in result.stdout
        assert "Session dump failed" not in result.stderr
    with InteractiveSession.load(path, trusted=True) as restored:
        restored.execute("[answer: int] = `value()`")
        assert restored._env["answer"] == 42


def test_python_idle_resumes_rich_state_in_fresh_process(tmp_path: Path) -> None:
    pytest.importorskip("cloudpickle")
    path = tmp_path / "fresh.state"
    _rich_snapshot(path)
    environment = dict(os.environ, NO_COLOR="1")
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kedi",
            "--idle",
            "--load",
            str(path),
            "--trust-snapshot",
            "--no-color",
        ],
        input=":show `value()`\n:exit\n",
        text=True,
        capture_output=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "42" in result.stdout
    assert "--trust-snapshot" in result.stdout
    assert "Session dump failed" not in result.stderr


class _ExecutableValue:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> object:
        expression = f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')"
        return eval, (expression,)


def test_idle_refuses_python_snapshot_before_executing_reducers(tmp_path: Path) -> None:
    pytest.importorskip("cloudpickle")
    marker = tmp_path / "executed"
    path = tmp_path / "executable.state"
    with InteractiveSession(runtime_globals={"payload": _ExecutableValue(marker)}) as session:
        session.execute("[value: Any] = `payload`")
        session.dump(path, format="python")
    assert not marker.exists()
    result = CliRunner().invoke(cli_module.main, ["--idle", "--load", str(path), "--no-color"])
    assert result.exit_code == 1
    assert "trust" in result.output.lower()
    assert not marker.exists()


def test_cli_untrusted_python_snapshot_points_to_trust_flag(tmp_path: Path) -> None:
    pytest.importorskip("cloudpickle")
    path = tmp_path / "rich.state"
    _rich_snapshot(path)
    result = CliRunner().invoke(cli_module.main, ["--idle", "--load", str(path)])
    assert result.exit_code == 1
    assert "--trust-snapshot" in result.output
    assert "trusted=True" not in result.output
