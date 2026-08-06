from __future__ import annotations

import builtins
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import kedi
import kedi.cli as cli_module
import kedi.idle as idle_module
from kedi.idle import IdleConsole


def test_idle_console_persists_native_values_and_uses_repr() -> None:
    stdout = StringIO()
    stderr = StringIO()

    with kedi.interactive() as session:
        console = IdleConsole(session, stdout=stdout, stderr=stderr)
        assert console.prompt == "+++ "
        assert console.push("[value: int] = `40`") is False
        assert console.push("= `value + 2`") is False

    assert stdout.getvalue() == "42\n"
    assert stderr.getvalue() == ""


def test_idle_console_uses_continuation_prompt_and_preserves_tab_indentation() -> None:
    stdout = StringIO()

    with kedi.interactive() as session:
        console = IdleConsole(session, stdout=stdout, stderr=StringIO())
        assert console.push("@double(value: int) -> int:") is True
        assert console.prompt == "... "
        assert console.push("\t= `value * 2`") is True
        assert console.push("") is False
        assert console.prompt == "+++ "
        console.push("= `double(4)`")

    assert stdout.getvalue() == "8\n"


@pytest.mark.parametrize(
    "source",
    [
        "> profile: helper:",
        "```",
        "~Person(",
        "= `sum(",
        "> use: \\",
    ],
)
def test_idle_console_detects_incomplete_fragments(source: str) -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("incomplete source must not execute")

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(source) is True


def test_idle_console_does_not_treat_natural_language_apostrophe_as_continuation() -> None:
    seen: list[str] = []

    class Session:
        def execute(self, source: str) -> None:
            seen.append(source)

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(">> What's the [capital] of Turkey?") is False
    assert seen == [">> What's the [capital] of Turkey?"]


def test_idle_console_renders_parse_execution_and_generic_errors() -> None:
    stderr = StringIO()
    with kedi.interactive() as session:
        console = IdleConsole(session, stdout=StringIO(), stderr=stderr)
        console.push("@")
        console.push("= `1 / 0`")

    class BrokenSession:
        def execute(self, _source: str) -> None:
            raise ValueError("broken backend")

    IdleConsole(BrokenSession(), stdout=StringIO(), stderr=stderr).push("= broken")  # type: ignore[arg-type]

    output = stderr.getvalue()
    assert "unexpected" in output
    assert "ZeroDivisionError" in output
    assert "ValueError" in output
    assert "broken backend" in output


def test_idle_console_handles_keyboard_interrupt_from_execution() -> None:
    class InterruptedSession:
        def execute(self, _source: str) -> None:
            raise KeyboardInterrupt

    stderr = StringIO()
    console = IdleConsole(InterruptedSession(), stdout=StringIO(), stderr=stderr)  # type: ignore[arg-type]

    assert console.push("= interrupted") is False
    assert stderr.getvalue() == "KeyboardInterrupt\n"


def test_idle_console_interact_handles_prompt_interrupt_eof_and_banner() -> None:
    prompts: list[str] = []
    inputs: list[str | BaseException] = [
        "@value() -> int:",
        KeyboardInterrupt(),
        "= `1`",
        EOFError(),
    ]

    def read(prompt: str) -> str:
        prompts.append(prompt)
        item = inputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    stdout = StringIO()
    stderr = StringIO()
    with kedi.interactive() as session:
        IdleConsole(
            session,
            stdout=stdout,
            stderr=stderr,
            input_reader=read,
        ).interact()

    assert prompts == ["+++ ", "... ", "+++ ", "+++ "]
    assert stdout.getvalue().startswith(" /\\_/\\\n( o.o )\n > ^ <\nKedi ")
    assert 'Type "help"' in stdout.getvalue()
    assert stdout.getvalue().endswith("1\n\n")
    assert stderr.getvalue() == "\nKeyboardInterrupt\n"


def test_idle_console_interact_can_exit_without_banner() -> None:
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return "exit()"

    console = IdleConsole(
        object(),  # type: ignore[arg-type]
        stdout=StringIO(),
        stderr=StringIO(),
        input_reader=read,
    )

    console.interact(banner=False)

    assert prompts == ["+++ "]
    assert console.exit_requested is True


@pytest.mark.parametrize("command", ["exit()", " quit() "])
def test_idle_console_exit_commands_stop_interaction(command: str) -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("exit commands must not execute")

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(command) is False
    assert console.exit_requested is True
    assert console.push("= ignored") is False


def test_idle_console_ignores_blank_primary_input_and_can_clear_buffer() -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("no source should execute")

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push("  ") is False
    assert console.push("@pending():") is True
    console.interrupt()
    assert console.prompt == "+++ "


@pytest.mark.parametrize("command", ["help", " help() "])
def test_idle_console_help_commands_explain_continuation(command: str) -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("help commands must not execute")

    stdout = StringIO()
    console = IdleConsole(Session(), stdout=stdout, stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(command) is False
    assert "Compound blocks continue at ..." in stdout.getvalue()
    assert "press Tab to indent" in stdout.getvalue()


def test_idle_console_show_is_repl_only_expression_shorthand() -> None:
    seen: list[str] = []

    class Session:
        def execute(self, source: str) -> str:
            seen.append(source)
            return "rendered"

    stdout = StringIO()
    stderr = StringIO()
    console = IdleConsole(Session(), stdout=stdout, stderr=stderr)  # type: ignore[arg-type]

    assert console.push(":show <value>") is False

    assert seen == ["= <value>"]
    assert stdout.getvalue() == "'rendered'\n"
    assert stderr.getvalue() == ""


def test_idle_console_show_requires_an_expression() -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("empty :show must not execute")

    stderr = StringIO()
    console = IdleConsole(Session(), stdout=StringIO(), stderr=stderr)  # type: ignore[arg-type]

    assert console.push(":show") is False
    assert stderr.getvalue() == "Usage: :show <expression>\n"


@pytest.mark.parametrize(
    "source, expected",
    [
        (r"`'a\'b'`([])", False),
        ("`'open", True),
        ("(]", True),
        ("]", False),
        ('`"value"`{x}', False),
    ],
)
def test_idle_delimiter_scanner_handles_python_quotes_and_pairs(
    source: str, expected: bool
) -> None:
    assert idle_module._has_unclosed_delimiter(source) is expected


def test_idle_readline_history_is_loaded_configured_and_saved(
    tmp_path: Path, monkeypatch: Any
) -> None:
    history = tmp_path / "history"
    history.write_text("= `1`\n", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    class Readline:
        @staticmethod
        def read_history_file(path: Path) -> None:
            calls.append(("read", path))

        @staticmethod
        def parse_and_bind(binding: str) -> None:
            calls.append(("bind", binding))

        @staticmethod
        def write_history_file(path: Path) -> None:
            calls.append(("write", path))

    monkeypatch.setenv("KEDI_HISTORY", str(history))
    monkeypatch.setitem(sys.modules, "readline", Readline)

    save = idle_module._configure_readline()
    save()

    assert calls == [
        ("read", history),
        ("bind", "tab: self-insert"),
        ("write", history),
    ]


def test_idle_readline_tolerates_missing_history_and_missing_module(
    tmp_path: Path, monkeypatch: Any
) -> None:
    history = tmp_path / "nested" / "history"
    written: list[Path] = []

    class Readline:
        @staticmethod
        def read_history_file(_path: Path) -> None:
            raise FileNotFoundError

        @staticmethod
        def parse_and_bind(_binding: str) -> None:
            pass

        @staticmethod
        def write_history_file(path: Path) -> None:
            written.append(path)

    monkeypatch.setenv("KEDI_HISTORY", str(history))
    monkeypatch.setitem(sys.modules, "readline", Readline)
    idle_module._configure_readline()()
    assert written == [history]
    assert history.parent.is_dir()

    original_import = builtins.__import__

    def missing_readline(name: str, *args: object, **kwargs: object) -> Any:
        if name == "readline":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_readline)
    idle_module._configure_readline()()


def test_idle_readline_tolerates_unwritable_history(tmp_path: Path, monkeypatch: Any) -> None:
    history = tmp_path / "history"

    class Readline:
        __doc__ = "libedit"

        @staticmethod
        def read_history_file(_path: Path) -> None:
            raise PermissionError

        @staticmethod
        def parse_and_bind(binding: str) -> None:
            assert binding == "bind ^I ed-insert"

        @staticmethod
        def write_history_file(_path: Path) -> None:
            raise PermissionError

    monkeypatch.setenv("KEDI_HISTORY", str(history))
    monkeypatch.setitem(sys.modules, "readline", Readline)

    idle_module._configure_readline()()


def test_run_idle_owns_session_and_saves_history(monkeypatch: Any) -> None:
    events: list[str] = []

    class Context:
        def __enter__(self) -> object:
            events.append("enter")
            return object()

        def __exit__(self, *args: object) -> None:
            events.append("exit")

    class Console:
        def __init__(self, _session: object) -> None:
            events.append("console")

        def interact(self) -> None:
            events.append("interact")

    monkeypatch.setattr(kedi.api, "interactive", lambda **_kwargs: Context())
    monkeypatch.setattr(idle_module, "IdleConsole", Console)
    monkeypatch.setattr(idle_module, "_configure_readline", lambda: lambda: events.append("save"))

    assert idle_module.run_idle(adapter="pydantic", model="test") == 0
    assert events == ["enter", "console", "interact", "exit", "save"]


def test_cli_idle_launches_without_source(monkeypatch: Any) -> None:
    received: dict[str, str] = {}

    def fake_run_idle(*, adapter: str, model: str) -> int:
        received.update(adapter=adapter, model=model)
        return 0

    monkeypatch.setattr(cli_module, "run_idle", fake_run_idle)
    result = CliRunner().invoke(
        cli_module.main,
        ["--idle", "--adapter", "langchain", "--adapter-model", "openai:gpt-test"],
    )

    assert result.exit_code == 0
    assert received == {"adapter": "langchain", "model": "openai:gpt-test"}


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--idle", "program.kedi"], "does not accept SOURCE"),
        (["--idle", "-c", "= done"], "does not accept SOURCE"),
        (["--idle", "--test"], "cannot be combined"),
        (["--idle", "--eval"], "cannot be combined"),
        (["--idle", "--optimize"], "cannot be combined"),
        (["--idle", "--parse"], "cannot be combined"),
        (["--idle", "--name", "Ada"], "does not accept SOURCE"),
    ],
)
def test_cli_idle_rejects_conflicting_modes(arguments: list[str], message: str) -> None:
    result = CliRunner().invoke(cli_module.main, arguments)

    assert result.exit_code == 2
    assert message in result.output


def test_cli_help_lists_idle_mode() -> None:
    result = CliRunner().invoke(cli_module.main, ["--help"])

    assert result.exit_code == 0
    assert "--idle" in result.output
