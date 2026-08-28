from __future__ import annotations

import builtins
import sys
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner
from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import kedi
import kedi.cli as cli_module
import kedi.idle as idle_module
import kedi.idle_input as idle_input_module
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


@pytest.mark.parametrize("keyword", ["if", "loop"])
def test_idle_console_buffers_template_condition_header_without_trailing_colon(
    keyword: str,
) -> None:
    seen: list[str] = []

    class Session:
        def execute(self, source: str) -> None:
            seen.append(source)

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(f"> {keyword}: İstanbul has about 16 million residents") is True
    assert console.prompt == "... "
    assert console.push("  = yes") is True
    assert console.push("") is False
    assert seen == [f"> {keyword}: İstanbul has about 16 million residents\n  = yes"]


def test_idle_console_executes_explicit_multiline_fragment_once() -> None:
    line_prompts: list[str] = []
    multiline_prompts: list[str] = []
    lines = iter([":multiline", ":exit"])
    source = "@double(value: int) -> int:\n  = `value * 2`\n\n= `double(4)`"

    def read_line(prompt: str) -> str:
        line_prompts.append(prompt)
        return next(lines)

    def read_multiline(prompt: str) -> str:
        multiline_prompts.append(prompt)
        return source

    stdout = StringIO()
    with kedi.interactive() as session:
        console = IdleConsole(
            session,
            stdout=stdout,
            stderr=StringIO(),
            input_reader=read_line,
            multiline_reader=read_multiline,
        )
        console.interact(banner=False)

    assert line_prompts == ["+++ ", "+++ "]
    assert multiline_prompts == ["... "]
    assert stdout.getvalue() == "8\n"


def test_idle_console_ignores_blank_explicit_multiline_fragment() -> None:
    lines = iter([":multiline", ":exit"])

    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("blank multiline source must not execute")

    console = IdleConsole(
        Session(),  # type: ignore[arg-type]
        stdout=StringIO(),
        stderr=StringIO(),
        input_reader=lambda _prompt: next(lines),
        multiline_reader=lambda _prompt: " \n",
    )

    console.interact(banner=False)

    assert console.exit_requested is True


def test_idle_console_lazily_shares_prompt_toolkit_input(monkeypatch: Any) -> None:
    created: list[tuple[bool, object, str]] = []

    class PromptInput:
        def __init__(
            self,
            *,
            highlight: bool,
            requires_continuation: object,
            continuation_marker: str,
        ) -> None:
            created.append((highlight, requires_continuation, continuation_marker))

        def read_line(self, prompt: str) -> str:
            return f"line:{prompt}"

        def read_multiline(self, prompt: str) -> str:
            return f"multiline:{prompt}"

    monkeypatch.setattr(idle_input_module, "PromptToolkitInput", PromptInput)
    console = IdleConsole(
        object(),  # type: ignore[arg-type]
        stdout=StringIO(),
        stderr=StringIO(),
        highlight=True,
    )

    assert console._read_line("+++ ") == "line:+++ "
    assert console._read_multiline("... ") == "multiline:... "
    assert created == [(True, idle_module._requires_continuation, "...")]


def test_idle_console_multiline_command_selects_continuation_prompt() -> None:
    console = IdleConsole(object(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(":multiline") is True
    assert console.prompt == "... "


def test_idle_console_multiline_command_validates_usage() -> None:
    stderr = StringIO()
    console = IdleConsole(object(), stdout=StringIO(), stderr=stderr)  # type: ignore[arg-type]

    assert console.push(":multiline now") is False
    assert stderr.getvalue() == "Usage: :multiline\n"


@pytest.mark.parametrize(
    "source, expected",
    [
        ("= `1`", False),
        ("= `1`\n", True),
        ("@value() -> int:\n", False),
        ("\n", True),
    ],
)
def test_multiline_double_enter_submits_only_complete_fragments(
    source: str,
    expected: bool,
) -> None:
    document = Document(source, cursor_position=len(source))

    assert (
        idle_input_module._should_submit_multiline(
            document,
            idle_module._requires_continuation,
        )
        is expected
    )


def test_idle_lexer_preserves_source_and_styles_kedi_and_python() -> None:
    source = "> if: `True`:\n  [value: int] = `1`\n  = <value>\n"
    lexer = idle_input_module.KediLexer()
    get_line = lexer.lex_document(Document(source))
    cached_get_line = lexer.lex_document(Document(source))
    fragments = [get_line(index) for index in range(len(source.split("\n")))]

    assert cached_get_line(0) == get_line(0)
    assert "\n".join("".join(text for _style, text in line) for line in fragments) == source
    assert any(
        style == "class:kedi.keyword" and "if" in text for line in fragments for style, text in line
    )
    assert any(
        style == "class:kedi.number" and text == "1" for line in fragments for style, text in line
    )


def test_idle_lexer_falls_back_to_plain_text_for_large_fragments() -> None:
    source = "> if: `True`:\n"
    lexer = idle_input_module.KediLexer(max_chars=len(source) - 1)

    get_line = lexer.lex_document(Document(source))

    assert get_line(0) == [("", "> if: `True`:")]


def test_idle_lexer_preserves_unicode_before_highlighted_tokens() -> None:
    source = ">> İstanbul için [summary].\n"
    lexer = idle_input_module.KediLexer()

    get_line = lexer.lex_document(Document(source))

    assert "".join(text for _style, text in get_line(0)) == source.rstrip("\n")


def test_idle_lexer_ignores_invalid_and_empty_semantic_tokens() -> None:
    lexer = idle_input_module.KediLexer()
    lexer._compute_semantic_tokens = lambda _source, _path: SimpleNamespace(
        data=[0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    )

    get_line = lexer.lex_document(Document("x"))

    assert get_line(0) == [("", "x")]
    assert get_line(2) == [("", "")]


def test_idle_line_fragments_ignore_fully_overlapped_spans() -> None:
    assert idle_input_module._line_fragments(
        "abcd",
        [(0, 3, "class:first"), (1, 2, "class:overlap")],
    ) == [("class:first", "abc"), ("", "d")]
    assert idle_input_module._line_fragments("", []) == [("", "")]


def test_prompt_toolkit_input_configures_prompts_and_key_bindings(monkeypatch: Any) -> None:
    sessions: list[dict[str, object]] = []
    prompts: list[tuple[str, dict[str, object]]] = []

    class PromptSession:
        def __init__(self, **kwargs: object) -> None:
            sessions.append(kwargs)

        def prompt(self, prompt: str, **kwargs: object) -> str:
            prompts.append((prompt, kwargs))
            return "entered"

    monkeypatch.setattr(idle_input_module, "PromptSession", PromptSession)
    prompt_input = idle_input_module.PromptToolkitInput(
        highlight=True,
        requires_continuation=idle_module._requires_continuation,
        continuation_marker="...",
    )

    assert prompt_input.read_line("+++ ") == "entered"
    assert prompt_input.read_multiline("... ") == "entered"
    assert prompt_input._continuation_prompt(6, 2, 0) == "  ... "
    assert isinstance(sessions[0]["lexer"], idle_input_module.KediLexer)
    assert sessions[0]["style"] is idle_input_module.KEDI_IDLE_STYLE
    assert prompts[0][1]["multiline"] is False
    assert prompts[1][1]["multiline"] is True

    plain_input = idle_input_module.PromptToolkitInput(
        highlight=False,
        requires_continuation=idle_module._requires_continuation,
        continuation_marker="...",
    )
    assert plain_input._color_depth is None
    assert sessions[1]["lexer"] is None
    assert sessions[1]["style"] is None

    class Buffer:
        def __init__(self, source: str = "") -> None:
            self.document = Document(source, cursor_position=len(source))
            self.inserted: list[str] = []
            self.deleted: list[tuple[str, int]] = []
            self.submitted = 0

        def insert_text(self, text: str) -> None:
            self.inserted.append(text)

        def delete_before_cursor(self, *, count: int) -> None:
            self.deleted.append(("before", count))

        def delete(self, *, count: int) -> None:
            self.deleted.append(("at", count))

        def validate_and_handle(self) -> None:
            self.submitted += 1

    line_buffer = Buffer()
    prompt_input._line_bindings.bindings[0].handler(SimpleNamespace(current_buffer=line_buffer))
    prompt_input._line_bindings.bindings[1].handler(SimpleNamespace(current_buffer=line_buffer))
    prompt_input._line_bindings.bindings[2].handler(SimpleNamespace(current_buffer=line_buffer))
    assert line_buffer.inserted == ["\t"]
    assert line_buffer.deleted == [("before", 1), ("at", 1)]

    tab_buffer = Buffer()
    force_buffer = Buffer()
    submit_buffer = Buffer("= `1`\n")
    continue_buffer = Buffer("@value() -> int:\n")
    bindings = prompt_input._multiline_bindings.bindings
    bindings[0].handler(SimpleNamespace(current_buffer=tab_buffer))
    bindings[1].handler(SimpleNamespace(current_buffer=force_buffer))
    bindings[2].handler(SimpleNamespace(current_buffer=submit_buffer))
    bindings[2].handler(SimpleNamespace(current_buffer=continue_buffer))
    bindings[3].handler(SimpleNamespace(current_buffer=continue_buffer))
    bindings[4].handler(SimpleNamespace(current_buffer=continue_buffer))

    assert tab_buffer.inserted == ["\t"]
    assert force_buffer.submitted == 1
    assert submit_buffer.submitted == 1
    assert continue_buffer.inserted == ["\n"]
    assert continue_buffer.deleted == [("before", 1), ("at", 1)]


def test_multiline_input_preserves_bracketed_paste_until_explicit_submission(
    monkeypatch: Any,
) -> None:
    source = "@double(value: int) -> int:\n\t= `value * 2`\n\n= `double(4)`"

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(
            idle_input_module,
            "PromptSession",
            lambda **kwargs: PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                **kwargs,
            ),
        )
        prompt_input = idle_input_module.PromptToolkitInput(
            highlight=False,
            requires_continuation=idle_module._requires_continuation,
            continuation_marker="...",
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(prompt_input.read_multiline, "... ")
            pipe_input.send_text(f"\x1b[200~{source}\x1b[201~\r\r")
            submitted = result.result(timeout=3)

    assert submitted == f"{source}\n"


@pytest.mark.parametrize(
    "typed",
    [
        "> if: İstanbulX\x7f has about 16 million residents\r",
        "> if: İstanbX\x1b[D\x1b[3~ul has about 16 million residents\r",
    ],
)
def test_highlighted_line_input_preserves_unicode_backspace_and_delete(
    typed: str,
    monkeypatch: Any,
) -> None:
    expected = "> if: İstanbul has about 16 million residents"

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(
            idle_input_module,
            "PromptSession",
            lambda **kwargs: PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                **kwargs,
            ),
        )
        prompt_input = idle_input_module.PromptToolkitInput(
            highlight=True,
            requires_continuation=idle_module._requires_continuation,
            continuation_marker="...",
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(prompt_input.read_line, "+++ ")
            pipe_input.send_text(typed)
            submitted = result.result(timeout=3)

    assert submitted == expected


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
    assert console.exit_requested is True
    assert stderr.getvalue() == ""


def test_idle_console_prompt_interrupt_exits_silently_after_banner() -> None:
    prompts: list[str] = []
    inputs: list[str | BaseException] = [
        "@value() -> int:",
        KeyboardInterrupt(),
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
        console = IdleConsole(
            session,
            stdout=stdout,
            stderr=stderr,
            input_reader=read,
        )
        console.interact()

    assert prompts == ["+++ ", "... "]
    assert stdout.getvalue().startswith(" /\\_/\\\n( o.o )\n > ^ <\nKedi ")
    assert 'Type "help"' in stdout.getvalue()
    assert console.exit_requested is True
    assert stderr.getvalue() == ""


def test_idle_console_eof_exits_with_terminal_newline() -> None:
    def read(_prompt: str) -> str:
        raise EOFError

    stdout = StringIO()
    console = IdleConsole(
        object(),  # type: ignore[arg-type]
        stdout=stdout,
        stderr=StringIO(),
        input_reader=read,
    )

    console.interact(banner=False)

    assert stdout.getvalue() == "\n"


def test_idle_console_interact_can_exit_without_banner() -> None:
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return ":exit"

    console = IdleConsole(
        object(),  # type: ignore[arg-type]
        stdout=StringIO(),
        stderr=StringIO(),
        input_reader=read,
    )

    console.interact(banner=False)

    assert prompts == ["+++ "]
    assert console.exit_requested is True


def test_idle_console_exit_command_stops_interaction() -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("exit commands must not execute")

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(" :exit ") is False
    assert console.exit_requested is True
    assert console.push("= ignored") is False


@pytest.mark.parametrize("source", ["exit()", "quit()"])
def test_idle_console_does_not_special_case_python_exit_calls(source: str) -> None:
    seen: list[str] = []

    class Session:
        def execute(self, fragment: str) -> None:
            seen.append(fragment)

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push(source) is False
    assert console.exit_requested is False
    assert seen == [source]


def test_idle_console_ignores_blank_primary_input() -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise AssertionError("no source should execute")

    console = IdleConsole(Session(), stdout=StringIO(), stderr=StringIO())  # type: ignore[arg-type]

    assert console.push("  ") is False


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


def test_idle_console_dump_persists_and_prints_resume_command(tmp_path: Path) -> None:
    snapshot = tmp_path / "session state.kedi-state"
    seen: list[Path] = []

    class Session:
        def dump(self, path: str | Path) -> Path:
            resolved = Path(path)
            seen.append(resolved)
            return resolved

    stdout = StringIO()
    console = IdleConsole(
        Session(),  # type: ignore[arg-type]
        stdout=stdout,
        stderr=StringIO(),
        session_path=snapshot,
    )

    assert console.push(":dump") is False
    assert seen == [snapshot.resolve()]
    assert stdout.getvalue() == (
        f"Session dumped to {snapshot.resolve()}\n"
        "To resume session, run -- "
        f"kedi --idle --load '{snapshot.resolve()}'\n"
    )


def test_idle_console_dump_reports_failure_and_rejects_arguments(tmp_path: Path) -> None:
    class Session:
        def dump(self, _path: str | Path) -> Path:
            raise kedi.SessionDumpError(["unsupported value"])

    stderr = StringIO()
    console = IdleConsole(
        Session(),  # type: ignore[arg-type]
        stdout=StringIO(),
        stderr=stderr,
        session_path=tmp_path / "session.kedi-state",
    )

    assert console.push(":dump") is False
    assert console.push(":dump elsewhere") is False
    assert "Session dump failed:" in stderr.getvalue()
    assert "unsupported value" in stderr.getvalue()
    assert stderr.getvalue().endswith("Usage: :dump\n")


def test_idle_console_propagates_system_exit_from_kedi_execution() -> None:
    class Session:
        def execute(self, _source: str) -> None:
            raise SystemExit(7)

    console = IdleConsole(
        Session(),  # type: ignore[arg-type]
        stdout=StringIO(),
        stderr=StringIO(),
    )

    with pytest.raises(SystemExit) as info:
        console.push("= `exit_now()`")
    assert info.value.code == 7


def test_idle_default_session_path_uses_private_registry_directory(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(idle_module.Path, "home", lambda: tmp_path)

    path = idle_module._default_session_path()

    assert path.parent == tmp_path / ".kedi" / "sessions"
    assert path.name.startswith("idle-")
    assert path.suffix == ".kedi-state"


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


def test_prompt_toolkit_uses_existing_readline_history(monkeypatch: Any) -> None:
    entries = ["= `1`", "@value():\n  = `2`"]

    class Readline:
        @staticmethod
        def get_current_history_length() -> int:
            return len(entries)

        @staticmethod
        def get_history_item(index: int) -> str:
            return entries[index - 1]

        @staticmethod
        def add_history(source: str) -> None:
            entries.append(source)

    monkeypatch.setitem(sys.modules, "readline", Readline)
    history = idle_input_module.ReadlineHistory()

    assert history.load_history_strings() == ["@value():\n  = `2`", "= `1`"]
    history.store_string("> if: `True`:\n  = done")
    assert entries[-1] == "> if: `True`:\n  = done"


def test_run_idle_owns_session_and_saves_history(monkeypatch: Any) -> None:
    events: list[str] = []

    class Context:
        def __enter__(self) -> object:
            events.append("enter")
            return object()

        def __exit__(self, *args: object) -> None:
            events.append("exit")

    class Console:
        def __init__(
            self,
            _session: object,
            *,
            session_path: object = None,
            highlight: bool = False,
        ) -> None:
            assert session_path is None
            assert highlight is False
            events.append("console")

        def interact(self) -> None:
            events.append("interact")

    monkeypatch.setattr(kedi.api, "interactive", lambda **_kwargs: Context())
    monkeypatch.setattr(idle_module, "IdleConsole", Console)
    monkeypatch.setattr(idle_module, "_configure_readline", lambda: lambda: events.append("save"))

    assert idle_module.run_idle(adapter="pydantic", model="test") == 0
    assert events == ["enter", "console", "interact", "exit", "save"]


def test_run_idle_records_before_close_and_preserves_system_exit(monkeypatch: Any) -> None:
    events: list[str] = []

    class Session:
        def __enter__(self) -> Session:
            events.append("enter")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("exit")

    class Console:
        def __init__(
            self,
            session: object,
            *,
            session_path: object = None,
            highlight: bool = False,
        ) -> None:
            assert isinstance(session, Session)
            assert session_path is None
            assert highlight is False

        def interact(self) -> None:
            events.append("interact")
            raise SystemExit(9)

        def dump(self) -> None:
            events.append("dump")

    monkeypatch.setattr(kedi.api, "interactive", lambda **_kwargs: Session())
    monkeypatch.setattr(idle_module, "IdleConsole", Console)
    monkeypatch.setattr(idle_module, "_configure_readline", lambda: lambda: events.append("save"))

    with pytest.raises(SystemExit) as info:
        idle_module.run_idle(adapter="pydantic", model="test", record=True)

    assert info.value.code == 9
    assert events == ["enter", "interact", "dump", "exit", "save"]


def test_run_idle_loads_and_continues_recording_same_snapshot(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    snapshot = tmp_path / "loaded.kedi-state"
    events: list[tuple[str, object]] = []

    class Session:
        def __enter__(self) -> Session:
            return self

        def __exit__(self, *args: object) -> None:
            events.append(("close", args))

    class Console:
        def __init__(
            self,
            session: object,
            *,
            session_path: object = None,
            highlight: bool = False,
        ) -> None:
            assert isinstance(session, Session)
            assert highlight is False
            events.append(("path", session_path))

        def interact(self) -> None:
            events.append(("interact", None))

        def dump(self) -> None:
            events.append(("dump", None))

    monkeypatch.setattr(
        kedi.interactive_session,
        "load_session",
        lambda path: events.append(("load", path)) or Session(),
    )
    monkeypatch.setattr(idle_module, "IdleConsole", Console)
    monkeypatch.setattr(idle_module, "_configure_readline", lambda: lambda: None)

    assert (
        idle_module.run_idle(
            adapter="pydantic",
            model="test",
            load_path=snapshot,
        )
        == 0
    )
    assert events == [
        ("load", snapshot),
        ("path", snapshot),
        ("interact", None),
        ("dump", None),
        ("close", (None, None, None)),
    ]


def test_run_idle_record_and_load_round_trip_real_session(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    snapshot = tmp_path / "recorded.kedi-state"
    inputs = iter(["[value: int] = `40`", ":exit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))
    monkeypatch.setattr(idle_module, "_default_session_path", lambda: snapshot)
    monkeypatch.setattr(idle_module, "_configure_readline", lambda: lambda: None)

    assert (
        idle_module.run_idle(
            adapter="pydantic",
            model="test-model",
            record=True,
        )
        == 0
    )

    resumed_inputs = iter(["[value: int] = `42`", ":exit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(resumed_inputs))
    assert (
        idle_module.run_idle(
            adapter="pydantic",
            model="ignored-after-load",
            load_path=snapshot,
        )
        == 0
    )

    with kedi.load_session(snapshot) as restored:
        assert restored.execute("= `value`") == 42


def test_run_idle_records_before_propagating_real_system_exit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    snapshot = tmp_path / "system-exit.kedi-state"
    inputs = iter(["`raise SystemExit(5)`"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))
    monkeypatch.setattr(idle_module, "_default_session_path", lambda: snapshot)
    monkeypatch.setattr(idle_module, "_configure_readline", lambda: lambda: None)

    with pytest.raises(SystemExit) as info:
        idle_module.run_idle(
            adapter="pydantic",
            model="test-model",
            record=True,
        )

    assert info.value.code == 5
    with kedi.load_session(snapshot) as restored:
        assert restored.execute("= `42`") == 42


def test_run_idle_records_after_prompt_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    snapshot = tmp_path / "keyboard-interrupt.kedi-state"

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupt)
    monkeypatch.setattr(idle_module, "_default_session_path", lambda: snapshot)
    monkeypatch.setattr(idle_module, "_configure_readline", lambda: lambda: None)

    assert (
        idle_module.run_idle(
            adapter="pydantic",
            model="test-model",
            record=True,
        )
        == 0
    )
    with kedi.load_session(snapshot) as restored:
        assert restored.execute("= `42`") == 42


def test_cli_idle_launches_without_source(monkeypatch: Any) -> None:
    received: dict[str, object] = {}

    def fake_run_idle(
        *,
        adapter: str,
        model: str,
        record: bool,
        load_path: Path | None,
        highlight: bool,
    ) -> int:
        received.update(
            adapter=adapter,
            model=model,
            record=record,
            load_path=load_path,
            highlight=highlight,
        )
        return 0

    monkeypatch.setattr(cli_module, "run_idle", fake_run_idle)
    result = CliRunner().invoke(
        cli_module.main,
        ["--idle", "--adapter", "langchain", "--adapter-model", "openai:gpt-test"],
    )

    assert result.exit_code == 0
    assert received == {
        "adapter": "langchain",
        "model": "openai:gpt-test",
        "record": False,
        "load_path": None,
        "highlight": False,
    }


def test_cli_idle_forwards_record_and_load(tmp_path: Path, monkeypatch: Any) -> None:
    snapshot = tmp_path / "session.kedi-state"
    received: dict[str, object] = {}

    def fake_run_idle(**kwargs: object) -> int:
        received.update(kwargs)
        return 0

    monkeypatch.setattr(cli_module, "run_idle", fake_run_idle)
    result = CliRunner().invoke(
        cli_module.main,
        ["--idle", "--record", "--load", str(snapshot)],
    )

    assert result.exit_code == 0
    assert received["record"] is True
    assert received["load_path"] == snapshot


def test_cli_idle_forwards_highlighting(monkeypatch: Any) -> None:
    received: dict[str, object] = {}

    def fake_run_idle(**kwargs: object) -> int:
        received.update(kwargs)
        return 0

    monkeypatch.setattr(cli_module, "run_idle", fake_run_idle)
    result = CliRunner().invoke(cli_module.main, ["--idle", "--highlight"])

    assert result.exit_code == 0
    assert received["highlight"] is True


def test_cli_idle_renders_snapshot_load_failure(monkeypatch: Any) -> None:
    def fail_run(**_kwargs: object) -> int:
        raise kedi.SessionLoadError("invalid snapshot")

    monkeypatch.setattr(cli_module, "run_idle", fail_run)
    result = CliRunner().invoke(
        cli_module.main,
        ["--idle", "--load", "missing.kedi-state"],
    )

    assert result.exit_code == 1
    assert "Error: invalid snapshot" in result.output


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
        (["program.kedi", "--record"], "require --idle"),
        (["program.kedi", "--load", "session.kedi-state"], "require --idle"),
        (["program.kedi", "--highlight"], "require --idle"),
        (["install", "--record"], "require --idle"),
        (["skills", "--load", "session.kedi-state"], "require --idle"),
        (["--idle", "install"], "does not accept SOURCE"),
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
    assert "--record" in result.output
    assert "--highlight" in result.output
    assert "--load" in result.output
