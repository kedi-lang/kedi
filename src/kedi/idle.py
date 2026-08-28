"""Terminal interactive mode for the Kedi CLI."""

from __future__ import annotations

import os
import platform
import secrets
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TextIO

from ._version import __version__
from .error_rendering import render_execution_error, render_parse_error
from .errors import KediExecutionError
from .lang import parse_program
from .lang.parser.diagnostics import KediParseError

if TYPE_CHECKING:
    from .idle_input import PromptToolkitInput
    from .interactive_session import InteractiveSession


BEGIN_MARKER = "+++"
CONTINUATION_MARKER = "..."
IDLE_BANNER = " /\\_/\\\n( o.o )\n > ^ <"
_EXIT_COMMAND = ":exit"
_DUMP_COMMAND = ":dump"
_HELP_COMMANDS = frozenset({"help", "help()"})
_MULTILINE_COMMAND = ":multiline"
_SHOW_COMMAND = ":show"


class IdleConsole:
    """A Python-REPL-style terminal frontend for an InteractiveSession."""

    def __init__(
        self,
        session: InteractiveSession,
        *,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
        input_reader: Callable[[str], str] | None = None,
        multiline_reader: Callable[[str], str] | None = None,
        session_path: str | Path | None = None,
        highlight: bool = False,
    ) -> None:
        self._session = session
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self._input_reader = input_reader
        self._multiline_reader = multiline_reader
        self._buffer: list[str] = []
        self._multiline_requested = False
        self._exit_requested = False
        self._highlight = highlight
        self._prompt_input: PromptToolkitInput | None = None
        self._session_path = (
            None if session_path is None else Path(session_path).expanduser().resolve()
        )

    @property
    def prompt(self) -> str:
        """Return the prompt for the next input line."""
        marker = CONTINUATION_MARKER if self._buffer or self._multiline_requested else BEGIN_MARKER
        return f"{marker} "

    @property
    def exit_requested(self) -> bool:
        """Whether the interactive session should terminate."""
        return self._exit_requested

    def push(self, line: str) -> bool:
        """Push one input line and return whether continuation input is required."""
        if self._exit_requested:
            return False

        if self._buffer:
            if line:
                self._buffer.append(line)
                return True
            source = "\n".join(self._buffer)
            self._buffer.clear()
            self._execute(source)
            return False

        stripped = line.strip()
        if not stripped:
            return False
        if stripped == _EXIT_COMMAND:
            self._exit_requested = True
            return False
        if stripped == _DUMP_COMMAND:
            self.dump()
            return False
        if stripped.startswith(f"{_DUMP_COMMAND} "):
            self._stderr.write("Usage: :dump\n")
            return False
        if stripped in _HELP_COMMANDS:
            self._write_help()
            return False
        if stripped == _MULTILINE_COMMAND:
            self._multiline_requested = True
            return True
        if stripped.startswith(f"{_MULTILINE_COMMAND} "):
            self._stderr.write("Usage: :multiline\n")
            return False
        if stripped == _SHOW_COMMAND or stripped.startswith(f"{_SHOW_COMMAND} "):
            expression = stripped.removeprefix(_SHOW_COMMAND).strip()
            if not expression:
                self._stderr.write("Usage: :show <expression>\n")
                return False
            self._execute(f"= {expression}")
            return False
        if _requires_continuation(line):
            self._buffer.append(line)
            return True

        self._execute(line)
        return False

    def interact(self, *, banner: bool = True) -> None:
        """Read and execute lines until EOF or an explicit exit command."""
        if banner:
            self._write_banner()
        while not self._exit_requested:
            try:
                if self._multiline_requested:
                    source = self._read_multiline(self.prompt)
                    self._multiline_requested = False
                    if source.strip():
                        self._execute(source)
                    continue
                line = self._read_line(self.prompt)
            except EOFError:
                self._stdout.write("\n")
                break
            except KeyboardInterrupt:
                self._exit_requested = True
                break
            self.push(line)

    def _read_line(self, prompt: str) -> str:
        if self._input_reader is not None:
            return self._input_reader(prompt)
        if self._highlight:
            return self._get_prompt_input().read_line(prompt)
        return input(prompt)

    def _read_multiline(self, prompt: str) -> str:
        if self._multiline_reader is not None:
            return self._multiline_reader(prompt)
        return self._get_prompt_input().read_multiline(prompt)

    def _get_prompt_input(self) -> PromptToolkitInput:
        if self._prompt_input is None:
            from .idle_input import PromptToolkitInput

            self._prompt_input = PromptToolkitInput(
                highlight=self._highlight,
                requires_continuation=_requires_continuation,
                continuation_marker=CONTINUATION_MARKER,
            )
        return self._prompt_input

    def dump(self) -> Path | None:
        """Persist the current session and print the command that resumes it."""
        path = self._session_path or _default_session_path()
        try:
            dumped = self._session.dump(path).resolve()
        except Exception as exc:
            self._stderr.write(f"Session dump failed:\n{exc}\n")
            return None
        self._session_path = dumped
        self._stdout.write(f"Session dumped to {dumped}\n")
        self._stdout.write(
            f"To resume session, run -- kedi --idle --load {shlex.quote(str(dumped))}\n"
        )
        return dumped

    def _execute(self, source: str) -> None:
        try:
            with redirect_stdout(self._stdout), redirect_stderr(self._stderr):
                value = self._session.execute(source)
            if value is not None:
                self._stdout.write(f"{value!r}\n")
        except KediParseError as exc:
            self._stderr.write(render_parse_error(exc, use_color=self._use_color()))
        except KediExecutionError as exc:
            self._stderr.write(render_execution_error(exc, use_color=self._use_color()))
        except KeyboardInterrupt:
            self._exit_requested = True
        except Exception as exc:
            wrapped = KediExecutionError(message=str(exc) or type(exc).__name__, original=exc)
            self._stderr.write(render_execution_error(wrapped, use_color=self._use_color()))

    def _write_banner(self) -> None:
        self._stdout.write(
            f"{IDLE_BANNER}\nKedi {__version__} on {platform.system().lower()}\n"
            'Type "help" for interactive help, ":show" to inspect a value, '
            '":multiline" for a multiline fragment, ":dump" to save, '
            'or ":exit" to leave.\n'
        )

    def _write_help(self) -> None:
        self._stdout.write(
            f"Enter a complete Kedi fragment at {BEGIN_MARKER}. Compound blocks "
            f"continue at {CONTINUATION_MARKER}; press Tab to indent and submit an "
            "empty line to run them. Use :multiline for a freely editable multiline "
            "fragment; press Enter twice to run it, or Alt+Enter to force submission. "
            "Use :show <expression> to inspect a value. "
            "Use :dump to save the session. Use :exit, Ctrl+C, or Ctrl+D to exit.\n"
        )

    def _use_color(self) -> bool:
        return bool(getattr(self._stderr, "isatty", lambda: False)())


def _requires_continuation(source: str) -> bool:
    try:
        parse_program(source, source_path="<idle-input>")
    except KediParseError:
        pass
    else:
        return False
    stripped = source.rstrip()
    if stripped.endswith((":", "\\")):
        return True
    if stripped.count("```") % 2:
        return True
    return _has_unclosed_delimiter(stripped)


def _has_unclosed_delimiter(source: str) -> bool:
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    in_inline_python = False
    python_quote: str | None = None
    escaped = False
    for character in source:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if python_quote is not None:
            if character == python_quote:
                python_quote = None
            continue
        if character == "`":
            in_inline_python = not in_inline_python
        elif in_inline_python and character in {'"', "'"}:
            python_quote = character
        elif character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
    return bool(stack or in_inline_python or python_quote)


def _history_path() -> Path:
    configured = os.getenv("KEDI_HISTORY")
    return Path(configured).expanduser() if configured else Path.home() / ".kedi_history"


def _default_session_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = secrets.token_hex(4)
    return Path.home() / ".kedi" / "sessions" / f"idle-{timestamp}-{suffix}.kedi-state"


def _configure_readline() -> Callable[[], None]:
    try:
        import readline
    except ImportError:
        return lambda: None

    history_path = _history_path()
    try:
        readline.read_history_file(history_path)
    except OSError:
        pass
    binding = "bind ^I ed-insert" if "libedit" in (readline.__doc__ or "") else "tab: self-insert"
    readline.parse_and_bind(binding)

    def save_history() -> None:
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(history_path)
        except OSError:
            pass

    return save_history


def run_idle(
    *,
    adapter: str,
    model: str,
    record: bool = False,
    load_path: str | Path | None = None,
    highlight: bool = False,
) -> int:
    """Run a terminal Kedi session with Python-style prompts."""
    from .api import interactive
    from .interactive_session import load_session

    save_history = _configure_readline()
    try:
        session = (
            load_session(load_path)
            if load_path is not None
            else interactive(adapter=adapter, model=model)
        )
        with session as active_session:
            console = IdleConsole(
                active_session,
                session_path=load_path,
                highlight=highlight,
            )
            try:
                console.interact()
            finally:
                if record or load_path is not None:
                    console.dump()
    finally:
        save_history()
    return 0
