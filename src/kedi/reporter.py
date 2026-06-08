from __future__ import annotations

import sys
from io import StringIO
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BufferReporter",
    "CliReporter",
    "NullReporter",
    "Reporter",
]


@runtime_checkable
class Reporter(Protocol):
    """Minimal progress/output contract for CLI and optimization flows."""

    def emit(self, message: str, *, flush: bool = False, end: str = "\n") -> None: ...

    def progress_char(self, char: str = ".") -> None: ...


class CliReporter:
    """Stdout-backed reporter for interactive CLI runs."""

    def __init__(self, *, quiet: bool = False, file: Any = None) -> None:
        self.quiet = quiet
        self._file = file or sys.stdout

    def emit(self, message: str, *, flush: bool = False, end: str = "\n") -> None:
        if self.quiet:
            return
        print(message, file=self._file, end=end, flush=flush)

    def progress_char(self, char: str = ".") -> None:
        if self.quiet:
            return
        print(char, end="", flush=True, file=self._file)


class NullReporter:
    """Discards all output (tests and quiet defaults)."""

    def emit(self, message: str, *, flush: bool = False, end: str = "\n") -> None:
        return None

    def progress_char(self, char: str = ".") -> None:
        return None


class BufferReporter:
    """Captures emitted lines for tests."""

    def __init__(self) -> None:
        self.buffer = StringIO()

    def emit(self, message: str, *, flush: bool = False, end: str = "\n") -> None:
        self.buffer.write(message)
        self.buffer.write(end)

    def progress_char(self, char: str = ".") -> None:
        self.buffer.write(char)

    def getvalue(self) -> str:
        return self.buffer.getvalue()
