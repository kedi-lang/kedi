from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.style import Style
from rich.text import Text

__all__ = ["KediConsole"]

_SECTION = Style(bold=True)
_MUTED = Style(color="bright_black")
_OK = Style(color="green")
_WARN = Style(color="yellow")
_FAIL = Style(color="red")
_ACCENT = Style(color="cyan", bold=True)


class KediConsole(Console):
    """Kedi CLI output with plain defaults and small semantic helpers."""

    def __init__(self, *args: Any, quiet: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("highlight", False)
        super().__init__(*args, **kwargs)
        self.quiet = quiet

    def print(self, *objects: Any, **kwargs: Any) -> None:
        flush = kwargs.pop("flush", False)
        kwargs.setdefault("markup", False)
        kwargs.setdefault("highlight", False)
        kwargs.setdefault("soft_wrap", True)
        super().print(*objects, **kwargs)
        if flush:
            self.file.flush()

    def banner(self, title: str, *, width: int = 80) -> None:
        if self.quiet:
            return
        rule = "=" * width
        self.print(Text(rule, style=_SECTION))
        self.print(Text(title, style=_ACCENT))
        self.print(Text(rule, style=_SECTION))

    def section(self, title: str, *, rule: str = "─", width: int = 80) -> None:
        if self.quiet:
            return
        self.print(Text(title, style=_SECTION))
        if rule:
            self.print(Text(rule * width, style=_MUTED))

    def detail(self, message: str) -> None:
        if self.quiet:
            return
        self.print(Text(f"   {message}", style=_MUTED))

    def progress_char(self, char: str = ".", *, flush: bool = True) -> None:
        if self.quiet:
            return
        self.print(char, end="", flush=flush)

    def ok(self, message: str) -> None:
        if self.quiet:
            return
        self.print(Text(f"✓ {message}", style=_OK))

    def warn(self, message: str) -> None:
        if self.quiet:
            return
        self.print(Text(f"⚠ {message}", style=_WARN))

    def fail(self, message: str) -> None:
        if self.quiet:
            return
        self.print(Text(f"✗ {message}", style=_FAIL))

    def inline(self, message: str, *, end: str = "", flush: bool = False) -> None:
        if self.quiet:
            return
        self.print(message, end=end, flush=flush)
