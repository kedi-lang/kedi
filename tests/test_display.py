"""Hosts can display values shown with ``> show: `value``` themselves (kedi.display)."""

from __future__ import annotations

from typing import Any

import pytest

from kedi.display import display, display_hook
from kedi.lang import compile_program, parse_program


def test_single_backtick_shows_go_to_the_display_hook(capsys: pytest.CaptureFixture[str]) -> None:
    shown: list[Any] = []

    def hook(value: Any) -> bool:
        if isinstance(value, int):
            shown.append(value)
            return True
        return False

    source = "> show: `40 + 2`\n> show: `'declined'`\n> show: text <`1`>\n> show:  `7`  \n"
    with display_hook(hook):
        compile_program(parse_program(source)).run_main()
    assert shown == [42, 7]
    assert capsys.readouterr().out == "declined\ntext 1\n"

    compile_program(parse_program("> show: `40 + 2`\n")).run_main()
    assert capsys.readouterr().out == "42\n"


def test_display_without_a_hook_declines() -> None:
    assert display(1) is False
    with display_hook(lambda value: "yes"):  # only True means displayed
        assert display(1) is False
