"""Type annotations resolve to types, and their failures say what went wrong."""

from __future__ import annotations

import warnings

import pytest

from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


def _run(source: str) -> object:
    return compile_program(parse_program(source)).run_main()


def test_a_parameter_default_syntax_error_is_located() -> None:
    source = '@greet(name, punct = `"!"]`):\n  = Hi <name><punct>\n= <greet(Ada)>\n'

    with pytest.raises(KediExecutionError) as caught:
        _run(source)

    assert "default of punct" in str(caught.value)
    assert "SyntaxError" in str(caught.value)


def test_an_invalid_backtick_type_is_not_reported_as_a_value_mismatch() -> None:
    with pytest.raises(KediExecutionError, match=r"Invalid type expression 'list\[int'") as caught:
        _run("[scores: `list[int`] = `[1, 2]`\n= <scores>\n")

    assert "does not match" not in str(caught.value)


def test_the_recursion_limit_is_not_reported_as_a_parameter_mismatch() -> None:
    source = (
        "@total(n: int) -> int:\n"
        "  > if: `n <= 0`:\n"
        "    = `0`\n"
        "  = `n + total(n - 1)`\n"
        "= <total(`300`)>\n"
    )

    with pytest.raises(KediExecutionError, match="maximum recursion depth") as caught:
        _run(source)

    assert "does not match" not in str(caught.value)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[str] = hello\n@shout(word):\n  = <word>!\n= <shout(hey)>\n", "hey!"),
        ("@total(list: list[int]) -> int:\n  = `sum(list)`\n= <total(`[1, 2]`)>\n", "3"),
        ("@f() -> int:\n  [int] = x\n  = `3`\n= <`f()`>\n", "3"),
        ("[int] = hello\n[n: int] = `3`\n= <n>\n", "3"),
        ("[x: `int | str`] = `3`\n= `repr(x)`\n", "3"),
        ("[T: Any] = `int`\n[n: T] = `4`\n= `repr(n)`\n", "4"),
    ],
)
def test_values_named_like_types_do_not_hide_them(source: str, expected: str) -> None:
    assert _run(source) == expected


def test_a_parameter_named_like_its_type_keeps_the_type_check() -> None:
    source = "@double(int: int) -> int:\n  = `int * 2`\n= <`repr(double(2.5))`>\n"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KediExecutionError, match="Parameter 'int' does not match"):
            _run(source)


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
        ("Literal[1, 2]", "2", "2"),
        ("Literal[-1, 2.5, 1e3]", "-1", "-1"),
        ("Optional[Literal[1]]", "None", "None"),
        ("Annotated[int, 3]", "7", "7"),
        ("Callable[..., int]", "len", "<built-in function len>"),
    ],
)
def test_number_and_ellipsis_type_arguments(annotation: str, value: str, expected: str) -> None:
    assert _run(f"[t: {annotation}] = `{value}`\n= `repr(t)`\n") == expected
