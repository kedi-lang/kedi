from __future__ import annotations

import pytest

from marker_pattern.lang import parse_program


invalid_sources = [
    # Unmatched angle
    "= <x",
    # Unmatched bracket
    "= [x",
    # Output with non-identifier
    "[42]",
    # Output with call inside
    "[foo(bar)]",
    # Bad procedure header - missing colon
    "@foo(x)\n  = x",
    # Bad procedure header - missing paren
    "@foo x):\n  = x",
    # Bad procedure header - missing end paren
    "@foo(x:\n  = x",
    # Unterminated call arg list
    "= <foo(alpha, <bar(x)>",
    # Nested unmatched: close order mismatch
    "= <foo(alpha, <bar(x)>]]>",
    # Bad output name characters
    "= [bad-name]",
    # Empty output name
    "= []",
    # Random unmatched close (not escaped)
    "= text >",
    # Dangling '[' and ']'
    "= text ]",
    # Bad output identifier
    "[not-valid!]",
    # Output with space
    "[two words]",
    # Unterminated procedure header
    "@proc(x: \n  = x",
    # Bad call with missing '>' closer
    "= <foo(x) ",
    # Bad nested with wrong order
    "= [a <x>]",
    # Call missing closing parenthesis
    "= <foo(alpha, <bar(x>)>",
    # Extra closing bracket
    "= <x>>",
    # Lone '@' line
    "@",
    # Header missing name
    "@ (x):\n  = x",
    # Empty brackets
    "[]",
    # Output containing call (forbidden)
    "[call(<x>)]",
]


@pytest.mark.parametrize("src", invalid_sources)
def test_invalid_sources_raise(src: str) -> None:
    with pytest.raises(Exception):
        parse_program(src)


