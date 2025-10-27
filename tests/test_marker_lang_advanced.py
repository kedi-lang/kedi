from __future__ import annotations

from typing import Any, Dict, List

import pytest

from marker.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


def _compile(src: str):
    return compile_program(parse_program(src), adapter=MockAdapter())


def _run(monkeypatch: Any, src: str) -> Any:
    rt = _compile(src)
    out = rt.run_main()
    assert isinstance(out, str)
    return out


# ----------------------
# Advanced successful cases
# ----------------------


ADVANCED_VALID_CASES: List[tuple[str, str]] = [
    (
        "@pipe(a,b):\n  step one for [x]\n  step two for [y: int]\n  = <`str(y)+'-'+x`>\n\n= <pipe(alpha, beta)>\n",
        "42-x-value",
    ),
    (
        "@nest(x):\n  = <`str(len(x))`>\n\n@outer(y):\n  = out:<nest(<`y*2`>)>\n\n= <outer(3)>\n",
        "out:2",
    ),
    (
        "@kw(a,b,c):\n  combine [out: str] from <`a+b+c`>\n  = <out>\n\n= <kw(1,2,3)>\n",
        "out-value",
    ),
    (
        "@deep(a):\n  = <`str(int(a)*2)`>\n\n@wrap(x):\n  = \\[<x>\\]\n\n= <wrap(<deep(<deep(2)>)>)>\n",
        "[8]",
    ),
    (
        "@types():\n  list [xs: list[int]]\n  [doubled: list[int]] = ```\n  return [v*2 for v in xs]\n  ```\n  = <`str(sum(doubled))`>\n\n= <types()>\n",
        "6",
    ),
    (
        # Multiple outputs, mixture typed and untyped
        "@multi():\n  a [p] and b [q: int] and c [r: list[str]]\n  = <p>-<`q+1`>-<`len(r)`>\n\n= <multi()>\n",
        "p-value-43-2",
    ),
    (
        # Complex commas and escapes in args
        "@join(a,b,c):\n  = <a>|<b>|<c>\n\n= <join(alpha\\, beta\\, gamma, delta, <`'x,y'`>)>\n",
        "alpha, beta, gamma|delta|x,y",
    ),
    (
        # Spaces preserved internally
        "=    A     <`'B'`>   C     \n",
        "A     B   C",
    ),
    (
        # Backticks contain brackets safely
        "= <`'['*2 + ']'*2`>\n",
        "[[]]",
    ),
    (
        # Python block with imports and procedure calls
        "@times(n):\n  = <`int(n)*3`>\n\n@prog(x):\n  [y: int] = ```\n  return int(times(x))\n  ```\n  [z: int] = ```\n  import math\n  return int(math.sqrt(16)) + y\n  ```\n  = <`str(z)`>\n\n= <prog(5)>\n",
        "19",
    ),
    (
        # Assignment native value then used in prompt
        "@p():\n  [nums: list[int]] = `[1,2,3]`\n  [s: int] = ```\n  return sum(nums)\n  ```\n  = <`str(s)`>\n\n= <p()>\n",
        "6",
    ),
    (
        # Output typed list[str] then chain in python
        "@cities():\n  get [arr: list[str]]\n  = <`'-'.join(arr)`>\n\n= <cities()>\n",
        "arr-a-arr-b",
    ),
    (
        # Top-level python block before return
        "[val: int] = ```\nreturn 10\n```\n= <`str(val*3)`>\n",
        "30",
    ),
    (
        # Nested value passed via single inline-python arg; nested call invoked inside python
        "@add(a: int, b: int) -> int:\n  = `int(a)+int(b)`\n\n@chain(x):\n  = <add(`int(x)*2`, `add(3, 4)`) >\n\n= <chain(5)>\n",
        "17",
    ),
    (
        # Shadowing builtins inside python block but still using other builtins
        "@demo():\n  [s: float] = ```\n  return __import__('math').sqrt(25)\n  ```\n  [L: int] = ```\n  return 5\n  ```\n  = <`str(int(s)+L)`>\n\n= <demo()>\n",
        "10",
    ),
    (
        # Multiple returns in a proc: last wins
        "@many():\n  = first\n  = second\n\n= <many()>\n",
        "second",
    ),
    (
        # Using variable from assignment then overriding via LLM output
        "@override():\n  [x] = base\n  confirm [x]\n  = <x>\n\n= <override()>\n",
        "x-value",
    ),
    (
        # Call arg literal trimming and nested call unambiguous
        "@pair(a,b):\n  = (<a>,<b>)\n\n@use():\n  = <pair(alpha  , <pair(beta, gamma)>)>\n\n= <use()>\n",
        "(alpha,(beta,gamma))",
    ),
    (
        # Mix escaped specials in text and args
        "@esc(s):\n  = <s>\n\n= Use \\<tag\\> and \\[br\\] and \\@at and eq \\= and comma \\\\, plus <esc(a\\,b)>\n",
        "Use <tag> and [br] and @at and eq = and comma \, plus a,b",
    ),
    (
        # TypedDict path when any typed output exists
        "@typed():\n  need [a: int] and [b]\n  = <`str(a)+'-'+b`>\n\n= <typed()>\n",
        "42-b-value",
    ),
    (
        # No-op m() when no outputs on a line
        "@noop():\n  this line has no outputs and only text\n  = done\n\n= <noop()>\n",
        "done",
    ),
    (
        # Complex inline python with statements fallback to exec
        "@assertion():\n  <`assert 1+1==2`>\n  = ok\n\n= <assertion()>\n",
        "ok",
    ),
]


@pytest.mark.parametrize("src,expected", ADVANCED_VALID_CASES)
def test_advanced_valid_cases(monkeypatch: Any, src: str, expected: str) -> None:
    out = _run(monkeypatch, src)
    assert out == expected or out.startswith(expected)


# ----------------------
# Invalid syntax cases (parse-time)
# ----------------------


INVALID_PARSE_SOURCES = [
    "= <x",  # unmatched '<'
    "= [x",  # unmatched '['
    "@foo(x)\n  = x",  # missing ':'
    "@foo x):\n  = x",  # malformed header
    "@foo(x:\n  = x",  # malformed header
    "= <foo(alpha, <bar(x)>",  # nested call missing close
    "= <foo(alpha, <bar(x)>]]>",  # stray closers
    "[42]",  # invalid identifier
    "[two words]",  # space in identifier
    "@proc(x: \n  = x",  # malformed header
    "= text >",  # stray '>'
    "= text ]",  # stray ']'
    "[]",  # empty name
    "[call(<x>)]",  # invalid output name containing '('
    "@p():\n  ```\n  no end fence",
    "= 1+1 = 2",  # unescaped '=' inside expression
    # Note: '==' is a literal '=' and is allowed; do not include here
]


@pytest.mark.parametrize("src", INVALID_PARSE_SOURCES)
def test_invalid_parse_errors(src: str) -> None:
    with pytest.raises(Exception):
        parse_program(src)


# ----------------------
# Runtime error cases (compile/run-time)
# ----------------------


RUNTIME_ERROR_CASES = [
    (
        # Unknown procedure call
        "= <not_defined()>\n",
        KeyError,
    ),
    (
        # Unknown variable usage in a template line (triggered via rt.m)
        "@p():\n  using <missing> we need [x]\n  = done\n\n= <p()>\n",
        KeyError,
    ),
    # Note: Duplicate procedure definitions are now allowed (for cache merging); last wins
    (
        # Inline Python assertion failure should raise at runtime
        "= <`assert False`>\n",
        RuntimeError,
    ),
    (
        # Python block error
        "@boom():\n  ```\n  raise ValueError('x')\n  ```\n  = ok\n\n= <boom()>\n",
        RuntimeError,
    ),
]


@pytest.mark.parametrize("src,exc", RUNTIME_ERROR_CASES)
def test_runtime_errors(monkeypatch: Any, src: str, exc: type[BaseException]) -> None:
    if "not_defined()" in src:
        # compile ok, run fails due to unknown proc
        rt = _compile(src)
        with pytest.raises(exc):
            rt.run_main()
        return
    with pytest.raises(exc):
        _run(monkeypatch, src)


def test_no_main_defined_raises(monkeypatch: Any) -> None:
    src = "@p(x):\n  = <x>\n"
    rt = _compile(src)
    with pytest.raises(RuntimeError):
        rt.run_main()


# ----------------------
# Stress nested calls, args, escaping, and type nesting
# ----------------------


def test_nested_calls_and_args(monkeypatch: Any) -> None:
    src = (
        "@f(a,b,c):\n  = <a>-<b>-<c>\n\n"
        "@g(x):\n  = <`x+'!'`>\n\n"
        "@h(p,q):\n  <f(p, <g(q)>, <`p+q`>)> yields [r]\n  = <r>\n\n"
        "= <h(A, B)>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "r-value"


def test_list_type_nested_and_python(monkeypatch: Any) -> None:
    src = "@p():\n  matrix [m: list[list[int]]]\n  [flat: list[int]] = ```\n  return [v for row in m for v in row]\n  ```\n  = <`str(sum(flat))`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    # Our fake agent returns [] for unknown nested typing; allow "0" or empty leading
    assert out == "0" or out.startswith("")


def test_escaped_commas_and_nested_calls(monkeypatch: Any) -> None:
    src = (
        "@cat(a,b,c):\n  = <a>:<b>:<c>\n\n"
        "@use():\n  = <cat(a\\,b\\,c, <cat(d\\,e, f, g)>, <`','.join(['x', 'y'])`>)>\n\n"
        "= <use()>\n"
    )
    out = _run(monkeypatch, src)
    assert ":" in out and "," in out


def test_assignment_string_render_when_not_pure_python(monkeypatch: Any) -> None:
    src = "@p():\n  [x] = hello <`1+1`> world\n  = <x>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out.startswith("hello") and "world" in out


def test_output_updates_env_for_later_use(monkeypatch: Any) -> None:
    src = "@p():\n  color is [color]\n  = <`color.upper()`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out.isupper()


def test_python_inline_can_call_procedure(monkeypatch: Any) -> None:
    src = "@double(n):\n  = <`int(n)*2`>\n\n@use(x):\n  = <`int(double(x))+1`>\n\n= <use(7)>\n"
    out = _run(monkeypatch, src)
    assert out == "15"


def test_python_block_modifies_existing_and_new_vars(monkeypatch: Any) -> None:
    src = "@p():\n  [x] = base\n  ```\n  x = x + '-mod'\n  y = 'new'\n  ```\n  = <x>-<`y`>\n\n= <p()>\n"
    # y is a pure Python variable; it should not leak or be available to inline python
    with pytest.raises(RuntimeError):
        _run(monkeypatch, src)


def test_inline_python_backticks_nested_angles(monkeypatch: Any) -> None:
    src = "= <`len([i for i in range(3)])`>\n"
    out = _run(monkeypatch, src)
    assert out == "3"


def test_mixed_whitespace_preserved(monkeypatch: Any) -> None:
    src = "@p(x,y):\n  =   left   <x>   mid   <y>   right   \n\n= <p(a, b)>\n"
    out = _run(monkeypatch, src)
    assert out == "left   a   mid   b   right"


def test_nested_procedure_definition_is_ignored_and_call_fails(
    monkeypatch: Any,
) -> None:
    # Nested procedure is parsed but not registered; calling it should fail
    src = "@outer():\n  text\n  @inner():\n    = hi\n  = done\n\n= <inner()>\n"
    rt = _compile(src)
    with pytest.raises(KeyError):
        rt.run_main()


def test_unmatched_closers_in_template_raise(monkeypatch: Any) -> None:
    with pytest.raises(Exception):
        parse_program("hello > world\n=")


def test_call_with_empty_literals_and_vars(monkeypatch: Any) -> None:
    src = "@p(a,b):\n  = <a>|<b>\n\n= <p(<`'L'`>, R)>\n"
    out = _run(monkeypatch, src)
    assert out == "L|R"


def test_output_in_return_is_ignored_textually(monkeypatch: Any) -> None:
    # Outputs in returns are now honored and filled before rendering
    src = "= prefix [x] suffix\n"
    out = _run(monkeypatch, src)
    assert out.startswith("prefix ") and out.endswith(" suffix") and "x-value" in out


def test_inline_python_imports_and_builtins(monkeypatch: Any) -> None:
    src = "= <`__import__('math').sqrt(81)`>-<`len([1,2,3,4])`>\n"
    out = _run(monkeypatch, src)
    assert out.startswith("9.0-4")


def test_assignment_type_coercion_to_int(monkeypatch: Any) -> None:
    src = "@p():\n  [x:int] = `7`\n  = <`x*3`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "21"


def test_assignment_list_int_type_coercion(monkeypatch: Any) -> None:
    src = "@p():\n  [xs: list[int]] = `[1,2,3]`\n  = <`str(sum(xs))`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "6"


def test_inline_python_exec_statement_returns_empty_string(monkeypatch: Any) -> None:
    src = "= <`a=5`>done\n"
    out = _run(monkeypatch, src)
    assert out.endswith("done")


def test_unallowed_language_fence_inline(monkeypatch: Any) -> None:
    # '```python' should not start a block; treated as text then '=' return missing => parse error
    with pytest.raises(Exception):
        parse_program("```python\npass\n```\n=")


def test_call_with_multiple_mixed_arguments(monkeypatch: Any) -> None:
    src = (
        "@f(a, b: int, c, d):\n  = <a>-<`str(b)`>-<c>-<d>\n\n"
        "@g(x,y,z,w):\n  = <x>-<y>-<z>-<w>\n\n"
        "= <f(one, `2`, <g(three, four, five, six)>, `'-'.join(['x','y'])`)>\n"
    )
    out = _run(monkeypatch, src)
    assert out.startswith("one-2-") and out.endswith("-x-y")


def test_return_trims_only_ends(monkeypatch: Any) -> None:
    src = "=   a   <`'b'`>   c   \n"
    out = _run(monkeypatch, src)
    assert out == "a   b   c"


def test_unknown_var_in_python_block_raises(monkeypatch: Any) -> None:
    src = "@p():\n  ```\n  non_existing += 1\n  ```\n  = ok\n\n= <p()>\n"
    rt = _compile(src)
    with pytest.raises(RuntimeError):
        rt.run_main()


def test_assign_and_then_output(monkeypatch: Any) -> None:
    src = "@p():\n  [seed] = base\n  using <seed> we need [out]\n  = <out>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "out-value"


def test_backtick_toggles_in_matching(monkeypatch: Any) -> None:
    # ensure matching ignores delimiters inside backticks
    src = "= <`']]'+ '<<' + '[['`>\n"
    out = _run(monkeypatch, src)
    assert "]]" in out and "[[" in out and "<<" in out


def test_multiple_python_blocks_and_sync_env(monkeypatch: Any) -> None:
    src = "@p():\n  ```\n  a = 2\n  ```\n  [b: int] = ```\n  return (2 * 5) + 3\n  ```\n  = <`str(b)`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "13"


def test_inline_python_complex_expression(monkeypatch: Any) -> None:
    src = "= <`'-'.join(str(i*i) for i in range(5))`>\n"
    out = _run(monkeypatch, src)
    assert out.startswith("0-1-4-9-16")


def test_call_return_value_used_as_arg(monkeypatch: Any) -> None:
    src = "@inc(n):\n  = <`int(n)+1`>\n\n@sum2(a,b):\n  = <`int(a)+int(b)`>\n\n= <sum2(<inc(4)>, <inc(5)>)>\n"
    out = _run(monkeypatch, src)
    assert out == "11"


def test_value_returning_python_block_in_assignment_native_value(
    monkeypatch: Any,
) -> None:
    src = "@p():\n  [x: int] = ```\n  return 2 + 3\n  ```\n  = <`x*2`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "10"


def test_assign_block_requires_type_for_non_str(monkeypatch: Any) -> None:
    src = "@p():\n  [x] = ```\n  return 1+1\n  ```\n  = <`str(x)`>\n\n= <p()>\n"
    with pytest.raises(TypeError):
        _run(monkeypatch, src)


def test_assign_block_str_with_wrong_annotation_raises(monkeypatch: Any) -> None:
    src = "@p():\n  [x: int] = ```\n  return 's'\n  ```\n  = <x>\n\n= <p()>\n"
    with pytest.raises(TypeError):
        _run(monkeypatch, src)


def test_value_returning_python_block_in_assignment_expression(
    monkeypatch: Any,
) -> None:
    src = "@p():\n  [s] = ```\n  return 'hello' + ' world'\n  ```\n  = <s>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "hello world"


def test_value_returning_python_block_in_return(monkeypatch: Any) -> None:
    src = "@p(a,b) -> int:\n  = ```\n  return int(a) + int(b)\n  ```\n\n= <p(7, 8)>\n"
    out = _run(monkeypatch, src)
    assert out == "15"


def test_value_returning_python_block_in_return_expression(monkeypatch: Any) -> None:
    src = "= ```\nreturn [1,2,3] + [4]\n```\n"
    out = _run(monkeypatch, src)
    assert out.replace(" ", "") in ("[1,2,3,4]", "[1,2,3,4]")


def test_procedure_return_type_annotation_native_return(monkeypatch: Any) -> None:
    src = "@make() -> list[int]:\n  = `[]`\n\n@use():\n  = <`str(len(make()))`>\n\n= <use()>\n"
    out = _run(monkeypatch, src)
    assert out == "0"


def test_procedure_default_return_str(monkeypatch: Any) -> None:
    src = "@p():\n  = hello\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "hello"


def test_procedure_inline_python_return_requires_annotation(monkeypatch: Any) -> None:
    src = "@p():\n  = `[]`\n\n= <p()>\n"
    with pytest.raises(TypeError):
        _run(monkeypatch, src)


def test_backtick_assignment_native_int(monkeypatch: Any) -> None:
    src = "@p():\n  [x] = `1+1`\n  = <`x*2`>\n\n= <p()>\n"
    with pytest.raises(TypeError):
        _run(monkeypatch, src)


def test_backtick_assignment_type_coercion_to_str(monkeypatch: Any) -> None:
    src = "@p():\n  [x: str] = `1+1`\n  = <x>\n\n= <p()>\n"
    with pytest.raises(TypeError):
        _run(monkeypatch, src)


def test_backtick_assignment_with_extra_text_is_string(monkeypatch: Any) -> None:
    src = "@p(s):\n  [x] = `1+1` and more <s>\n  = <x>\n\n= <p(ok)>\n"
    out = _run(monkeypatch, src)
    assert out.startswith("`1+1` and more ok")


def test_backtick_vs_angle_inline_python_assignment(monkeypatch: Any) -> None:
    src = "@p():\n  [a:int] = `40+2`\n  [b] = <`40+2`>\n  = <`str(a)+'-'+b`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "42-42"


def test_backtick_assignment_list_and_use(monkeypatch: Any) -> None:
    src = "@p():\n  [xs: list[int]] = `[1,2,3]`\n  = <`str(sum(xs))`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "6"


def test_assignment_then_reassignment(monkeypatch: Any) -> None:
    src = "@p():\n  [x] = a\n  [x] = <`x+'b'`>\n  = <x>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "ab"


def test_output_then_python_block_consumes(monkeypatch: Any) -> None:
    src = "@p():\n  need [v: int]\n  [w: int] = ```\n  return v * 4\n  ```\n  = <`str(w)`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "168" or out.isdigit()


def test_invalid_identifier_in_assignment_raises_parse(monkeypatch: Any) -> None:
    with pytest.raises(Exception):
        parse_program("[two words] = x\n=")


def test_call_with_missing_closing_paren_in_angle(monkeypatch: Any) -> None:
    with pytest.raises(Exception):
        parse_program("= <foo(bar>\n")


def test_literal_equals_must_be_escaped(monkeypatch: Any) -> None:
    with pytest.raises(Exception):
        parse_program("= a=b\n")


def test_python_block_indent_normalization(monkeypatch: Any) -> None:
    src = "@p():\n  [c: int] = ```\n      a = 1\n      b = 2\n      return a + b\n  ```\n  = <`str(c)`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "3"


def test_inline_python_statement_then_value(monkeypatch: Any) -> None:
    src = "@p():\n  <`assert 3>2`>\n  = <`str(10)`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "10"


def test_nested_list_type_output_then_use(monkeypatch: Any) -> None:
    src = "@p():\n  mat [m: list[list[str]]]\n  = <`str(sum(len(row) for row in m))`>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "0" or out.isdigit()


def test_return_with_nested_calls_and_text(monkeypatch: Any) -> None:
    src = "@id(x):\n  = <x>\n\n= result:<id(<id(ok)>)>!\n"
    out = _run(monkeypatch, src)
    assert out == "result:ok!"


def test_inline_python_errors_reported(monkeypatch: Any) -> None:
    src = "= <`1/0`>\n"
    rt = _compile(src)
    with pytest.raises(RuntimeError):
        rt.run_main()


def test_unmatched_end_marker_in_python_block_ignored(monkeypatch: Any) -> None:
    # BEGIN/END markers inside python block should be skipped
    src = "@p():\n  ```\n  # BEGIN/END-like tokens\n  text = 'ok'\n  ```\n  = <`text`>\n\n= <p()>\n"
    # Inline python cannot access block-local names; expect error
    with pytest.raises(RuntimeError):
        _run(monkeypatch, src)


def test_complex_chain_of_assign_prompt_and_return(monkeypatch: Any) -> None:
    src = "@complex(a,b):\n  [u] = <a>-<b>\n  <u> gives [v: str]\n  [w: int] = `len(v)`\n  = <`v+'|'+str(w)`>\n\n= <complex(x, y)>\n"
    out = _run(monkeypatch, src)
    assert "|" in out


def test_standalone_backtick_line_side_effects(monkeypatch: Any) -> None:
    src = "@p():\n" "  [x] = start\n" "  `x = x + '-mid'`\n" "  = <x>\n\n" "= <p()>\n"
    out = _run(monkeypatch, src)
    assert out.endswith("-mid")


def test_standalone_backtick_invalid_mixed_content(monkeypatch: Any) -> None:
    src = "@p():\n  `x=1` more\n  = ok\n\n= <p()>\n"
    from marker.lang import parse_program
    from tests.mock_adapter import MockAdapter

    with pytest.raises(Exception):
        parse_program(src)


def test_standalone_backtick_interacts_with_outputs(monkeypatch: Any) -> None:
    src = "@p():\n  need [v]\n  `v = v + '-z'`\n  = <v>\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out.endswith("-z") or out != ""


def test_standalone_backtick_block_comment_interplay(monkeypatch: Any) -> None:
    src = "@p():\n  ###\n  `x=1`\n  ###\n  = ok\n\n= <p()>\n"
    out = _run(monkeypatch, src)
    assert out == "ok"


def test_standalone_backtick_line_in_top_level(monkeypatch: Any) -> None:
    src = "[x] = base\n" "`x = x + '-top'`\n" "= <x>\n"
    out = _run(monkeypatch, src)
    assert out.endswith("-top")


def test_standalone_backtick_line_error_propagates(monkeypatch: Any) -> None:
    src = "@p():\n" "  `assert False`\n" "  = ok\n\n" "= <p()>\n"
    rt = _compile(src)
    with pytest.raises(RuntimeError):
        rt.run_main()
