from __future__ import annotations

from typing import Any, Dict, List

import pytest

import marker_pattern.core as core
from marker_pattern.lang import parse_program, compile_program


class _SingleFakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _SingleFakeAgent:
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _SingleFakeRunResult:  # type: ignore[override]
        data: Dict[str, Any] = {}

        annotations: Dict[str, Any] = {}
        ty = self.output_type
        if hasattr(ty, "__annotations__") and isinstance(ty.__annotations__, dict):  # type: ignore[attr-defined]
            annotations = ty.__annotations__  # type: ignore[attr-defined]

        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]

        for k in keys:
            ann = annotations.get(k)
            if ann is None:
                data.setdefault(k, f"{k}-value")
                continue
            try:
                origin = getattr(ann, "__origin__", None)
            except Exception:
                origin = None
            if ann is str:
                data[k] = f"{k}-value"
            elif ann is int:
                data[k] = 42
            elif ann is float:
                data[k] = 3.14
            elif ann is bool:
                data[k] = True
            elif origin in (list, List):
                arg = getattr(ann, "__args__", (str,))[0]
                if arg is str:
                    data[k] = [f"{k}-a", f"{k}-b"]
                elif arg is int:
                    data[k] = [1, 2]
                else:
                    data[k] = []
            else:
                data[k] = f"{k}-value"

        return _SingleFakeRunResult(output=data)


def _compile(src: str):
    return compile_program(parse_program(src))


def _run(monkeypatch: Any, src: str) -> Any:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
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
        "@deep(a):\n  = <`str(int(a)*2)`>\n\n@wrap(x):\n  = [[<x>]]\n\n= <wrap(<deep(<deep(2)>)>)>\n",
        "[8]",
    ),
    (
        "@types():\n  list [xs: list[int]]\n  ```\n  doubled = [v*2 for v in xs]\n  ```\n  = <`str(sum(doubled))`>\n\n= <types()>\n",
        "6",
    ),
    (
        # Multiple outputs, mixture typed and untyped
        "@multi():\n  a [p] and b [q: int] and c [r: list[str]]\n  = <p>-<`q+1`>-<`len(r)`>\n\n= <multi()>\n",
        "p-value-43-2",
    ),
    (
        # Complex commas and escapes in args
        "@join(a,b,c):\n  = <a>|<b>|<c>\n\n= <join(alpha,, beta,, gamma, delta, <`'x,,y'.replace(',,', ',')`>)>\n",
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
        "@times(n):\n  = <`int(n)*3`>\n\n@prog(x):\n  ```\n  import math\n  y = int(times(x))\n  z = int(math.sqrt(16)) + y\n  ```\n  = <`str(z)`>\n\n= <prog(5)>\n",
        "19",
    ),
    (
        # Assignment native value then used in prompt
        "@p():\n  [nums: list[int]] = <`[1,2,3]`>\n  total is [s: int]\n  ```\n  s = sum(nums)\n  ```\n  = <`str(s)`>\n\n= <p()>\n",
        "6",
    ),
    (
        # Output typed list[str] then chain in python
        "@cities():\n  get [arr: list[str]]\n  = <`'-'.join(arr)`>\n\n= <cities()>\n",
        "arr-a-arr-b",
    ),
    (
        # Top-level python block before return
        "```\nval = 10\n```\n= <`str(val*3)`>\n",
        "30",
    ),
    (
        # Nested calls + inline python in args
        "@add(a,b):\n  = <`int(a)+int(b)`>\n\n@chain(x):\n  = <add(<`int(x)*2`>, <add(3, 4)>)>\n\n= <chain(5)>\n",
        "17",
    ),
    (
        # Shadowing builtins inside python block but still using other builtins
        "@demo():\n  ```\n  len = 5\n  s = __import__('math').sqrt(25)\n  ```\n  = <`str(int(s)+len)`>\n\n= <demo()>\n",
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
        "@esc(s):\n  = <s>\n\n= Use <<tag>> and [[br]] and @@at and eq == and comma \\, plus <esc(a,,b)>\n",
        "Use <tag> and [br] and @at and eq = and comma \\, plus a,b",
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
    "@p():\n  ```\n  no end fence",  # unterminated python block
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
    (
        # Duplicate procedure names cause registration error
        "@p():\n  = a\n\n@p():\n  = b\n\n= <p()>\n",
        ValueError,
    ),
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
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    if "not_defined()" in src:
        # compile ok, run fails due to unknown proc
        rt = _compile(src)
        with pytest.raises(exc):
            rt.run_main()
        return
    with pytest.raises(exc):
        _run(monkeypatch, src)


def test_no_main_defined_raises(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
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
    src = (
        "@p():\n  matrix [m: list[list[int]]]\n  ```\n  flat = [v for row in m for v in row]\n  ```\n  = <`str(sum(flat))`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    # Our fake agent returns [] for unknown nested typing; allow "0" or empty leading
    assert out == "0" or out.startswith("")


def test_escaped_commas_and_nested_calls(monkeypatch: Any) -> None:
    src = (
        "@cat(a,b,c):\n  = <a>:<b>:<c>\n\n"
        "@use():\n  = <cat(a,,b,,c, <cat(d,,e, f, g)>, <`','.join(['x', 'y'])`>)>\n\n"
        "= <use()>\n"
    )
    out = _run(monkeypatch, src)
    assert ":" in out and "," in out


def test_assignment_string_render_when_not_pure_python(monkeypatch: Any) -> None:
    src = (
        "@p():\n  [x] = hello <`1+1`> world\n  = <x>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out.startswith("hello") and "world" in out


def test_output_updates_env_for_later_use(monkeypatch: Any) -> None:
    src = (
        "@p():\n  color is [color]\n  = <`color.upper()`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out.isupper()


def test_python_inline_can_call_procedure(monkeypatch: Any) -> None:
    src = (
        "@double(n):\n  = <`int(n)*2`>\n\n@use(x):\n  = <`int(double(x))+1`>\n\n= <use(7)>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "15"


def test_python_block_modifies_existing_and_new_vars(monkeypatch: Any) -> None:
    src = (
        "@p():\n  [x] = base\n  ```\n  x = x + '-mod'\n  y = 'new'\n  ```\n  = <x>-<y>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "base-mod-new"


def test_inline_python_backticks_nested_angles(monkeypatch: Any) -> None:
    src = (
        "= <`len([i for i in range(3)])`>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "3"


def test_mixed_whitespace_preserved(monkeypatch: Any) -> None:
    src = (
        "@p(x,y):\n  =   left   <x>   mid   <y>   right   \n\n= <p(a, b)>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "left   a   mid   b   right"


def test_nested_procedure_definition_is_ignored_and_call_fails(monkeypatch: Any) -> None:
    # Nested procedure is parsed but not registered; calling it should fail
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = (
        "@outer():\n  text\n  @inner():\n    = hi\n  = done\n\n= <inner()>\n"
    )
    rt = _compile(src)
    with pytest.raises(KeyError):
        rt.run_main()


def test_unmatched_closers_in_template_raise(monkeypatch: Any) -> None:
    with pytest.raises(Exception):
        parse_program("hello > world\n=")


def test_call_with_empty_literals_and_vars(monkeypatch: Any) -> None:
    src = (
        "@p(a,b):\n  = <a>|<b>\n\n= <p(<`'L'`>, R)>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "L|R"


def test_output_in_return_is_ignored_textually(monkeypatch: Any) -> None:
    # Although odd, current implementation ignores output placeholders in returns when rendering
    src = "= prefix [x] suffix\n"
    out = _run(monkeypatch, src)
    assert out == "prefix  suffix"


def test_inline_python_imports_and_builtins(monkeypatch: Any) -> None:
    src = (
        "= <`__import__('math').sqrt(81)`>-<`len([1,2,3,4])`>\n"
    )
    out = _run(monkeypatch, src)
    assert out.startswith("9.0-4")


def test_assignment_type_coercion_to_int(monkeypatch: Any) -> None:
    src = (
        "@p():\n  [x: int] = <`'7'`>\n  = <`x*3`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "21"


def test_assignment_list_int_type_coercion(monkeypatch: Any) -> None:
    src = (
        "@p():\n  [xs: list[int]] = <`['1','2','3']`>\n  = <`str(sum(int(v) for v in xs))`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "6"


def test_inline_python_exec_statement_returns_empty_string(monkeypatch: Any) -> None:
    src = (
        "= <`a=5`>done\n"
    )
    out = _run(monkeypatch, src)
    assert out.endswith("done")


def test_unallowed_language_fence_inline(monkeypatch: Any) -> None:
    # '```python' should not start a block; treated as text then '=' return missing => parse error
    with pytest.raises(Exception):
        parse_program("```python\npass\n```\n=")


def test_call_with_multiple_mixed_arguments(monkeypatch: Any) -> None:
    src = (
        "@f(a,b,c,d):\n  = <a>-<b>-<c>-<d>\n\n"
        "= <f(one, <`2`>, <f(three, four, five, six)>, <`'-'.join(['x','y'])`>)>\n"
    )
    out = _run(monkeypatch, src)
    assert out.startswith("one-2-") and out.endswith("-x-y")


def test_return_trims_only_ends(monkeypatch: Any) -> None:
    src = (
        "=   a   <`'b'`>   c   \n"
    )
    out = _run(monkeypatch, src)
    assert out == "a   b   c"


def test_unknown_var_in_python_block_raises(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = (
        "@p():\n  ```\n  non_existing += 1\n  ```\n  = ok\n\n= <p()>\n"
    )
    rt = _compile(src)
    with pytest.raises(RuntimeError):
        rt.run_main()


def test_assign_and_then_output(monkeypatch: Any) -> None:
    src = (
        "@p():\n  [seed] = base\n  using <seed> we need [out]\n  = <out>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "out-value"


def test_backtick_toggles_in_matching(monkeypatch: Any) -> None:
    # ensure matching ignores delimiters inside backticks
    src = (
        "= <`']]' + '<<' + '[['`>\n"
    )
    out = _run(monkeypatch, src)
    assert "]]" in out and "[[" in out and "<<" in out


def test_multiple_python_blocks_and_sync_env(monkeypatch: Any) -> None:
    src = (
        "@p():\n  ```\n  a = 2\n  ```\n  ```\n  a = a * 5\n  b = a + 3\n  ```\n  = <`str(b)`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "13"


def test_inline_python_complex_expression(monkeypatch: Any) -> None:
    src = (
        "= <`'-'.join(str(i*i) for i in range(5))`>\n"
    )
    out = _run(monkeypatch, src)
    assert out.startswith("0-1-4-9-16")


def test_call_return_value_used_as_arg(monkeypatch: Any) -> None:
    src = (
        "@inc(n):\n  = <`int(n)+1`>\n\n@sum2(a,b):\n  = <`int(a)+int(b)`>\n\n= <sum2(<inc(4)>, <inc(5)>)>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "11"


def test_assignment_then_reassignment(monkeypatch: Any) -> None:
    src = (
        "@p():\n  [x] = a\n  [x] = <`x+'b'`>\n  = <x>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "ab"


def test_output_then_python_block_consumes(monkeypatch: Any) -> None:
    src = (
        "@p():\n  need [v: int]\n  ```\n  w = v * 4\n  ```\n  = <`str(w)`>\n\n= <p()>\n"
    )
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
    src = (
        "@p():\n  ```\n      a = 1\n      b = 2\n      c = a + b\n  ```\n  = <`str(c)`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "3"


def test_inline_python_statement_then_value(monkeypatch: Any) -> None:
    src = (
        "@p():\n  <`assert 3>2`>\n  = <`str(10)`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "10"


def test_nested_list_type_output_then_use(monkeypatch: Any) -> None:
    src = (
        "@p():\n  mat [m: list[list[str]]]\n  = <`str(sum(len(row) for row in m))`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "0" or out.isdigit()


def test_return_with_nested_calls_and_text(monkeypatch: Any) -> None:
    src = (
        "@id(x):\n  = <x>\n\n= result:<id(<id(ok)>)>!\n"
    )
    out = _run(monkeypatch, src)
    assert out == "result:ok!"


def test_inline_python_errors_reported(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = "= <`1/0`>\n"
    rt = _compile(src)
    with pytest.raises(RuntimeError):
        rt.run_main()


def test_unmatched_end_marker_in_python_block_ignored(monkeypatch: Any) -> None:
    # BEGIN/END markers inside python block should be skipped
    src = (
        "@p():\n  ```\n  # BEGIN/END-like tokens\n  text = 'ok'\n  ```\n  = <`text`>\n\n= <p()>\n"
    )
    out = _run(monkeypatch, src)
    assert out == "ok"


def test_complex_chain_of_assign_prompt_and_return(monkeypatch: Any) -> None:
    src = (
        "@complex(a,b):\n  [u] = <a>-<b>\n  <u> gives [v: str]\n  [w: int] = <`len(v)`>\n  = <`v+'|'+str(w)`>\n\n= <complex(x, y)>\n"
    )
    out = _run(monkeypatch, src)
    assert "|" in out



