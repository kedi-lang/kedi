from __future__ import annotations

from typing import Any, Dict, List, Tuple

import marker_pattern.core as core
import pytest
from marker_pattern.lang import parse_program, compile_program


class _FakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _FakeAgent:
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _FakeRunResult:
        # Deterministic: fill keys from the question prompt like "Fill in the fields: [a, b]."
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
        data: Dict[str, Any] = {k: f"{k}-value" for k in keys}
        return _FakeRunResult(output=data)


def _run(src: str) -> str:
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    return out


CASES: List[Tuple[str, str]] = [
    # 1. simple literal return
    (
        "= Paris\n",
        "Paris",
    ),
    # 2. echo variable through call with literal arg
    (
        "@echo(x):\n  = <x>\n\n= <echo(alpha)>\n",
        "alpha",
    ),
    # 3. single output then return it
    (
        "@p(x):\n  generate [y] from <x>\n  = <y>\n\n= <p(foo)>\n",
        "y-value",
    ),
    # 4. multiple outputs in one line
    (
        "@p(x):\n  generate [a], [b] for <x>\n  = <a>-<b>\n\n= <p(z)>\n",
        "a-value-b-value",
    ),
    # 5. nested calls with literals
    (
        "@inner(x):\n  = <x>\n\n@outer(y):\n  = P(<y>)\n\n= <outer(<inner(beta)>)>\n",
        "P(beta)",
    ),
    # 6. nested calls with literal pass-through
    (
        "@title(name):\n  = Dr. <name>\n\n=   Report:   <title(who)>   \n",
        "Report:   Dr. who",
    ),
    # 7. escaped angle brackets in text
    (
        "= Use <<hello>> literally\n",
        "Use <hello> literally",
    ),
    # 8. escaped brackets in text
    (
        "= Avoid [[square]] here\n",
        "Avoid [square] here",
    ),
    # 9. escaped equals in text
    (
        "= not an == assignment\n",
        "not an = assignment",
    ),
    # 10. escaped at sign in text
    (
        "= mention @@symbol\n",
        "mention @symbol",
    ),
    # 11. escaped comma in arg becomes literal comma
    (
        "@concat(s):\n  = <s>\n\n= <concat(alpha,, beta,, gamma)>\n",
        "alpha, beta, gamma",
    ),
    # 12. two-arg function with literals
    (
        "@pair(a,b):\n  = <a>|<b>\n\n= <pair(foo,bar)>\n",
        "foo|bar",
    ),
    # 13. outputs available across lines
    (
        "@q(c):\n  <c> yields [r]\n  now we use <r>\n  = <r>\n\n= <q(city)>\n",
        "r-value",
    ),
    # 14. outputs used in subsequent output request (chaining)
    (
        "@chain(s):\n  make [x] from <s>\n  then make [y] from <x>\n  = <y>\n\n= <chain(seed)>\n",
        "y-value",
    ),
    # 15. return preserves internal spaces, trims ends
    (
        "=    A     B   C     \n",
        "A     B   C",
    ),
    # 16. substitution with preserved spaces
    (
        "@r(a,b):\n  =  left   <a>   mid   <b>   right   \n\n= <r(x, y)>\n",
        "left   x   mid   y   right",
    ),
    # 17. nested call in return with outputs
    (
        "@mk():\n  produce [v]\n  = <v>\n\n@use():\n  = pre-<mk()>-post\n\n= <use()>\n",
        "pre-v-value-post",
    ),
    # 18. multiple outputs and nested call combination
    (
        "@make():\n  [a] and [b]\n  = <a>/<b>\n\n@use():\n  = (<make()>)\n\n= <use()>\n",
        "(a-value/b-value)",
    ),
    # 19. produce literal brackets around a value using escaping
    (
        "@id(x):\n  = <x>\n\n@wrap(x):\n  = [[<x>]]\n\n= <wrap(<id(num)>)>\n",
        "[num]",
    ),
    # 20. multiple params and outputs reused
    (
        "@calc(a,b):\n  get [sum] from <a> and <b>\n  get [prod] from <a> and <b>\n  = <sum>-<prod>\n\n= <calc(2,3)>\n",
        "sum-value-prod-value",
    ),
    # 21. text with escaped mix
    (
        "= angle << and bracket [[ and at @@ and eq == and comma \\,,\n",
        "angle < and bracket [ and at @ and eq = and comma \\,",
    ),
    # 22. call with nested call and literal
    (
        "@inner(x):\n  = <x>!\n\n@outer(a,b):\n  = <a>-<b>\n\n= <outer(alpha, <inner(beta)>)>\n",
        "alpha-beta!",
    ),
    # 23. outputs referenced before return
    (
        "@proc(s):\n  prepare [out] from <s>\n  using <out> now\n  = done <out>\n\n= <proc(z)>\n",
        "done out-value",
    ),
    # 24. deep nested calls
    (
        "@f1(x):\n  = <x>1\n\n@f2(x):\n  = <x>2\n\n@f3(x):\n  = <x>3\n\n= <f3(<f2(<f1(a)>)>)>\n",
        "a123",
    ),
    # 25. literal commas in arg
    (
        "@e(s):\n  = <s>\n\n= <e(alice,, bob,, and charlie)>\n",
        "alice, bob, and charlie",
    ),
    # 26. multiline procedure with only returns
    (
        "@g():\n  = one\n  = two\n\n= <g()>\n",
        "two",
    ),
    # 27. outputs on multiple lines, last used
    (
        "@h():\n  set [a]\n  set [b]\n  = <b>\n\n= <h()>\n",
        "b-value",
    ),
    # 28. variable shadowing by outputs
    (
        "@s(x):\n  override [x]\n  = <x>\n\n= <s(orig)>\n",
        "x-value",
    ),
    # 29. combine literal and variable in nested call
    (
        "@k(a,b):\n  = <a>+<b>\n\n@z(x):\n  = <k(alpha, <x>)>\n\n= <z(beta)>\n",
        "alpha+beta",
    ),
    # 30. trailing spaces trimmed only at ends
    (
        "=   hello   world    \n",
        "hello   world",
    ),
    # 31. outputs embedded between text segments
    (
        "@m():\n  A [x] B [y] C\n  = <x>|<y>\n\n= <m()>\n",
        "x-value|y-value",
    ),
    # 32. call-return used in text
    (
        "@q(x):\n  = (<x>)\n\n= result:<q(ok)>\n",
        "result:(ok)",
    ),
]


@pytest.mark.parametrize("src,expected", CASES)
def test_functional_cases(monkeypatch: Any, src: str, expected: str) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    out = _run(src)
    assert out == expected


