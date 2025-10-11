from __future__ import annotations

from typing import Any, Dict, List

import pytest

import marker_pattern.core as core
from marker_pattern.lang import compile_program, parse_program


class _SingleFakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _SingleFakeAgent:
    def __init__(
        self, model: str | None = None, *, output_type: Any = None, **_: Any
    ) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _SingleFakeRunResult:  # type: ignore[override]
        data: Dict[str, Any] = {}

        # TypedDict-aware filling
        annotations: Dict[str, Any] = {}
        ty = self.output_type
        if hasattr(ty, "__annotations__") and isinstance(ty.__annotations__, dict):  # type: ignore[attr-defined]
            annotations = ty.__annotations__  # type: ignore[attr-defined]

        # Parse required keys from question prompt
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]

        # No description heuristics; tests rely on required keys only

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
# MarkerRuntime core API
# ----------------------
def test_runtime_city_country(monkeypatch: Any) -> None:
    import marker_pattern.core as core_mod

    monkeypatch.setattr(core_mod, "Agent", _SingleFakeAgent, raising=True)

    rt = core.MarkerRuntime()

    @rt.procedure
    def foo(city: str):  # type: ignore[no-redef]
        rt.m([rt.i("city"), " is located in ", rt.o("country"), ". "])  # type: ignore[name-defined]
        return rt.i("country")  # type: ignore[name-defined]

    @rt.main
    def main():  # type: ignore[no-redef]
        return rt.c("foo", "Sivas")  # type: ignore[name-defined]

    out = rt.run_main()
    assert isinstance(out, str)
    assert out == "country-value"


def test_runtime_nested_calls(monkeypatch: Any) -> None:
    import marker_pattern.core as core_mod

    monkeypatch.setattr(core_mod, "Agent", _SingleFakeAgent, raising=True)
    rt = core.MarkerRuntime()

    @rt.procedure
    def get_country_info(city: str):  # type: ignore[no-redef]
        rt.m([rt.i("city"), " is located in ", rt.o("country")])  # type: ignore[name-defined]
        return rt.i("country")  # type: ignore[name-defined]

    @rt.procedure
    def get_language(country: str):  # type: ignore[no-redef]
        rt.m(["The primary language spoken in ", rt.i("country"), " is ", rt.o("language")])  # type: ignore[name-defined]
        return rt.i("language")  # type: ignore[name-defined]

    @rt.procedure
    def translate_greeting(language: str):  # type: ignore[no-redef]
        rt.m(["'Hello' in ", rt.i("language"), " is ", rt.o("greeting")])  # type: ignore[name-defined]
        return rt.i("greeting")  # type: ignore[name-defined]

    @rt.main
    def main():  # type: ignore[no-redef]
        return rt.c("translate_greeting", rt.c("get_language", rt.c("get_country_info", "Paris")))  # type: ignore[name-defined]

    out = rt.run_main()
    assert isinstance(out, str)
    assert out == "greeting-value"


# ----------------------
# Functional DSL suite
# ----------------------
CASES: List[tuple[str, str]] = [
    ("= Paris\n", "Paris"),
    ("@echo(x):\n  = <x>\n\n= <echo(alpha)>\n", "alpha"),
    ("@p(x):\n  generate [y] from <x>\n  = <y>\n\n= <p(foo)>\n", "y-value"),
    (
        "@p(x):\n  generate [a], [b] for <x>\n  = <a>-<b>\n\n= <p(z)>\n",
        "a-value-b-value",
    ),
    (
        "@inner(x):\n  = <x>\n\n@outer(y):\n  = P(<y>)\n\n= <outer(<inner(beta)>)>\n",
        "P(beta)",
    ),
    (
        "@title(name):\n  = Dr. <name>\n\n=   Report:   <title(who)>   \n",
        "Report:   Dr. who",
    ),
    ("= Use <<hello>> literally\n", "Use <hello> literally"),
    ("= Avoid [[square]] here\n", "Avoid [square] here"),
    ("= not an == assignment\n", "not an = assignment"),
    ("= mention @@symbol\n", "mention @symbol"),
    (
        "@concat(s):\n  = <s>\n\n= <concat(alpha,, beta,, gamma)>\n",
        "alpha, beta, gamma",
    ),
    ("@pair(a,b):\n  = <a>|<b>\n\n= <pair(foo,bar)>\n", "foo|bar"),
    ("@q(c):\n  <c> yields [r]\n  now we use <r>\n  = <r>\n\n= <q(city)>\n", "r-value"),
    (
        "@chain(s):\n  make [x] from <s>\n  then make [y] from <x>\n  = <y>\n\n= <chain(seed)>\n",
        "y-value",
    ),
    ("=    A     B   C     \n", "A     B   C"),
    (
        "@r(a,b):\n  =  left   <a>   mid   <b>   right   \n\n= <r(x, y)>\n",
        "left   x   mid   y   right",
    ),
    (
        "@mk():\n  produce [v]\n  = <v>\n\n@use():\n  = pre-<mk()>-post\n\n= <use()>\n",
        "pre-v-value-post",
    ),
    (
        "@make():\n  [a] and [b]\n  = <a>/<b>\n\n@use():\n  = (<make()>)\n\n= <use()>\n",
        "(a-value/b-value)",
    ),
    ("@id(x):\n  = <x>\n\n@wrap(x):\n  = [[<x>]]\n\n= <wrap(<id(num)>)>\n", "[num]"),
    (
        "@calc(a,b):\n  get [sum] from <a> and <b>\n  get [prod] from <a> and <b>\n  = <sum>-<prod>\n\n= <calc(2,3)>\n",
        "sum-value-prod-value",
    ),
    (
        "= angle << and bracket [[ and at @@ and eq == and comma \\\\ ,\n",
        "angle < and bracket [ and at @ and eq = and comma \ ,",
    ),
    (
        "@inner(x):\n  = <x>!\n\n@outer(a,b):\n  = <a>-<b>\n\n= <outer(alpha, <inner(beta)>)>\n",
        "alpha-beta!",
    ),
    (
        "@proc(s):\n  prepare [out] from <s>\n  using <out> now\n  = done <out>\n\n= <proc(z)>\n",
        "done out-value",
    ),
    (
        "@f1(x):\n  = <x>1\n\n@f2(x):\n  = <x>2\n\n@f3(x):\n  = <x>3\n\n= <f3(<f2(<f1(a)>)>)>\n",
        "a123",
    ),
    (
        "@e(s):\n  = <s>\n\n= <e(alice,, bob,, and charlie)>\n",
        "alice, bob, and charlie",
    ),
    ("@g():\n  = one\n  = two\n\n= <g()>\n", "two"),
    ("@h():\n  set [a]\n  set [b]\n  = <b>\n\n= <h()>\n", "b-value"),
    ("@s(x):\n  override [x]\n  = <x>\n\n= <s(orig)>\n", "x-value"),
    (
        "@k(a,b):\n  = <a>+<b>\n\n@z(x):\n  = <k(alpha, <x>)>\n\n= <z(beta)>\n",
        "alpha+beta",
    ),
    ("=   hello   world    \n", "hello   world"),
    ("@m():\n  A [x] B [y] C\n  = <x>|<y>\n\n= <m()>\n", "x-value|y-value"),
    ("@q(x):\n  = (<x>)\n\n= result:<q(ok)>\n", "result:(ok)"),
]


@pytest.mark.parametrize("src,expected", CASES)
def test_functional_cases(monkeypatch: Any, src: str, expected: str) -> None:
    out = _run(monkeypatch, src)
    assert out == expected


# ----------------------
# Spec: valid and invalid
# ----------------------
def test_valid_basic_paris(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@get_country(city):\n"
            "  <city> is located in [country].\n"
            "  = <country>\n\n"
            "= <get_country(Paris)>\n"
        ),
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_valid_nested_calls(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@get_country_info(city):\n"
            "  <city> is located in [country].\n"
            "  = <country>\n\n"
            "@get_language(country):\n"
            "  The primary language spoken in <country> is [language].\n"
            "  = <language>\n\n"
            "@translate_greeting(language):\n"
            "  'Hello' in <language> is [greeting]\n"
            "  = <greeting>\n\n"
            "= <translate_greeting(<get_language(<get_country_info(Paris)>)>)>\n"
        ),
    )
    assert isinstance(out, str)
    assert len(out) > 0


invalid_sources = [
    "= <x",
    "= [x",
    "[42]",
    "[foo(bar)]",
    "@foo(x)\n  = x",
    "@foo x):\n  = x",
    "@foo(x:\n  = x",
    "= <foo(alpha, <bar(x)>",
    "= <foo(alpha, <bar(x)>]]>",
    "= [bad-name]",
    "= []",
    "= text >",
    "= text ]",
    "[not-valid!]",
    "[two words]",
    "@proc(x: \n  = x",
    "= <foo(x) ",
    "= [a <x>]",
    "= <foo(alpha, <bar(x>)>",
    "= <x>>",
    "@",
    "@ (x):\n  = x",
    "[]",
    "[call(<x>)]",
]


@pytest.mark.parametrize("src", invalid_sources)
def test_invalid_sources_raise(src: str) -> None:
    with pytest.raises(Exception):
        parse_program(src)


# ----------------------
# Inline python features
# ----------------------
def test_inline_python_with_variables(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@add(x,y):\n"
            "  <x> + <y> is [s]; computed: <`int(x)+int(y)`>\n"
            "  = <`int(x)+int(y)`>\n\n"
            "= <add(2, 3)>\n"
        ),
    )
    assert out == "5"


def test_typed_params_basic_sum(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@add(x: int, y: int):\n" "  = <`x + y`>\n\n" "= <add(`2`, `3`)>\n"),
    )
    assert out == "5"


def test_typed_params_string_default_and_type_mismatch(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    rt = _compile("@concat(a, b):\n  = <`a + b`>\n\n= <concat(hello, world)>\n")
    out = rt.run_main()
    assert out == "helloworld"
    # Now mismatch: passing int to int-typed param is ok; passing str to int-typed should raise
    src = "@add(x: int, y: int):\n  = <`x + y`>\n\n= <add(alpha, 3)>\n"
    rt2 = _compile(src)
    with pytest.raises(TypeError):
        rt2.run_main()


def test_inline_python_with_procedure_calls(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@times2(n: int):\n"
            "  = <`int(n)*2`>\n\n"
            "@calc(a,b):\n"
            "  double of <a> is <`times2(int(a))`> and sum is <`int(a)+int(b)`>\n"
            "  = result:<`times2(int(a)+int(b))`>\n\n"
            "= <calc(4, 5)>\n"
        ),
    )
    assert out == "result:18"


def test_inline_python_chaining_llm_outputs(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@chain():\n"
            "  get [x]\n"
            "  then compute <`len(x)`> and set [y]\n"
            "  = <`str(len(x)) + '-' + str(len(y))`>\n\n"
            "= <chain()>\n"
        ),
    )
    assert out == "7-7" or out.endswith("-0")


def test_inline_python_with_builtins_and_import(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@builtin_demo():\n"
            "  = <`len([1,2,3])`>-<`__import__('math').sqrt(16)`>\n\n"
            "= <builtin_demo()>\n"
        ),
    )
    assert out.startswith("3-4.0")


def test_inline_python_as_call_arguments(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@concat(a: int,b: int):\n"
            "  = <a>+<b>\n\n"
            "@sum_and_concat(x,y):\n"
            "  = <concat(`int(x)+1`, `int(y)+2`)>\n\n"
            "= <sum_and_concat(5, 7)>\n"
        ),
    )
    assert out == "6+9"


def test_inline_python_nested_call_arguments(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@wrap(v):\n"
            "  = [[<v>]]\n\n"
            "@compose(a,b):\n"
            "  = <wrap(<`str(int(a)*int(b))`>)>\n\n"
            "= <compose(3, 4)>\n"
        ),
    )
    assert out == "[12]"


def test_inline_python_with_comparison_and_assert(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@f():\n" "  = <`1 + 1`>\n\n" "= <`assert 5 > 3`>\n"),
    )
    assert out == ""


def test_inline_python_assert_failure_in_return(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    rt = _compile("= <`assert False`>\n")
    with pytest.raises(RuntimeError):
        rt.run_main()


def test_inline_python_assert_failure_in_template_line(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = "@p():\n" "  <`assert False`>\n" "  = ok\n\n" "= <p()>\n"
    rt = _compile(src)
    with pytest.raises(RuntimeError):
        rt.run_main()


# ----------------------
# Python block features
# ----------------------
def test_python_block_sets_variables(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@calc(a,b):\n"
            "  [s: int] = ```\n  return int(a) + int(b)\n  ```\n"
            "  [t: int] = ```\n  return int(a) * int(b)\n  ```\n"
            "  = sum:<`s`> prod:<`t`>\n\n"
            "= <calc(6, 7)>\n"
        ),
    )
    assert out == "sum:13 prod:42"


def test_python_block_can_call_procedures(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@times2(n):\n"
            "  = <`int(n)*2`>\n\n"
            "@use(a):\n"
            "  [doubled: int] = ```\n  return times2(a)\n  ```\n"
            "  [tripled: int] = ```\n  return int(a) * 3\n  ```\n"
            "  = <`str(doubled)+'-'+str(tripled)`>\n\n"
            "= <use(5)>\n"
        ),
    )
    assert out == "10-15"


def test_python_block_with_llm_outputs(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@pipeline(x):\n"
            "  from <x> we ask for [y]\n"
            "  [z: str] = ```\n  return 'prefix-' + y\n  ```\n"
            "  final [out] uses <`z`>\n"
            "  = <out>\n\n"
            "= <pipeline(alpha)>\n"
        ),
    )
    assert out.startswith("out-value") or out.startswith("prefix-")


# ----------------------
# Assignment features
# ----------------------
def test_assign_basic(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@p(x):\n" "  [y] = <x>\n" "  = <y>\n\n" "= <p(alpha)>\n"),
    )
    assert out == "alpha"


def test_assign_with_call(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@id(x):\n  = <x>\n\n@p():\n  [v] = <id(beta)>\n  = <v>\n\n= <p()>\n"),
    )
    assert out == "beta"


def test_assign_then_llm(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@p(city):\n"
            "  [c] = <city>\n"
            "  <c> is located in [country] \n"
            "  = <country>\n\n"
            "= <p(Paris)>\n"
        ),
    )
    assert out == "country-value"


def test_assign_overwrite(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@p():\n  [x] = alpha\n  [x] = beta\n  = <x>\n\n= <p()>\n"),
    )
    assert out == "beta"


def test_assign_inline_python_rhs(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@p(a,b):\n  [s] = <`str(a)+str(b)`>\n  = <s>\n\n= <p(1, 2)>\n"),
    )
    assert out == "12"


def test_assign_works_before_return(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@p():\n  [x] = hello\n  = world <x>\n\n= <p()>\n"),
    )
    assert out == "world hello"


def test_assign_interacts_with_outputs(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@p():\n  gain [a]\n  [b] = <a>\n  = <b>\n\n= <p()>\n"),
    )
    assert out == "a-value"


# ----------------------
# Type annotation features
# ----------------------
def test_typed_output_string(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        "@p():\n  city is [city: str]\n  = <city>\n\n= <p()>\n",
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_typed_output_list_of_str(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        "@p(country):\n  neighbors: [neigh: list[str]]\n  = <`','.join(neigh)`>\n\n= <p(Turkey)>\n",
    )
    assert "," in out or out == ""


def test_typed_assignment_int(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        "@p():\n  [x: int] = <`1+2+3`>\n  = <`x*2`>\n\n= <p()>\n",
    )
    assert out == "12"


def test_typed_assignment_list_int(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        "@p():\n  [nums: list[int]] = `[1,2,3]`\n  = <`sum(nums)`>\n\n= <p()>\n",
    )
    assert out == "6"


def test_mixed_typed_and_untyped_outputs(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@p():\n"
            "  get [a: int] and [b]\n"
            "  = <`str(a)+'-'+str(len(str(b)))`>\n\n"
            "= <p()>\n"
        ),
    )
    assert "-" in out


def test_typed_output_used_in_python_block(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@p():\n"
            "  numbers [arr: list[int]]\n"
            "  [s: int] = ```\n  return sum(arr)\n  ```\n"
            "  = <`str(s)`>\n\n"
            "= <p()>\n"
        ),
    )
    assert out.isdigit()


def test_typed_list_cross_call_and_inline(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@id(x):\n  = <x>\n\n"
            "@p():\n  items [xs: list[str]] from <id(seed)>\n  [ys: list[str]] = <`[v+'!' for v in xs]`>\n  = <`'-'.join(ys)`>\n\n"
            "= <p()>\n"
        ),
    )
    assert isinstance(out, str)


def test_typed_assignment_then_llm(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@p():\n"
            "  [prefix: str] = pre-\n"
            "  name is [name: str]\n"
            "  = <prefix><name>\n\n"
            "= <p()>\n"
        ),
    )
    assert isinstance(out, str)


def test_typed_outputs_multiple(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@p():\n"
            "  need [a: int], [b: float], and [c: bool]\n"
            "  = <a>-<`int(b)`>-<`'T' if c else 'F'`>\n\n"
            "= <p()>\n"
        ),
    )
    assert out.count("-") == 2


def test_typed_output_in_return_context(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        (
            "@mk():\n  produce [vals: list[int]]\n  = <`str(sum(vals))`>\n\n"
            "= <mk()>\n"
        ),
    )
    assert out.isdigit()


def test_invalid_type_name_falls_back_to_str(monkeypatch: Any) -> None:
    out = _run(
        monkeypatch,
        ("@p():\n  value [x: NotARealType]\n  = <x>\n\n= <p()>\n"),
    )
    assert isinstance(out, str)


@pytest.mark.parametrize(
    "src",
    [
        "@p():\n [x: int] = `2**5`\n = <`x+1`>\n\n= <p()>\n",
        "@p():\n [s: str] = hello\n = <s>!\n\n= <p()>\n",
        "@p():\n arr [a: list[int]]\n [first: int] = ```\n return a[0] if a else 0\n ```\n = <`str(first)`>\n\n= <p()>\n",
        "@times2(n):\n = <`int(n)*2`>\n\n@use():\n list [xs: list[int]]\n [ys: list[int]] = `[2,4]`\n = <`str(sum(int(v) for v in ys))`>\n\n= <use()>\n",
        "@p():\n alpha [a] and ints [b: list[int]]\n = <a>-<`sum(b)`>\n\n= <p()>\n",
    ],
)
def test_parametrized_typed_snippets(monkeypatch: Any, src: str) -> None:
    out = _run(monkeypatch, src)
    assert isinstance(out, str)
