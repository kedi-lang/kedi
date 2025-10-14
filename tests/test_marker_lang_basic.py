from __future__ import annotations

from typing import Any, List

import pytest

import marker.core as core
from marker.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter
from marker.tests_runner import run_tests, run_evals



def _compile(src: str):
    return compile_program(parse_program(src), adapter=MockAdapter())


def _run(monkeypatch: Any, src: str) -> Any:
    rt = _compile(src)
    out = rt.run_main()
    assert isinstance(out, str)
    return out


# ----------------------
# MarkerRuntime core API
# ----------------------
def test_runtime_city_country(monkeypatch: Any) -> None:
    import marker.core as core_mod

    rt = core.MarkerRuntime(adapter=MockAdapter())

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
    import marker.core as core_mod

    rt = core.MarkerRuntime(adapter=MockAdapter())

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
    ("= Use \\<hello\\> literally\n", "Use <hello> literally"),
    ("= Avoid \\[square\\] here\n", "Avoid [square] here"),
    ("= not an \\= assignment\n", "not an = assignment"),
    ("= mention \\@symbol\n", "mention @symbol"),
    (
        "@concat(s):\n  = <s>\n\n= <concat(alpha\\, beta\\, gamma)>\n",
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
    ("@id(x):\n  = <x>\n\n@wrap(x):\n  = \\[<x>\\]\n\n= <wrap(<id(num)>)>\n", "[num]"),
    (
        "@calc(a,b):\n  get [sum] from <a> and <b>\n  get [prod] from <a> and <b>\n  = <sum>-<prod>\n\n= <calc(2,3)>\n",
        "sum-value-prod-value",
    ),
    (
        "= angle \\<tag\\> and bracket \\[br\\] and at \\@ and eq \\= and comma \\\\ ,\n",
        "angle <tag> and bracket [br] and at @ and eq = and comma \\ ,",
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
        "@e(s):\n  = <s>\n\n= <e(alice\\, bob\\, and charlie)>\n",
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


def test_validation_blocks_parse_and_run(monkeypatch: Any) -> None:
    src = (
        "@get_cities_in(country) -> list[str]:\n"
        "  All cities in <country> are [cities: list[str]].\n"
        "  = `cities`\n\n"
        "@test: get_cities_in:\n"
        "  > case: tr_cities_len:\n"
        "    `assert len(get_cities_in('Turkey')) >= 0`\n"
        "  > case: sg_cities_len:\n"
        "    `assert len(get_cities_in('Singapore')) >= 0`\n\n"
        "@eval: get_cities_in:\n"
        "  > metric: num:\n"
        "    = `1.5`\n"
    )
    tests = run_tests(src, adapter=MockAdapter())
    assert tests and all(c.success for s in tests for c in s.cases)
    evals = run_evals(src, adapter=MockAdapter())
    assert evals and all(m.score >= 0 for s in evals for m in s.metrics)


def test_validation_companion_file_discovery(tmp_path: Any, monkeypatch: Any) -> None:
    p = tmp_path / "prog.marker"
    pv = tmp_path / "prog.val.marker"
    p.write_text("@id(x):\n  = <x>\n\n= <id(ok)>\n", encoding="utf-8")
    pv.write_text("@test: id:\n  > case: ok:\n    `assert id('x') == 'x'`\n\n@eval: id:\n  > metric: basic:\n    = `1.0`\n", encoding="utf-8")
    tests = run_tests(str(p), adapter=MockAdapter())
    assert tests and tests[0].cases[0].success
    evals = run_evals(str(p), adapter=MockAdapter())
    assert evals and evals[0].metrics[0].score == 1.0


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
            "  = \\[<v>\\]\n\n"
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
    rt = _compile("= <`assert False`>\n")
    with pytest.raises(RuntimeError):
        rt.run_main()


def test_inline_python_assert_failure_in_template_line(monkeypatch: Any) -> None:
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
            "@times2(n) -> int:\n"
            "  = `int(n)*2`\n\n"
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
        "@p():\n  [x: int] = `1+2+3`\n  = <`x*2`>\n\n= <p()>\n",
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
            "@p():\n  items [xs: list[str]] from <id(seed)>\n  [ys: list[str]] = `[v+'!' for v in xs]`\n  = <`'-'.join(ys)`>\n\n"
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
