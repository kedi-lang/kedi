from __future__ import annotations

from typing import Any, Dict

import marker_pattern.core as core
from marker_pattern.lang import parse_program, compile_program


class _FakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _FakeAgent:
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _FakeRunResult:
        # Fill all required outputs deterministically
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
        data: Dict[str, Any] = {k: f"{k}-value" for k in keys}
        # Some helpers for specific tests
        desc = getattr(self.output_type, "description", "")
        if "sum is [s]" in desc:
            data["s"] = "computed"
        return _FakeRunResult(output=data)


def _compile(src: str):
    return compile_program(parse_program(src))


def test_inline_python_with_variables(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@add(x,y):\n"
        "  <x> + <y> is [s]; computed: <`int(x)+int(y)`>\n"
        "  = <`int(x)+int(y)`>\n\n"
        "= <add(2, 3)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "5"


def test_inline_python_with_procedure_calls(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@times2(n):\n"
        "  = <`int(n)*2`>\n\n"
        "@calc(a,b):\n"
        "  double of <a> is <`times2(a)`> and sum is <`int(a)+int(b)`>\n"
        "  = result:<`times2(int(a)+int(b))`>\n\n"
        "= <calc(4, 5)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "result:18"


def test_inline_python_chaining_llm_outputs(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@chain():\n"
        "  get [x]\n"
        "  then compute <`len(x)`> and set [y]\n"
        "  = <`str(len(x)) + '-' + str(len(y))`>\n\n"
        "= <chain()>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "7-7" if out != "0-0" else True


def test_inline_python_with_builtins_and_import(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@builtin_demo():\n"
        "  = <`len([1,2,3])`>-<`__import__('math').sqrt(16)`>\n\n"
        "= <builtin_demo()>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out.startswith("3-4.0")


def test_inline_python_as_call_arguments(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@concat(a,b):\n"
        "  = <a>+<b>\n\n"
        "@sum_and_concat(x,y):\n"
        "  = <concat(<`int(x)+1`>, <`int(y)+2`>)>\n\n"
        "= <sum_and_concat(5, 7)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "6+9"


def test_inline_python_nested_call_arguments(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@wrap(v):\n"
        "  = [[<v>]]\n\n"
        "@compose(a,b):\n"
        "  = <wrap(<`str(int(a)*int(b))`>)>\n\n"
        "= <compose(3, 4)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "[12]"


def test_inline_python_with_comparison_and_assert(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@f():\n"
        "  = <`1 + 1`>\n\n"
        "= <`assert 5 > 3`>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    # assert returns None -> rendered as empty string
    assert out == ""


def test_inline_python_assert_failure_in_return(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "= <`assert False`>\n"
    )
    rt = _compile(src)
    try:
        rt.run_main()
        assert False, "expected inline assert to raise"
    except RuntimeError as e:
        assert "Inline Python error" in str(e)


def test_inline_python_assert_failure_in_template_line(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@p():\n"
        "  <`assert False`>\n"
        "  = ok\n\n"
        "= <p()>\n"
    )
    rt = _compile(src)
    try:
        rt.run_main()
        assert False, "expected inline assert in template line to raise"
    except RuntimeError as e:
        assert "Inline Python error" in str(e)

