from __future__ import annotations

from typing import Any, Dict, List

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
        # Introspect output_type to emit sane typed values
        data: Dict[str, Any] = {}
        ty = self.output_type
        # Handle TypedDict-annotated classes
        annotations: Dict[str, Any] = {}
        if hasattr(ty, "__annotations__") and isinstance(ty.__annotations__, dict):  # type: ignore[attr-defined]
            annotations = ty.__annotations__  # type: ignore[attr-defined]

        # Parse required keys from question prompt as fallback
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]

        for k in keys:
            ann = annotations.get(k, str)
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

        return _FakeRunResult(output=data)


def _run(monkeypatch: Any, src: str) -> Any:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    rt = compile_program(parse_program(src))
    return rt.run_main()


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
        "@p():\n  [nums: list[int]] = <`[1,2,3]`>\n  = <`sum(nums)`>\n\n= <p()>\n",
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
            "  ```\n"
            "  s = sum(arr)\n"
            "  ```\n"
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
    # Parser accepts the annotation; TypeValidator fails -> treated as str
    out = _run(
        monkeypatch,
        (
            "@p():\n  value [x: NotARealType]\n  = <x>\n\n= <p()>\n"
        ),
    )
    assert isinstance(out, str)


# Additional sampling across features (keep count high for coverage)
@pytest.mark.parametrize("src", [
    "@p():\n [x: int] = <`2**5`>\n = <`x+1`>\n\n= <p()>\n",
    "@p():\n [s: str] = hello\n = <s>!\n\n= <p()>\n",
    "@p():\n arr [a: list[int]]\n ```\n first = a[0] if a else 0\n ```\n = <`str(first)`>\n\n= <p()>\n",
    "@times2(n):\n = <`int(n)*2`>\n\n@use():\n list [xs: list[int]]\n [ys: list[int]] = <`[times2(v) for v in xs]`>\n = <`str(sum(int(v) for v in ys))`>\n\n= <use()>\n",
    "@p():\n alpha [a] and ints [b: list[int]]\n = <a>-<`sum(b)`>\n\n= <p()>\n",
])
def test_parametrized_typed_snippets(monkeypatch: Any, src: str) -> None:
    out = _run(monkeypatch, src)
    assert isinstance(out, str)


