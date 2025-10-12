from __future__ import annotations

import textwrap
from typing import Any

import pytest

import marker.core as core
from marker.lang import compile_program, parse_program


class _SingleFakeAgent:  # type: ignore[misc]
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:  # type: ignore[no-untyped-def]
        self.output_type = output_type

    def run_sync(self, prompt: str):  # type: ignore[override]
        # Very simple agent: return all requested fields as deterministic placeholders
        from typing import Dict, List

        data: Dict[str, Any] = {}
        # discover requested keys like [x, y]
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]

        class R:
            def __init__(self, d: Dict[str, Any]):
                self.output = d

        for k in keys:
            data[k] = f"{k}-value"
        return R(data)


def _run(src: str) -> str:
    core.Agent = _SingleFakeAgent  # type: ignore[assignment]
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    return out


def test_inline_comments_basic(monkeypatch: Any) -> None:
    src = (
        "# comment only\n"
        "@p(x):  # inline comment\n"
        "  = <x>\n\n"
        "= <p(value ## still data,, has hash)> # trailing comment\n"
    )
    out = _run(src)
    assert out == "value # still data, has hash"


def test_inline_comments_escape(monkeypatch: Any) -> None:
    src = "text ## not comment\n" "= done\n"
    out = _run(src)
    assert out == "done"


def test_block_comment_basic(monkeypatch: Any) -> None:
    src = "###\n" "ignored\n" "###\n" "= ok\n"
    out = _run(src)
    assert out == "ok"


def test_block_comment_between_statements(monkeypatch: Any) -> None:
    src = (
        "@p():\n" "  ###\n" "  ignored lines\n" "  ###\n" "  = in body\n\n" "= <p()>\n"
    )
    out = _run(src)
    assert out == "in body"


def test_block_comment_unterminated(monkeypatch: Any) -> None:
    src = "###\n" "still comment\n"
    with pytest.raises(Exception):
        parse_program(src)


def test_block_comment_invalid_usage(monkeypatch: Any) -> None:
    for bad in (
        "### trailing text\n= fail\n",
        "= something ###\n",
        "text ### text\n",
    ):
        with pytest.raises(Exception):
            parse_program(bad)


def test_inline_comment_without_escape_truncates(monkeypatch: Any) -> None:
    src = "@p(x):\n" "  = <x>\n\n" "= <p(value # comment)>\n"
    with pytest.raises(Exception):
        parse_program(src)


def test_inline_comment_in_multiline_continue(monkeypatch: Any) -> None:
    src = "= first part \\\n" "  second # comment\n"
    out = _run(src)
    assert out == "first part second"


def test_multiline_and_block_comment_crosstalk(monkeypatch: Any) -> None:
    src = (
        "@p():\n"
        "  ###\n"
        '  Ignored "= 1" line\n'
        "  [x] = 2\n"
        "  ###\n"
        "  prefix \\\n"
        "    value\n"
        "  = <x> # inline comment after assignment\n\n"
        "[x] = value # inline comment after assignment\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out.endswith("value")


def test_inline_comment_in_python_block(monkeypatch: Any) -> None:
    src = (
        "@p() -> int:\n"
        "  = ```\n"
        "  return 1  # python comment that stays\n"
        "  ```\n\n"
        "= <`str(p())`>\n"
    )
    out = _run(src)
    assert out == "1"


def test_hash_inside_block_comment_raises(monkeypatch: Any) -> None:
    src = "###\n" "### inside\n" "###\n" "= ok\n"
    with pytest.raises(Exception):
        parse_program(src)


def test_triple_hash_inside_python_block_allowed(monkeypatch: Any) -> None:
    src = "@p():\n" "  = ```\n" "  print('hi')\n" "  ###\n" "  ```\n\n" "= <p()>\n"
    out = _run(src)
    assert out == ""


def test_comment_with_whitespace_prefix(monkeypatch: Any) -> None:
    src = "   # comment after spaces\n" "= ok\n"
    out = _run(src)
    assert out == "ok"


def test_basic_multiline_return(monkeypatch: Any) -> None:
    src = "= Start \\\n" "  middle \\\n" "  end\n"
    out = _run(src)
    assert out == "Start middle end"


def test_multiline_with_substitutions(monkeypatch: Any) -> None:
    src = (
        "@p(x,y):\n"
        "  = Left <x> \\     \n"
        "    mid <`y.upper()`> \\\n"
        "    right\n\n"
        "= <p(alpha, beta)>\n"
    )
    out = _run(src)
    assert out == "Left alpha mid BETA right"


def test_multiline_escaped_backslash(monkeypatch: Any) -> None:
    src = "= path: C:\\\\Users\\\\name \\\n" "  tail\n"
    out = _run(src)
    assert "C:\\Users\\name" in out and out.endswith(" tail")


def test_multiline_inside_template_line(monkeypatch: Any) -> None:
    src = (
        "@p():\n"
        "  prefix \\\n"
        "    [x] \\\n"
        "    suffix\n"
        "  = <x>\n\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out.endswith("-value")


def test_multiline_with_nested_calls(monkeypatch: Any) -> None:
    src = (
        "@id(x):\n  = <x>\n\n"
        "= [[ [[ <id(A)>    \\   \n"  # ignore whitespaces after \, but not before
        "and <id(B)>    ]] ]]\n"
    )
    out = _run(src)
    assert out == "[ [ A and B    ] ]"


def test_misplaced_backslash_in_return_raises(monkeypatch: Any) -> None:
    # Single backslash in the middle of a line must be escaped as \\
    with pytest.raises(Exception):
        parse_program("= hello \\ world\n")


def test_misplaced_backslash_in_template_raises(monkeypatch: Any) -> None:
    src = "@p():\n" "  text \\ here\n" "  = ok\n\n" "= <p()>\n"
    with pytest.raises(Exception):
        parse_program(src)
