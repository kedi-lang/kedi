"""Prelude and fragment Python sees Kedi variables live and writes them back under contract."""

from __future__ import annotations

from typing import Any

import pytest

import kedi
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.interactive_helpers import inspect_fragment


def _value(session: Any, expression: str) -> Any:
    return inspect_fragment(session, f"[inspection_value: Any] = `{expression}`")


def test_a_python_fragment_writes_back_and_keeps_its_new_names() -> None:
    with kedi.interactive() as session:
        session.execute("[x] = a\n[z: int] = `1`")
        session.execute("```\nx = 'b'\nimport math\n```")

        assert _value(session, "(x, math.floor(math.pi))") == ("b", 3)
        assert inspect_fragment(session, "[inspection_value] = <x>") == "b"

        with pytest.raises(KediExecutionError, match="write-back for 'z' does not match"):
            session.execute("```\nz = 'not an int'\n```")
        assert _value(session, "z") == 1


def test_a_rejected_write_back_changes_no_kedi_variable() -> None:
    with kedi.interactive() as session:
        session.execute("[x] = a\n[z: int] = `1`")

        with pytest.raises(KediExecutionError, match="annotated type"):
            session.execute("```\nimport json\nx = 'b'\nz = 'bad'\n```")
        with pytest.raises(KediExecutionError, match="boom"):
            session.execute("```\nx = 'c'\nraise ValueError('boom')\n```")

        # Kedi variables are written all or nothing; like the rest of a session, Python names
        # completed before an error stay visible.
        assert _value(session, "(x, z, 'json' in dir())") == ("a", 1, True)


def test_fragment_functions_read_and_write_kedi_variables_live() -> None:
    with kedi.interactive() as session:
        session.execute("[scale: int] = `2`")
        session.execute(
            "```\n"
            "def doubled():\n    return scale * 2\n\n"
            "def bump():\n    global scale\n    scale = scale + 1\n\n"
            "def spoil():\n    global scale\n    scale = 'oops'\n\n"
            "def fail():\n    global scale\n    scale = 99\n    raise ValueError('no')\n"
            "```"
        )
        session.execute("[scale] := `10`")
        assert _value(session, "doubled()") == 20

        session.execute("`bump()`")
        assert _value(session, "scale") == 11
        with pytest.raises(KediExecutionError, match="write-back for 'scale' does not match"):
            session.execute("`spoil()`")
        with pytest.raises(KediExecutionError, match="no"):
            session.execute("`fail()`")
        assert _value(session, "scale") == 11


def test_a_prelude_name_is_not_replaced_by_a_kedi_variable_of_the_same_name() -> None:
    with kedi.interactive() as session:
        session.execute("```\nlimit = 5\n\ndef own_limit():\n    return limit\n```")
        session.execute("[limit] = seven")
        session.execute("```\nlimit = 'nine'\n```")

        assert inspect_fragment(session, "[inspection_value] = <limit>") == "nine"
        assert _value(session, "own_limit()") == 5


def test_file_prelude_functions_see_kedi_variables_live() -> None:
    source = (
        "```\n"
        "def doubled():\n    return scale * 2\n\n"
        "def bump():\n    global scale\n    scale = scale + 1\n"
        "```\n"
        "[scale: int] = `2`\n"
        "[scale] := `10`\n"
        "`bump()`\n"
        "= `(scale, doubled())`\n"
    )

    assert compile_program(parse_program(source)).run_main() == (11, 22)


def test_prelude_functions_see_the_calling_procedures_variables() -> None:
    source = (
        "```\n"
        "def doubled():\n    return n * 2\n\n"
        "def bump():\n    global n\n    n = n + 1\n"
        "```\n"
        "@twice_bumped() -> int:\n"
        "  [n: int] = `4`\n"
        "  `bump()`\n"
        "  = `doubled()`\n"
        "= `twice_bumped()`\n"
    )

    assert compile_program(parse_program(source)).run_main() == 10


def test_prelude_namespace_falls_back_to_the_running_environment_only() -> None:
    from kedi.core.runtime import KediRuntime
    from kedi.lang.compiler.environment import KediEnv, PreludeNamespace, running_python

    namespace = PreludeNamespace(own=1)
    running = KediEnv()
    running["visible"] = 2

    assert namespace["own"] == 1
    assert "__builtins__" in namespace
    with pytest.raises(KeyError):
        namespace["visible"]
    with running_python(KediRuntime(), running, namespace):
        assert namespace["visible"] == 2
        with pytest.raises(KeyError):
            namespace["missing"]
