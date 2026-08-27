"""Inline-Python scoping and the Python origin map.

Existing Kedi vars mutate back, new Python names don't leak, empty blocks are
no-ops, and each embedded snippet maps to its own line in tracebacks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.executors.base import DefaultExecutor
from kedi.lang import compile_program, parse_program


def test_inline_statement_mutates_existing_kedi_var() -> None:
    # an inline statement that mutates an existing Kedi var must sync back.
    executor = DefaultExecutor()
    rt = KediRuntime()

    local_env = {"x": 1}
    executor.evaluate_inline(rt, "x = x + 1", local_env)

    assert local_env == {"x": 2}


def test_inline_statement_does_not_leak_new_python_names() -> None:
    # New names created in the statement fallback stay local to the block.
    executor = DefaultExecutor()
    rt = KediRuntime()

    local_env = {"b": 10}
    executor.evaluate_inline(rt, "b = b - 4; temp = 999", local_env)

    assert local_env == {"b": 6}


def test_empty_python_block_runs_as_noop() -> None:
    # an empty block must compile to a no-op rather than raising.
    executor = DefaultExecutor()
    rt = KediRuntime()

    assert executor.execute_block(rt, "", {}) is None


def test_block_mutation_propagates_existing_var_only() -> None:
    executor = DefaultExecutor()
    rt = KediRuntime()

    local_env = {"y": 5}
    result = executor.execute_block(rt, "y = y + 10\nnewvar = 99", local_env)

    assert result is None
    assert local_env == {"y": 15}


def test_block_return_value_with_mutation() -> None:
    executor = DefaultExecutor()
    rt = KediRuntime()

    local_env = {"a": 2}
    result = executor.execute_block(rt, "a = a * 3\nreturn a + 1", local_env)

    assert result == 7
    assert local_env == {"a": 6}


def test_block_mutation_syncs_runtime_env() -> None:
    # Mutations to existing vars must also reach rt._current_env.
    executor = DefaultExecutor()
    rt = KediRuntime()

    runtime_env = {"count": 1}
    token = rt._current_env.set(runtime_env)
    try:
        local_env = {"count": 1}
        executor.execute_block(rt, "count = count + 41", local_env)
    finally:
        rt._current_env.reset(token)

    assert local_env == {"count": 42}
    assert runtime_env == {"count": 42}


def test_block_cannot_delete_an_existing_kedi_binding() -> None:
    executor = DefaultExecutor()
    runtime = KediRuntime()
    local_env = {"value": 1}

    executor.execute_block(runtime, "del value", local_env)

    assert local_env == {"value": 1}


def test_python_trace_maps_to_first_snippet_in_same_file(tmp_path: Path) -> None:
    # the first embedded snippet raising must map to its own line, not a later snippet's.
    srcp = tmp_path / "two_snippets.kedi"
    srcp.write_text(
        "@boom() -> str:\n  = <`1 / 0`>\n  = <`2 + 2`>\n\n= <boom()>\n",
        encoding="utf-8",
    )
    program = parse_program(srcp.read_text(encoding="utf-8"), source_path=str(srcp))
    rt = compile_program(program)

    with pytest.raises(KediExecutionError) as info:
        rt.run_main()

    trace = info.value.python_traces[-1]
    assert trace.code == "1 / 0"
    assert trace.python_lineno == 1
    assert trace.kedi_lineno == 2
    assert trace.span is not None
    assert trace.span.source_path == str(srcp)


def test_python_origins_use_unique_keys_per_snippet() -> None:
    # Each embedded snippet registers under a unique virtual filename so origins
    # in the same .kedi file never overwrite one another.
    rt = KediRuntime()

    first = rt.register_python_origin(kind="inline", code="1 / 0", span=None)
    second = rt.register_python_origin(kind="inline", code="2 + 2", span=None)

    assert first != second
    assert set(rt._python_origins) == {first, second}
    assert rt._python_origins[first].code == "1 / 0"
    assert rt._python_origins[second].code == "2 + 2"


def test_python_origins_reuse_an_identical_snippet() -> None:
    rt = KediRuntime()

    first = rt.register_python_origin(kind="type_expression", code="str", span=None)
    second = rt.register_python_origin(kind="type_expression", code="str", span=None)

    assert second == first
    assert tuple(rt._python_origins) == (first,)
