from __future__ import annotations

import asyncio
from threading import Event, Lock
from typing import Any

import pytest

from kedi.engine import ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import (
    ActiveForm,
    ConditionalLoopStmt,
    GroupStmt,
    IfStmt,
    LoopStmt,
    ProcDefStmt,
    TypeDefStmt,
)
from kedi.lang.compiler.control_flow import defines_scoped_names


@pytest.mark.parametrize("procedure_scope", [False, True])
def test_map_definitions_keep_source_order_and_restore_the_enclosing_scope(
    procedure_scope: bool,
) -> None:
    arrived, lock = Event(), Lock()
    entered = 0
    records = []

    def checkpoint() -> None:
        nonlocal entered
        with lock:
            entered += 1
            if entered > 1:
                arrived.set()
        arrived.wait(timeout=0.05)

    def record(index: int, value: int, model: Any) -> None:
        with lock:
            records.append((index, value, model.model_fields["value"].description))

    body = """
> loop [index]: `[1, 2, 3]`:
  [item: int] = `index`
> map:
  > if: `True`:
    ~Box(value: Annotated[int, `str(item)`])
    @read() -> int:
      = `item`
    `checkpoint()`
    `record(item, read(), Box)`
""".lstrip()
    source = "~Box(value: str)\n@read() -> int:\n  = `99`\n"
    if procedure_scope:
        source += "@run() -> int:\n" + "".join("  " + line + "\n" for line in body.splitlines())
        source += "  = `read()`\n= `(run(), read())`\n"
    else:
        source += body + "= `read()`\n"
    engine = ThreadEngine(4)
    runtime = compile_program(
        parse_program(source),
        engine=engine,
        runtime_globals={"checkpoint": checkpoint, "record": record},
    )
    try:
        assert runtime.run_main() == ((3, 99) if procedure_scope else 3)
        assert records == [(1, 1, "1"), (2, 2, "2"), (3, 3, "3")]
    finally:
        asyncio.run(runtime.aclose())
        engine.shutdown()


@pytest.mark.parametrize(
    "statements, expected",
    [
        ([], False),
        ([ProcDefStmt()], True),
        ([IfStmt(body=[], else_body=[TypeDefStmt()])], True),
        ([IfStmt(body=[], else_body=[])], False),
        ([LoopStmt(body=[], map_body=[TypeDefStmt()])], True),
        ([LoopStmt(body=[], map_body=[])], False),
        ([ConditionalLoopStmt(body=[TypeDefStmt()])], True),
        ([ConditionalLoopStmt(body=[])], False),
        ([GroupStmt(body=[TypeDefStmt()])], True),
        ([GroupStmt(body=[])], False),
        (
            [GroupStmt(body=[], form=ActiveForm([TypeDefStmt()], "", "supplied"))],
            True,
        ),
    ],
)
def test_map_definition_detection_uses_active_nested_bodies(statements, expected) -> None:
    assert defines_scoped_names(statements) is expected


def test_definition_bearing_map_still_drains_later_iterations_after_failure() -> None:
    records = []

    def record(value: int) -> None:
        records.append(value)
        if value == 1:
            raise ValueError("first iteration failed")

    engine = ThreadEngine(2)
    runtime = compile_program(
        parse_program(
            """
@run():
  > loop [item]: `[1, 2, 3]`:
    [value: int] = `item`
  > map:
    ~Box(value: int)
    `record(value)`
= <run()>
""".lstrip()
        ),
        engine=engine,
        runtime_globals={"record": record},
    )
    try:
        with pytest.raises(KediExecutionError, match="first iteration failed"):
            runtime.run_main()
        assert records == [1, 2, 3]
    finally:
        asyncio.run(runtime.aclose())
        engine.shutdown()
