from __future__ import annotations

import pytest

from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


@pytest.mark.parametrize(
    "expression, expected",
    [
        ('"""roses are red\nviolets are blue"""', "roses are red\nviolets are blue"),
        ('f"""{1 + 1}\n  second line"""', "2\n  second line"),
        ('r"""first\\n\nsecond"""', "first\\n\nsecond"),
        ('b"""first\nsecond"""', b"first\nsecond"),
        ('"""first\r\nsecond"""', "first\nsecond"),
    ],
)
@pytest.mark.parametrize("context", ["initialize", "assign", "return"])
def test_value_blocks_preserve_python_string_contents(expression, expected, context) -> None:
    block = f"```\nreturn {expression}\n```"
    if context == "initialize":
        source = f"[value: Any] = {block}\n= `value`\n"
    elif context == "assign":
        source = f"[value: Any] = `None`\n[value] := {block}\n= `value`\n"
    else:
        source = f"= {block}\n"
    runtime = compile_program(parse_program(source))
    assert runtime.run_main() == expected


def test_value_block_preserves_nested_function_literals_and_writeback() -> None:
    source = '''[value: str] = initial
[result: str] = ```
def nested():
    return """first
second"""
value = nested()
return value
```
= `result == value and value == "first\\nsecond"`
'''
    runtime = compile_program(parse_program(source))
    assert runtime.run_main() is True


def test_value_block_error_keeps_original_python_line() -> None:
    source = '[value: str] = ```\ntext = """first\nsecond"""\nreturn 1 / 0\n```\n'
    runtime = compile_program(parse_program(source, source_path="literal.kedi"))
    with pytest.raises(KediExecutionError) as caught:
        runtime.run_main()
    trace = caught.value.python_traces[-1]
    assert trace.python_lineno == 3
    assert trace.kedi_lineno == 4
    assert trace.code.splitlines()[2] == "return 1 / 0"
