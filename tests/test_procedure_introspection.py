from __future__ import annotations

import inspect

from kedi.lang import compile_program, parse_program


def _runtime(source: str):
    return compile_program(parse_program(source))


def test_python_interop_exposes_stub_signature_name_and_source() -> None:
    rt = _runtime("@kedi_proc(x: int, y: int) -> int:\n  = `x + y`\n\n= ok\n")

    name = rt.executor.evaluate_inline(rt, "kedi_proc.__name__", {})
    signature = rt.executor.evaluate_inline(
        rt, "str(inspect.signature(kedi_proc))", {"inspect": inspect}
    )
    source = rt.executor.evaluate_inline(
        rt, "inspect.getsource(kedi_proc)", {"inspect": inspect}
    )

    assert name == "kedi_proc"
    assert signature == "(x: int, y: int) -> int"
    assert source == (
        "def kedi_proc(x: int, y: int) -> int:\n"
        '    """This stub is generated for Python introspection."""\n'
        "    return __kedi_call__(x, y)\n"
    )
    assert "wrapper(*args: Any)" not in source


def test_python_interop_stub_keeps_typed_annotations() -> None:
    src = (
        "~Person(name: str)\n"
        "@format_person(person: Person, note: Optional[str]) -> list[str]:\n"
        "  = `[]`\n\n"
        "= ok\n"
    )
    rt = _runtime(src)

    signature = rt.executor.evaluate_inline(
        rt, "str(inspect.signature(format_person))", {"inspect": inspect}
    )

    assert "person:" in signature
    assert "Person" in signature
    assert "note: Optional[str]" in signature
    assert "-> list[str]" in signature


def test_nested_procedure_python_interop_uses_nested_stub() -> None:
    src = (
        "@outer() -> str:\n"
        "  @inner(value: int) -> int:\n"
        "    = `value + 1`\n"
        "  [info: str] = ```\n"
        "  import inspect\n"
        '  return f"{inner.__name__}|{inspect.signature(inner)}|{inspect.getsource(inner)}"\n'
        "  ```\n"
        "  = <info>\n\n"
        "= <outer()>\n"
    )
    out = _runtime(src).run_main()

    assert out.startswith("inner|(value: int) -> int|def inner(value: int) -> int:\n")
    assert '"""This stub is generated for Python introspection."""' in out
    assert "return __kedi_call__(value)" in out
    assert "wrapper(*args: Any)" not in out
