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
    source = rt.executor.evaluate_inline(rt, "inspect.getsource(kedi_proc)", {"inspect": inspect})

    assert name == "kedi_proc"
    assert signature == "(x: int, y: int) -> int"
    assert source == ("def kedi_proc(x: int, y: int) -> int:\n    return __kedi_call__(x, y)\n")
    assert "wrapper(*args: Any)" not in source
    assert rt.executor.evaluate_inline(rt, "kedi_proc.__doc__", {}) is None


def test_python_interop_docstring_uses_first_multiline_comment() -> None:
    rt = _runtime(
        "@kedi_proc(x: int) -> int:\n"
        "  ###\n"
        "  Add one to the input.\n"
        "  Keeps the prose as the Python docstring.\n"
        "  ###\n"
        "  = `x + 1`\n\n"
        "= ok\n"
    )

    source = rt.executor.evaluate_inline(rt, "inspect.getsource(kedi_proc)", {"inspect": inspect})
    doc = rt.executor.evaluate_inline(rt, "kedi_proc.__doc__", {})

    assert source == (
        "def kedi_proc(x: int) -> int:\n"
        '    """Add one to the input.\n'
        '    Keeps the prose as the Python docstring."""\n'
        "    return __kedi_call__(x)\n"
    )
    assert doc == "Add one to the input.\nKeeps the prose as the Python docstring."


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
    assert "return __kedi_call__(value)" in out
    assert "wrapper(*args: Any)" not in out


def test_type_init_exposes_stub_signature_and_source() -> None:
    src = "~Person(name: str, salary: int = `0`)\n" "[person: Person] = `Person('Ada')`\n" "= ok\n"
    rt = _runtime(src)

    signature = rt.executor.evaluate_inline(
        rt, "str(inspect.signature(Person.__init__))", {"inspect": inspect}
    )
    source = rt.executor.evaluate_inline(
        rt, "inspect.getsource(Person.__init__)", {"inspect": inspect}
    )
    salary = rt.executor.evaluate_inline(rt, "Person('Ada').salary", {})

    assert signature == "(self, name: str, salary: int = 0) -> None"
    assert source == (
        "def __init__(self, name: str, salary: int = 0) -> None:\n"
        '    """This stub is generated for Python introspection."""\n'
        "    __kedi_model_init__(self, name=name, salary=salary)\n"
    )
    assert salary == 0
    assert "_init(self, *args" not in source


def test_type_class_source_omits_new_stub() -> None:
    src = "~Person(user: str, salary: float = `100000`)\n= ok\n"
    rt = _runtime(src)

    class_source = rt.executor.evaluate_inline(
        rt, "inspect.getsource(Person)", {"inspect": inspect}
    )
    salary = rt.executor.evaluate_inline(rt, "Person('Ada').salary", {})

    assert class_source == (
        "class Person:\n"
        '    """This stub is generated for Python introspection."""\n'
        "    user: str\n"
        "    salary: float = 100000\n"
        "\n"
        "    def __init__(self, user: str, salary: float = 100000) -> None:\n"
        '        """This stub is generated for Python introspection."""\n'
        "        __kedi_model_init__(self, user=user, salary=salary)\n"
    )
    assert salary == 100000
