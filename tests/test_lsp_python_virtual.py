"""LSP feature tests."""

from __future__ import annotations

import ast
import io
import json
from types import SimpleNamespace
from typing import Any, cast

from kedi.lsp.python_virtual import compute_python_virtual_document


def test_python_virtual_document_models_procedure_scope():
    src = (
        "@calculate_stats(numbers: list[int]) -> int:\n"
        "  [total: int] = ```\n"
        "  return sum(numbers)\n"
        "  ```\n"
        "  [count: int] = ```\n"
        "  return len(numbers)\n"
        "  ```\n"
        "  [average: float] = ```\n"
        "  return total / count if count > 0 else 0\n"
        "  ```\n"
        "  = Average of <`count`> numbers is <`average`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    text = virtual["text"]
    assert "def calculate_stats(numbers: list[int]) -> int: ..." in text
    assert "def __kedi_scope_calculate_stats(numbers: list[int]) -> int:" in text
    assert "total: int" in text
    assert "count: int" in text
    assert "average: float" in text
    assert "return total / count if count > 0 else 0" in text
    assert any(
        s["kind"] == "parameter"
        and s["name"] == "numbers"
        and s["sourceRange"]["start"] == {"line": 0, "character": 17}
        for s in virtual["symbols"]
    )
    assert any(
        s["kind"] == "local"
        and s["name"] == "total"
        and s["sourceRange"]["start"] == {"line": 1, "character": 3}
        for s in virtual["symbols"]
    )
    assert any(
        s["kind"] == "local"
        and s["name"] == "average"
        and s["sourceRange"]["start"] == {"line": 7, "character": 3}
        for s in virtual["symbols"]
    )
    assert any(
        m["sourceRange"]["start"]["line"] == 8
        and m["virtualRange"]["start"]["line"] > m["sourceRange"]["start"]["line"]
        for m in virtual["mappings"]
    )


def test_python_virtual_document_includes_procedure_docstring():
    src = (
        "@calculate_stats(numbers: list[int]) -> int:\n"
        "  ###\n"
        "  Compute aggregate metrics.\n"
        "  Reuses Python locals below.\n"
        "  ###\n"
        "  [total: int] = ```\n"
        "  return sum(numbers)\n"
        "  ```\n"
        "  = <`str(total)`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    text = virtual["text"]
    assert '"""Compute aggregate metrics.' in text
    assert "Reuses Python locals below." in text


def test_python_virtual_document_can_focus_single_procedure_body():
    src = (
        "@alpha(items: list[int]) -> int:\n"
        "  [total: int] = ```\n"
        "  return sum(items)\n"
        "  ```\n"
        "  = <`total`>\n"
        "\n"
        "@beta(name: str) -> str:\n"
        "  [upper: str] = ```\n"
        "  return name.upper()\n"
        "  ```\n"
        "  = <`upper`>\n"
        "\n"
        "= done\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=2)
    ast.parse(virtual["text"])
    assert virtual["focusKey"] == "proc:alpha:1"
    assert "def alpha(items: list[int]) -> int: ..." in virtual["text"]
    assert "def beta(name: str) -> str: ..." in virtual["text"]
    assert "def __kedi_scope_alpha(items: list[int]) -> int:" in virtual["text"]
    assert "return sum(items)" in virtual["text"]
    assert "def __kedi_scope_beta(name: str) -> str:" not in virtual["text"]
    assert "return name.upper()" not in virtual["text"]
    assert "def __kedi_toplevel__() -> Any:" not in virtual["text"]


def test_python_virtual_document_includes_nested_procedure_body():
    src = (
        "@foo() -> str:\n"
        "  @bar(name: str) -> str:\n"
        "    [upper: str] = ```\n"
        "    return name.upper()\n"
        "    ```\n"
        "    = <`upper`>\n"
        "  = <bar(ok)>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def bar(name: str) -> str:" in virtual["text"]
    assert "return name.upper()" in virtual["text"]
    assert any(
        s["kind"] == "procedure"
        and s["name"] == "bar"
        and s["sourceRange"]["start"] == {"line": 1, "character": 3}
        for s in virtual["symbols"]
    )


def test_python_virtual_document_can_focus_nested_procedure_body():
    src = (
        "@foo() -> str:\n"
        "  @bar(name: str) -> str:\n"
        "    [upper: str] = ```\n"
        "    return name.upper()\n"
        "    ```\n"
        "    = <`upper`>\n"
        "  = <bar(ok)>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=3)
    ast.parse(virtual["text"])
    assert virtual["focusKey"] == "proc:bar:2"
    assert "def __kedi_scope_bar(name: str) -> str:" in virtual["text"]
    assert "return name.upper()" in virtual["text"]
    assert "def __kedi_scope_foo() -> str:" not in virtual["text"]


def test_python_virtual_document_emits_pass_for_empty_toplevel_scope():
    src = "= done\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_toplevel__() -> Any:" in virtual["text"]
    assert "    pass" in virtual["text"]


def test_python_virtual_document_includes_prelude_symbols_for_toplevel_scope():
    src = (
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "class Person(BaseModel):" in virtual["text"]
    assert "person: Person = ..." in virtual["text"]
    assert "    person" in virtual["text"]


def test_focused_virtual_document_stubs_unfocused_module_python():
    src = (
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=9)
    ast.parse(virtual["text"])
    assert "from pydantic import BaseModel, Field" not in virtual["text"]
    assert "class BaseModel: ..." in virtual["text"]
    assert "def Field(*args: Any, **kwargs: Any) -> Any: ..." in virtual["text"]
    assert "class Person:" in virtual["text"]
    assert "def __init__(self, name: str, age: int) -> None: ..." in virtual["text"]
    assert "person: Person = ..." in virtual["text"]


def test_focused_virtual_document_keeps_focused_module_python_mapped():
    src = (
        "```\n"
        "from pydantic import BaseModel\n"
        "\n"
        "class Person(BaseModel):\n"
        "    name: str\n"
        "```\n\n"
        "= <`Person(name='Ada')`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=3)
    ast.parse(virtual["text"])
    assert "from pydantic import BaseModel" in virtual["text"]
    assert "class Person(BaseModel):" in virtual["text"]
    assert any(region["range"]["start"]["line"] == 1 for region in virtual["ranges"])


def test_virtual_document_exposes_runtime_args_as_attrdict():
    src = "[city: str] = `args.city`\n= <`args.debug`>\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")

    ast.parse(virtual["text"])
    assert "ArgType: TypeAlias = Union[str, bool, None]" in virtual["text"]
    assert "class AttrDict(dict[_K, _V], Generic[_K, _V]):" in virtual["text"]
    assert "args: AttrDict[str, ArgType] = ..." in virtual["text"]
    assert "args.city" in virtual["text"]
    assert "args.debug" in virtual["text"]


def test_python_virtual_document_hoists_leading_toplevel_python_blocks():
    src = (
        "```\n"
        "import this\n"
        "```\n\n"
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        "Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "import this" in virtual["text"]
    assert "class Person(BaseModel):" in virtual["text"]
    assert "def __kedi_toplevel__() -> Any:" in virtual["text"]
    assert "    person: Person = ..." in virtual["text"]


def test_python_virtual_document_emits_ellipsis_for_empty_generated_python_blocks():
    src = (
        "@p() -> str:\n"
        "  [x: int] = ```\n"
        "  # comment only\n"
        "  ```\n"
        "  ```\n"
        "  # comment only\n"
        "  ```\n"
        "  = ok\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_block_x_2() -> int:" in virtual["text"]
    assert "        ..." in virtual["text"]
    assert "def __kedi_scope_p() -> str:" in virtual["text"]
    assert "    ..." in virtual["text"]


def test_python_virtual_document_emits_ellipsis_for_comment_only_validation_blocks():
    src = (
        "@score() -> int:\n"
        "  = `1`\n"
        "\n"
        "@eval: score:\n"
        "  > metric: check:\n"
        "    = ```\n"
        "    # comment only\n"
        "    ```\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_metric_score_check() -> Any:" in virtual["text"]
    assert "    expected: Any = ..." in virtual["text"]
    assert "    ..." in virtual["text"]


def test_python_virtual_document_models_type_stub_init_signature():
    src = "~Person(name, age: int, tags: list[str])\n= ok\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    assert "class Person:" in virtual["text"]
    assert "    name: str" in virtual["text"]
    assert "    age: int" in virtual["text"]
    assert "    tags: list[str]" in virtual["text"]
    assert (
        "    def __init__(self, name: str, age: int, tags: list[str]) -> None: ..."
        in virtual["text"]
    )


def test_python_virtual_document_models_defaulted_signatures():
    src = (
        "~Person(name: str, salary: int = `0`)\n"
        "@make(name: str, salary: int = `0`) -> Person:\n"
        "  = `Person(name, salary)`\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "    salary: int = 0" in virtual["text"]
    assert "    def __init__(self, name: str, salary: int = 0) -> None: ..." in virtual["text"]
    assert "def make(name: str, salary: int = 0) -> Person: ..." in virtual["text"]
    assert "def __kedi_scope_make(name: str, salary: int = 0) -> Person:" in virtual["text"]
    assert any(region["text"] == "0" for region in virtual["ranges"])


def test_python_virtual_document_defaults_untyped_proc_params_to_str():
    src = "@greet(name):\n  = ```\n  return name.upper()\n  ```\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def greet(name: str) -> str: ..." in virtual["text"]
    assert "def __kedi_scope_greet(name: str) -> str:" in virtual["text"]


def test_python_virtual_document_maps_header_default_expression():
    src = (
        "@get_name() -> str:\n"
        "  = Mert\n"
        "@say_hi(name: str = `get_name()`) -> str:\n"
        '  = `f"Hello, {name}!"`\n'
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    mapped = [
        region
        for region in virtual["ranges"]
        if region["text"] == "get_name()"
        and region["range"]["start"] == {"line": 2, "character": 21}
        and region["range"]["end"] == {"line": 2, "character": 31}
    ]
    assert len(mapped) == 1
    assert mapped[0]["virtualRange"]["start"]["line"] >= 0


def test_python_virtual_document_imports_typing_symbols_used_in_backtick_types():
    src = (
        '~Issue(line: int, severity: `Literal["critical", "low", "medium"]`)\n'
        "~Review(issues: `list[Issue]`)\n"
        "@review_code() -> Review:\n"
        "  = ok\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    assert "from typing import Any, Literal" in virtual["text"]
    assert 'severity: Literal["critical", "low", "medium"]' in virtual["text"]
    assert "issues: list[Issue]" in virtual["text"]


def test_python_virtual_document_models_kedi_proc_stubs_for_metrics():
    src = (
        "@get_country(city):\n"
        "  <city> is located in [country: str].\n"
        "  = <country>\n"
        "\n"
        "@eval: get_country:\n"
        "  > metric: accuracy:\n"
        "    = ```\n"
        '    result = get_country("France")\n'
        "    return (1.0, result)\n"
        "    ```\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def get_country(city: str) -> str: ..." in virtual["text"]
    assert "def __kedi_metric_get_country_accuracy() -> Any:" in virtual["text"]
    assert 'result = get_country("France")' in virtual["text"]
    assert any(r["kind"] == "fenced" for r in virtual["ranges"])
    assert any(
        s["kind"] == "procedure"
        and s["name"] == "get_country"
        and s["sourceRange"]["start"] == {"line": 0, "character": 1}
        for s in virtual["symbols"]
    )
    assert "    city: str = ..." in virtual["text"]
    assert 'result = get_country("France")' in virtual["text"]


def test_python_virtual_document_models_metric_scope_params_as_any():
    src = (
        "@solve_knapsack(problems, capacity):\n"
        "  = ok\n"
        "\n"
        "@eval: solve_knapsack:\n"
        "  > metric: accuracy(problems):\n"
        "    = ```\n"
        "    items_str, capacity = problems\n"
        "    pred = solve_knapsack(items_str, capacity)\n"
        "    return pred\n"
        "    ```\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_metric_solve_knapsack_accuracy() -> Any:" in virtual["text"]
    assert "    problems: Any = ..." in virtual["text"]
    assert "    capacity: str = ..." in virtual["text"]
    assert "    items_str, capacity = problems" in virtual["text"]
    assert "    pred = solve_knapsack(items_str, capacity)" in virtual["text"]


def test_python_virtual_helper_edges(monkeypatch):
    from kedi.lang.ast import (
        AssignStmt,
        BacktickLineStmt,
        EvalMetric,
        EvalSuite,
        OutputSegment,
        ProcedureDef,
        ProgramWithValidation,
        PythonBlock,
        PythonExprNativeSegment,
        Span,
        TestCase,
        TestSuite,
        TextSegment,
        TypeApply,
        TypeDef,
        TypeField,
        TypePython,
        TypeRef,
        TypeUnion,
    )
    from kedi.lsp import python_virtual as pv

    source = "```\n    # comment\n    value = 1\n```\n"
    builder = pv.VirtualBuilder(source, "/tmp/x.kedi")
    builder.add_symbol(
        kind="missing",
        name="x",
        source_range=None,
        virtual_range={
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
    )
    assert builder.symbols == []

    empty_region = pv.PythonRegion(
        kind="inline",
        range={
            "start": {"line": 99, "character": 0},
            "end": {"line": 99, "character": 0},
        },
        text="",
    )
    assert builder.add_mapped_region(empty_region)["start"]["line"] == 0
    assert "..." in builder.finish()

    comment_region = pv.PythonRegion(
        kind="fenced",
        range={
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 13},
        },
        text="# comment",
    )
    builder.add_mapped_region(comment_region, "    ")
    assert builder.lines[-1] == "    ..."
    assert builder._common_indent(99, 100) == 0

    reversed_region = pv.PythonRegion(
        kind="inline",
        range={
            "start": {"line": 2, "character": 9},
            "end": {"line": 2, "character": 2},
        },
        text="",
    )
    builder.add_mapped_region(reversed_region)

    region_cursor = pv.RegionCursor(
        [
            {
                "kind": "inline",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 1},
                },
                "text": "x",
            }
        ]
    )
    assert region_cursor.next("fenced") is None
    assert (
        region_cursor.next(
            "inline",
            Span(0, 1, 2, 1, 2, 2),
        )
        is None
    )

    stub_builder = pv.VirtualBuilder("", None)
    pv._emit_python_module_stub(stub_builder, "if")
    pv._emit_python_module_stub(stub_builder, "1 + 1")
    pv._emit_python_module_stub(
        stub_builder,
        "import os, sys as system\n"
        "from typing import *\n"
        "from pydantic import Field\n"
        "from models import User\n"
        "from package import value as alias\n"
        "class Empty:\n"
        "    pass\n"
        "class OnlyMethod:\n"
        "    def method(self):\n"
        "        return None\n"
        "class Model:\n"
        "    name: str\n"
        "    a = b = 1\n"
        "    count = 1\n"
        "    enabled: bool = Field(default=True)\n"
        "    required: int = Field(...)\n"
        "    async def run(self) -> str:\n"
        "        return 'ok'\n"
        "def fn(x: int) -> str:\n"
        "    return str(x)\n"
        "a = b = 1\n"
        "(c, [d, obj.attr]) = (1, [2, 3])\n",
    )
    text = stub_builder.finish()
    assert "os: Any = ..." in text
    assert "def Field(*args: Any, **kwargs: Any) -> Any: ..." in text
    assert "class User: ..." in text
    assert "class Empty:" in text and "    pass" in text
    assert "async def run" in text
    assert "c: Any = ..." in text and "d: Any = ..." in text
    field_call = cast(ast.Call, ast.parse("obj.Field(...)", mode="eval").body)
    factory_call = cast(ast.Call, ast.parse("factory()", mode="eval").body)
    assert pv._python_call_name(field_call.func) == "Field"
    assert pv._python_call_name(factory_call.func) == "factory"
    assert pv._unparse_or_any(None) == "Any"

    class BadNode(ast.AST):
        _fields = ("bad",)

        @property
        def bad(self):
            raise RuntimeError("bad")

    assert pv._unparse_or_any(BadNode()) == "Any"
    required_field = cast(ast.AnnAssign, ast.parse("x: int = Field(...)\n").body[0])
    defaulted_field = cast(ast.AnnAssign, ast.parse("x: int = Field(1)\n").body[0])
    assert pv._python_field_has_default(required_field.value) is False
    assert pv._python_field_has_default(defaulted_field.value) is True
    assert pv._python_field_has_default(ast.Constant(value=1)) is True
    assert pv._assigned_python_names(ast.parse("x: int\n").body[0]) == ["x"]
    multi_assign_class = cast(ast.ClassDef, ast.parse("class C:\n    a = b = 1\n").body[0])
    mixed_assign_class = cast(ast.ClassDef, ast.parse("class C:\n    self.a = b = 1\n").body[0])
    assert pv._python_class_fields(multi_assign_class) == [
        ("a", "Any", True),
        ("b", "Any", True),
    ]
    assert pv._python_class_fields(mixed_assign_class) == [("b", "Any", True)]
    assert (
        pv._single_python_segment([TextSegment(text="x"), PythonExprNativeSegment(code="1")])
        is None
    )
    assert pv._single_python_segment([TextSegment(text="x")]) is None
    assert pv._type_init_params(cast(Any, SimpleNamespace(fields=[]))) == "self"
    assert pv._indent_width("\t  x") == 6
    assert pv._python_call_name(ast.Constant(value=1)) == ""

    typing_symbols = pv._typing_symbols_in_type_expr(
        TypeUnion(
            left=TypeApply(name="Optional", args=[TypeRef(name="Literal")]),
            right=TypePython(code="Annotated[str, 'x']"),
        )
    )
    assert {
        ("typing", "Optional"),
        ("typing", "Literal"),
        ("typing", "Annotated"),
    } <= typing_symbols
    assert ("builtins", "str") in pv._typing_symbols_in_type_expr(
        TypeApply(name="Custom", args=[TypeRef(name="str")])
    )
    assert pv._typing_symbols_in_type_expr(cast(Any, TypeField(name="not-type"))) == set()
    assert pv._path_from_uri(None) is None
    assert pv._path_from_uri("untitled:test") is None
    assert pv._path_from_uri("file:///tmp/a%20b.kedi") == "/tmp/a b.kedi"

    empty_type_builder = pv.VirtualBuilder("", None)
    pv._emit_type_stub(empty_type_builder, TypeDef(name="Empty", fields=[]), {})
    assert "    pass" in empty_type_builder.finish()

    nested_builder = pv.VirtualBuilder("@outer():\n  @inner():\n    ###\n    doc\n    ###\n", None)
    inner = ProcedureDef(
        name="inner",
        params=[],
        body=[],
        docstring="doc",
        loc=Span(0, 1, 2, 3, 5, 8),
    )
    pv._emit_nested_procedure(nested_builder, pv.RegionCursor([]), inner, "    ", {})
    empty_inner = ProcedureDef(name="empty", params=[], body=[], loc=Span(0, 1, 1, 1, 1, 10))
    pv._emit_nested_procedure(nested_builder, pv.RegionCursor([]), empty_inner, "    ", {})
    assert '"""doc"""' in nested_builder.finish()
    assert "        pass" in nested_builder.finish()
    assert pv._docstring_lines("one line", "  ") == ['  """one line"""']

    assign = AssignStmt(
        name="x",
        expr=[PythonExprNativeSegment(code="1", loc=Span(0, 1, 1, 12, 1, 13))],
        loc=Span(0, 1, 1, 2, 1, 3),
    )
    emit_builder = pv.VirtualBuilder("[x: int] = `1`\n", None)
    pv._emit_assign_stmt(emit_builder, pv.RegionCursor([]), assign, {}, "    ")
    assert "x: str = (" in emit_builder.finish()
    plain_assign = AssignStmt(
        name="y",
        expr=[TextSegment(text="hello")],
        loc=Span(0, 1, 1, 2, 1, 3),
    )
    pv._emit_assign_stmt(emit_builder, pv.RegionCursor([]), plain_assign, {}, "    ")
    assert "y: str = ..." in emit_builder.finish()
    region_for_assign = {
        "kind": "inline",
        "range": {
            "start": {"line": 0, "character": 11},
            "end": {"line": 0, "character": 12},
        },
        "text": "1",
    }
    pv._emit_assign_stmt(emit_builder, pv.RegionCursor([region_for_assign]), assign, {}, "    ")

    stmt_builder = pv.VirtualBuilder("[a] = ok\n`x = 1`\n", None)
    pv._emit_statements(
        stmt_builder,
        pv.RegionCursor(
            [
                {
                    "kind": "inline",
                    "range": {
                        "start": {"line": 1, "character": 1},
                        "end": {"line": 1, "character": 6},
                    },
                    "text": "x = 1",
                }
            ]
        ),
        [
            AssignStmt(name="a", expr=[TextSegment(text="ok")], loc=Span(0, 1, 1, 2, 1, 3)),
            PythonBlock(code="x = 1", loc=Span(0, 1, 2, 1, 2, 7)),
            BacktickLineStmt(code="x = 1", loc=Span(0, 1, 2, 1, 2, 7)),
            BacktickLineStmt(code="y = 2", loc=Span(0, 1, 2, 1, 2, 7)),
        ],
        {},
        "    ",
        {},
    )
    assert "a: str = ..." in stmt_builder.finish()
    assert "x = 1" in stmt_builder.finish()
    pv._emit_statements(
        stmt_builder,
        pv.RegionCursor([]),
        [
            BacktickLineStmt(code="z = 3", loc=Span(0, 1, 2, 1, 2, 7)),
            pv.ReturnBlockStmt(code="return 1", loc=Span(0, 1, 2, 1, 2, 8)),
            pv.Stmt(),
        ],
        {},
        "    ",
        {},
    )

    validation_builder = pv.VirtualBuilder("`assert True`\n```\nDATA = []\n```\n", None)
    program = ProgramWithValidation(
        tests=[
            TestSuite(
                procedure_name="solve",
                cases=[
                    TestCase(
                        name="basic",
                        code="assert True",
                        is_block=False,
                        loc=Span(0, 1, 1, 1, 1, 12),
                    )
                ],
            )
        ],
        evals=[
            EvalSuite(
                procedure_name="solve",
                datasets={
                    "train": pv.EvalDataset(
                        name="train",
                        code="DATA = []",
                        is_block=True,
                        loc=Span(0, 1, 3, 1, 3, 10),
                    )
                },
                metrics=[EvalMetric(name="score", code="return 1.0")],
            )
        ],
    )
    pv._emit_validation(
        validation_builder,
        pv.RegionCursor(
            [
                {
                    "kind": "inline",
                    "range": {
                        "start": {"line": 0, "character": 1},
                        "end": {"line": 0, "character": 12},
                    },
                    "text": "assert True",
                },
                {
                    "kind": "fenced",
                    "range": {
                        "start": {"line": 2, "character": 0},
                        "end": {"line": 2, "character": 9},
                    },
                    "text": "DATA = []",
                },
            ]
        ),
        program,
        {},
    )
    validation_text = validation_builder.finish()
    assert "__kedi_test_solve_basic" in validation_text
    assert "__kedi_metric_solve_score" in validation_text
    assert "    pass" in validation_text
    metric_builder = pv.VirtualBuilder("", None)
    pv._emit_metric(
        metric_builder,
        pv.RegionCursor([]),
        EvalSuite(procedure_name="solve"),
        EvalMetric(name="score", code="return 1.0"),
        None,
    )
    assert "    pass" in metric_builder.finish()

    symbol_builder = pv.VirtualBuilder("", None)
    pv._add_scope_param_symbols(symbol_builder, {"x": pv.ScopeName("str", None)}, 0, "no params")
    pv._add_scope_param_symbols(symbol_builder, {"missing": pv.ScopeName("str", None)}, 0, "fn(x)")
    assert symbol_builder.symbols == []
    assert not pv._range_within_span(
        {"start": {"line": 0, "character": 0}, "end": {"line": 3, "character": 0}},
        Span(0, 1, 2, 1, 2, 5),
    )
    assert not pv._range_within_span(
        {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 99}},
        Span(0, 1, 2, 2, 2, 5),
    )
    assert pv._proc_name_range(ProcedureDef(name="p", params=[])) is None
    assert pv._type_name_range(TypeDef(name="T", fields=[])) is None
    assert pv._assign_name_range(AssignStmt(name="x")) is None
    assert pv._output_name_range(OutputSegment(name="x")) is None
    assert (
        pv._param_name_range(
            ["@p(x):"],
            ProcedureDef(name="p", params=["x"]),
            "x",
        )
        is None
    )
    assert (
        pv._param_name_range(
            [],
            ProcedureDef(name="p", params=["x"], loc=Span(0, 1, 1, 1, 1, 5)),
            "x",
        )
        is None
    )
    assert (
        pv._param_name_range(
            ["@p x):"],
            ProcedureDef(name="p", params=["x"], loc=Span(0, 1, 1, 1, 1, 5)),
            "x",
        )
        is None
    )
    assert (
        pv._param_name_range(
            ["@p(x"],
            ProcedureDef(name="p", params=["y"], loc=Span(0, 1, 1, 1, 1, 5)),
            "y",
        )
        is None
    )
    assert (
        pv._param_name_range(
            ["@p(x):"],
            ProcedureDef(name="p", params=["y"], loc=Span(0, 1, 1, 1, 1, 6)),
            "y",
        )
        is None
    )
    assert pv._container_line(ProcedureDef(name="p", params=[])) == -1
    pv._add_proc_default_regions(
        symbol_builder,
        ProcedureDef(
            name="p",
            params=["x"],
            param_defaults={"x": "1"},
            loc=Span(0, 1, 1, 1, 1, 6),
        ),
        signature_line=0,
        signature="def p(y: str = 1):",
        header_defaults={
            "param": {
                0: [
                    {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(pv, "_proc_param_to_python", lambda proc, name: "x: str")
    pv._add_proc_default_regions(
        symbol_builder,
        ProcedureDef(
            name="p",
            params=["x"],
            param_defaults={"x": "1"},
            loc=Span(0, 1, 1, 1, 1, 6),
        ),
        signature_line=0,
        signature="def p(x: str):",
        header_defaults={
            "param": {
                0: [
                    {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    }
                ]
            }
        },
    )
    monkeypatch.undo()
    pv._add_default_expr_region(
        symbol_builder,
        source_range={
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
        line=0,
        line_text="x: int = 1",
        expr="missing",
    )

    assert pv._find_deepest_procedure_at_line([ProcedureDef(name="p", params=[])], 1) is None
    assert pv._should_emit_full_module_python(None, 1) is False
    assert pv._indent_width("") == 0
    pv._fallback_virtual_document(
        "```\nvalue = 1\n```\n",
        [
            {
                "kind": "fenced",
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 1, "character": 9},
                },
                "text": "value = 1",
            }
        ],
    )

    monkeypatch.setattr(pv, "compute_python_ranges", lambda src: [])
    outside_focus = pv.compute_python_virtual_document(
        "@p():\n  = ok\n\n```\nSIDE = 2\n```\n",
        focus_line=99,
    )
    assert "SIDE: Any = ..." in outside_focus["text"]
    fallback = pv.compute_python_virtual_document("```\nvalue = 1\n```\n@bad(:\n")
    assert fallback["focusKey"] is None

    monkeypatch.setattr(pv, "compute_python_ranges", lambda src: [])
    no_region = pv.compute_python_virtual_document("```\nPRELUDE = 1\n```\n```\nSIDE = 2\n```\n")
    assert "PRELUDE = 1" in no_region["text"]
    assert "SIDE = 2" in no_region["text"]

    stdin = io.StringIO('{"text":"= ok\\n","uri":"file:///tmp/x.kedi","focusLine":0}')
    stdout = io.StringIO()
    monkeypatch.setattr(pv.sys, "stdin", stdin)
    monkeypatch.setattr(pv.sys, "stdout", stdout)
    pv.main()
    assert json.loads(stdout.getvalue())["focusKey"] is None

    stdin = io.StringIO('\n{"id":1,"text":"= ok\\n"}\nnot-json\n')
    stdout = io.StringIO()
    monkeypatch.setattr(pv.sys, "stdin", stdin)
    monkeypatch.setattr(pv.sys, "stdout", stdout)
    pv.main_loop()
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is False


# ---------------------------------------------------------------------
# Server construction smoke
# ---------------------------------------------------------------------
