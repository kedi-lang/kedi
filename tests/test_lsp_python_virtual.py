"""LSP feature tests."""

from __future__ import annotations

import ast
import io
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

import kedi.lang.ast as kedi_ast
import kedi.lsp.python_virtual.builder as virtual_builder_module
import kedi.lsp.python_virtual.document as virtual_document_module
import kedi.lsp.python_virtual.scope as virtual_scope_module
import kedi.lsp.python_virtual.statements as virtual_statements_module
import kedi.lsp.python_virtual.stubs as virtual_stubs_module
from kedi.lsp.python_virtual.document import compute_python_virtual_document


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
    assert (
        "def calculate_stats(numbers: list[int]) -> int: return cast(int, _kedi_placeholder())"
    ) in text
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


def test_python_virtual_document_models_procedure_local_type():
    src = (
        "@outer() -> str:\n"
        "  ~Inner(name: str)\n"
        "  [item: Inner] = ```\n"
        '  return Inner(name="Ada")\n'
        "  ```\n"
        "  = <`item.name`>\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/local_type.kedi")
    ast.parse(virtual["text"])
    assert "    class Inner:" in virtual["text"]
    assert "        name: str" in virtual["text"]
    assert "        def __init__(self, name: str) -> None: ..." in virtual["text"]
    assert 'return Inner(name="Ada")' in virtual["text"]
    assert any(
        symbol["kind"] == "type"
        and symbol["name"] == "Inner"
        and symbol["sourceRange"]["start"] == {"line": 1, "character": 3}
        for symbol in virtual["symbols"]
    )


def test_python_virtual_document_exposes_optimize_block_outputs_to_fenced_python():
    src = (
        "@solve_knapsack(items: str, capacity: int) -> int:\n"
        "  > optimize: solver:\n"
        "    >> <items>\n"
        "    Knapsack capacity: <capacity>\n"
        "    Reasoning: [reasoning: str]\n"
        "    Total value: [value: int]\n"
        "  = ```\n"
        '  print(f"Reasoning for {items}:\\n\\n", reasoning)\n'
        "  return value\n"
        "  ```\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/knapsack.kedi", focus_line=7)

    assert "reasoning: str = cast(str, _kedi_placeholder())" in virtual["text"]
    assert "value: int = cast(int, _kedi_placeholder())" in virtual["text"]
    assert 'print(f"Reasoning for {items}:\\n\\n", reasoning)' in virtual["text"]
    assert "return value" in virtual["text"]
    assert {(symbol["name"], symbol["kind"]) for symbol in virtual["symbols"]} >= {
        ("reasoning", "local"),
        ("value", "local"),
    }


def test_python_virtual_document_maps_native_inline_toplevel_return() -> None:
    src = (
        "@solve_knapsack(items: str, capacity: int) -> int:\n"
        "  = `capacity`\n"
        "\n"
        '= `solve_knapsack("A", 10)`\n'
    )

    virtual = compute_python_virtual_document(src, "/tmp/knapsack.kedi", focus_line=3)

    assert 'solve_knapsack("A", 10)' in virtual["text"]
    assert any(
        mapping["kind"] == "inline"
        and mapping["sourceRange"]["start"] == {"line": 3, "character": 3}
        for mapping in virtual["mappings"]
    )


def test_python_virtual_document_defaults_untyped_procedure_params_to_str_in_python_blocks():
    src = '@get_capital(country) -> str:\n  ```\n  print("called", country)\n  ```\n  = Stockholm\n'
    virtual = compute_python_virtual_document(src, "/tmp/sys.kedi")
    ast.parse(virtual["text"])

    assert "def __kedi_scope_get_capital(country: str) -> str:" in virtual["text"]
    assert 'print("called", country)' in virtual["text"]
    assert any(
        symbol["kind"] == "parameter"
        and symbol["name"] == "country"
        and symbol["sourceRange"]["start"] == {"line": 0, "character": 13}
        for symbol in virtual["symbols"]
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


def test_python_virtual_document_wraps_template_python_substitutions_in_initializations():
    src = "@greet(name: str) -> str:\n  [upper: str] = `name.upper()`\n  = Hello <`upper`>\n"

    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")

    assert "__kedi_inline_expression_" in virtual["text"]
    assert "        upper" in virtual["text"]
    assert any(
        mapping["sourceRange"]["start"] == {"line": 2, "character": 12}
        for mapping in virtual["mappings"]
    )


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
    assert (
        "def alpha(items: list[int]) -> int: return cast(int, _kedi_placeholder())"
        in virtual["text"]
    )
    assert "def beta(name: str) -> str: return cast(str, _kedi_placeholder())" in virtual["text"]
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
    assert "def __kedi_scope_foo() -> str:" in virtual["text"]
    assert "def bar(name: str) -> str:" in virtual["text"]
    assert "return name.upper()" in virtual["text"]


def test_nested_python_scope_inherits_outer_kedi_locals() -> None:
    src = (
        "@outer(value: str) -> str:\n"
        "  @inner() -> str:\n"
        "    ```\n"
        "    return value.upper()\n"
        "    ```\n"
        "    = <value>\n"
        "  = <inner()>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=3)
    ast.parse(virtual["text"])

    assert "def __kedi_scope_outer(value: str) -> str:" in virtual["text"]
    assert "    def inner() -> str:" in virtual["text"]
    assert "        return value.upper()" in virtual["text"]
    assert "def __kedi_scope_inner" not in virtual["text"]


def test_python_virtual_document_emits_literal_toplevel_return():
    src = "= done\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def __kedi_toplevel__() -> Any:" in virtual["text"]
    assert "    return cast(Any, _kedi_placeholder())" in virtual["text"]


def test_python_virtual_document_includes_prelude_symbols_for_toplevel_scope():
    src = (
        "```\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Person(BaseModel):\n"
        '    name: str = Field(examples=["Ayşe", "Ali", "Berk"])\n'
        "    age: int = Field(gt=50)\n"
        "```\n\n"
        ">> Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "class Person(BaseModel):" in virtual["text"]
    assert "person: Person = cast(Person, _kedi_placeholder())" in virtual["text"]
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
        ">> Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=9)
    ast.parse(virtual["text"])
    assert "from pydantic import BaseModel, Field" not in virtual["text"]
    assert "class BaseModel: pass" in virtual["text"]
    assert (
        "def Field(*args: Any, **kwargs: Any) -> Any: return cast(Any, _kedi_placeholder())"
    ) in virtual["text"]
    assert "class Person:" in virtual["text"]
    assert "def __init__(self, name: str, age: int) -> None: ..." in virtual["text"]
    assert "person: Person = cast(Person, _kedi_placeholder())" in virtual["text"]


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
    assert (
        "args: AttrDict[str, ArgType] = cast(AttrDict[str, ArgType], _kedi_placeholder())"
    ) in virtual["text"]
    assert "args.city" in virtual["text"]
    assert "args.debug" in virtual["text"]


def test_python_virtual_document_exposes_imported_python_exports_in_python_blocks(
    tmp_path,
):
    source_path = tmp_path / "main.kedi"
    src = "> import: require\n\n```\nrequire('pydantic_monty', required_by='Sandbox')\n```\n"

    virtual = compute_python_virtual_document(src, str(source_path))

    ast.parse(virtual["text"])
    assert "def require(package_names: str | list[str]," in virtual["text"]
    assert "required_by: str | None=None) -> bool: ..." in virtual["text"]
    assert "require('pydantic_monty', required_by='Sandbox')" in virtual["text"]


def test_python_virtual_document_exposes_imported_kedi_and_python_symbols(tmp_path):
    module_path = tmp_path / "profiles.kedi"
    module_path.write_text(
        "```\n"
        "async def fetch_profile_id() -> int:\n"
        "    return 1\n"
        "\n"
        "class Helper:\n"
        "    pass\n"
        "```\n"
        "\n"
        "~Profile(name: str)\n"
        "@make_profile(name: str) -> Profile:\n"
        "  = `Profile(name)`\n"
        "[profile_id: int] = `1`\n"
        "\n"
        "> export:\n"
        "  fetch_profile_id\n"
        "  Helper\n"
        "  Profile\n"
        "  make_profile\n"
        "  profile_id\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "main.kedi"
    src = (
        "> import: profiles\n\n"
        "```\n"
        'profile = make_profile("Ada")\n'
        "pid = profile_id\n"
        "helper = Helper()\n"
        "```\n"
    )

    virtual = compute_python_virtual_document(src, str(source_path))

    ast.parse(virtual["text"])
    assert "async def fetch_profile_id() -> int: ..." in virtual["text"]
    assert "class Helper:\n    pass" in virtual["text"]
    assert "class Profile:" in virtual["text"]
    assert (
        "def make_profile(name: str) -> Profile: return cast(Profile, _kedi_placeholder())"
    ) in virtual["text"]
    assert "profile_id: int = cast(int, _kedi_placeholder())" in virtual["text"]
    assert 'profile = make_profile("Ada")' in virtual["text"]
    assert any(symbol["kind"] == "imported_value" for symbol in virtual["symbols"])


def test_python_virtual_document_exposes_only_selective_imports(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "helpers.kedi").write_text(
        "@selected(name: str) -> str:\n"
        "  = <name>\n"
        "@excluded() -> str:\n"
        "  = excluded\n"
        "> export:\n"
        "  selected\n"
        "  excluded\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "main.kedi"
    source = '> import: package/helpers:\n  selected\n\n```\nresult = selected("Ada")\n```\n'

    virtual = compute_python_virtual_document(source, str(source_path))

    ast.parse(virtual["text"])
    assert "def selected(name: str) -> str:" in virtual["text"]
    assert "def excluded(" not in virtual["text"]
    assert 'result = selected("Ada")' in virtual["text"]


def test_python_virtual_document_preserves_imported_python_type_metadata(tmp_path):
    module_path = tmp_path / "helpers.kedi"
    module_path.write_text(
        "```\n"
        "from typing import Annotated, Optional\n"
        "from datetime import datetime\n"
        "\n"
        "class Record:\n"
        "    label: Annotated[str, 'Display label']\n"
        "    alias: Optional[str]\n"
        "    created_at: datetime\n"
        "\n"
        "def fetch_record(identifier: int, *, include_archived: bool = False) -> Record:\n"
        "    return Record()\n"
        "\n"
        "endpoint: str = 'https://example.test'\n"
        "untyped_value = object()\n"
        "```\n\n"
        "> export:\n"
        "  Record\n"
        "  fetch_record\n"
        "  endpoint\n"
        "  untyped_value\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "main.kedi"
    source = (
        "> import: helpers\n\n"
        "```\n"
        "record = fetch_record(1)\n"
        "label = record.label\n"
        "url = endpoint\n"
        "unknown = untyped_value\n"
        "```\n"
    )

    virtual = compute_python_virtual_document(source, str(source_path))
    ast.parse(virtual["text"])

    assert "class Record:\n    label: Annotated[str, 'Display label']" in virtual["text"]
    assert "alias: Optional[str]" in virtual["text"]
    assert "created_at: datetime" in virtual["text"]
    assert "from typing import Annotated, Any, Optional" in virtual["text"]
    assert "from datetime import datetime" in virtual["text"]
    assert (
        "def fetch_record(identifier: int, *, include_archived: bool=False) -> Record: ..."
        in virtual["text"]
    )
    assert "endpoint: str = cast(str, _kedi_placeholder())" in virtual["text"]
    assert "untyped_value: object = cast(object, _kedi_placeholder())" in virtual["text"]
    assert "record = fetch_record(1)" in virtual["text"]


def test_python_virtual_type_import_discovery_handles_partial_and_star_imports() -> None:
    from kedi.lang import parse_program

    program = parse_program('~Request(note: Annotated[str, "Human note"])\n')

    assert virtual_stubs_module._typing_import_lines(program, None) == [
        "from typing import Annotated, Any"
    ]
    assert (
        virtual_stubs_module._typing_import_lines(
            program,
            None,
            ("from typing import *", "this is not valid Python:"),
        )
        == []
    )
    assert virtual_stubs_module._typing_symbols_in_python_stub("not valid Python:") == set()
    assert (
        "typing",
        "Optional",
    ) in virtual_stubs_module._typing_symbols_in_python_stub(
        "def normalize(value: Optional[str]): ..."
    )


def test_python_virtual_document_orders_prelude_before_kedi_signatures() -> None:
    src = (
        "```\n"
        "from typing import Annotated\n"
        "DEFAULT_LIMIT: int = 3\n"
        "```\n\n"
        '~Request(note: Annotated[str, "Human note"])\n'
        "@run(limit: int = `DEFAULT_LIMIT`) -> str:\n"
        "  = ok\n"
    )

    full = compute_python_virtual_document(src, "/tmp/x.kedi")
    focused = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=6)

    ast.parse(full["text"])
    ast.parse(focused["text"])
    assert full["text"].index("DEFAULT_LIMIT: int = 3") < full["text"].index(
        "def run(limit: int = DEFAULT_LIMIT)"
    )
    assert "from typing import Annotated" in focused["text"]
    assert full["text"].count("from typing import Annotated") == 1
    assert focused["text"].count("from typing import Annotated") == 1
    assert "class Annotated" not in focused["text"]
    assert "DEFAULT_LIMIT: int = cast(int, _kedi_placeholder())" in focused["text"]


def test_python_virtual_document_exposes_toplevel_values_to_procedure_python() -> None:
    src = (
        "[top_inline: int] = `1`\n"
        "[top_block: str] = ```\n"
        "return str(top_inline)\n"
        "```\n\n"
        "@read_top(prefix) -> str:\n"
        "  [value: str] = ```\n"
        '  return f"{prefix}:{top_block}:{top_inline}"\n'
        "  ```\n"
        "  = <value>\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=7)

    ast.parse(virtual["text"])
    assert "top_inline: int = cast(int, _kedi_placeholder())" in virtual["text"]
    assert "top_block: str = cast(str, _kedi_placeholder())" in virtual["text"]
    assert 'return f"{prefix}:{top_block}:{top_inline}"' in virtual["text"]


def test_python_virtual_document_maps_inline_python_nested_in_call_arguments() -> None:
    src = (
        "@compose(topic):\n  >> <compose(`topic`)> <compose(<`topic.upper()`>)> <`topic.lower()`>\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/x.kedi", focus_line=1)

    assert [region["text"] for region in virtual["ranges"]] == [
        "topic",
        "topic.upper()",
        "topic.lower()",
    ]
    assert virtual["text"].count("__kedi_inline_expression_") == 3


def test_python_virtual_document_maps_control_flow_headers_and_bodies() -> None:
    src = (
        "[total: int] = `0`\n"
        "> if: `total == 0`:\n"
        "  > loop [n]: `range(1, 4)`:\n"
        "    `total += n`\n"
        "> else:\n"
        "  `total = -1`\n"
        "= `total`\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/control.kedi")

    ast.parse(virtual["text"])
    assert [region["text"] for region in virtual["ranges"]] == [
        "0",
        "total == 0",
        "range(1, 4)",
        "total += n",
        "total = -1",
        "total",
    ]
    assert "    if (" in virtual["text"]
    assert "        for n in (" in virtual["text"]
    assert "        del n" in virtual["text"]
    assert any(
        symbol["name"] == "n" and symbol["sourceRange"]["start"] == {"line": 2, "character": 10}
        for symbol in virtual["symbols"]
    )


def test_python_virtual_document_maps_template_and_conditional_loop_headers() -> None:
    src = (
        "[city] = Ankara\n"
        "> if: <city> has <`len(city)`> letters\n"
        "  [status] = yes\n"
        "[remaining: int] = `2`\n"
        "> loop: `remaining > 0`:\n"
        "  `remaining -= 1`\n"
        "> loop: <remaining> is below <`3`>\n"
        "  `remaining += 1`\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/control.kedi")

    ast.parse(virtual["text"])
    mapped = [region["text"] for region in virtual["ranges"]]
    assert "len(city)" in mapped
    assert "remaining > 0" in mapped
    assert "3" in mapped
    assert virtual["text"].count("while (") == 2


def test_python_virtual_map_scope_and_assignment_do_not_leak_or_shadow() -> None:
    src = (
        "[total: int] = `0`\n"
        "> loop [item]: `[1, 2]`:\n"
        "  [double: int] = `item * 2`\n"
        "> map:\n"
        "  [total] := `total + double`\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/map.kedi")
    text = virtual["text"]
    module_stubs = text.split("def __kedi_toplevel__", 1)[0]

    ast.parse(text)
    assert "double:" not in module_stubs
    assert "        total\n" in text
    assert "__kedi_assigned_total_5: int = (" in text
    assert "            total + double" in text


def test_python_virtual_emits_procedure_map_and_assignment_variants() -> None:
    src = (
        "@collect() -> int:\n"
        "  [total: int] = `0`\n"
        "  > loop [item]: `[1, 2]`:\n"
        "    @identity() -> int:\n"
        "      = `item`\n"
        "    [double: int] = `item * 2`\n"
        "  > map:\n"
        "    [total] := `total + double`\n"
        "  > loop: `False`:\n"
        "    [unused: int] = `0`\n"
        "  [total] := ```\n"
        "  return total\n"
        "  ```\n"
        "  [total] := unchanged\n"
        "  = `total`\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/procedure-map.kedi")

    ast.parse(virtual["text"])
    assert "def __kedi_assign_block_total_11" in virtual["text"]
    assert "__kedi_assigned_total_14: int" in virtual["text"]
    assert "def identity() -> int:" in virtual["text"]


def test_python_virtual_control_flow_defensive_fallbacks_and_scope_isolation() -> None:
    from kedi.lang.ast import (
        AssignmentBlockStmt,
        AssignmentStmt,
        ConditionalLoopStmt,
        IfStmt,
        LoopStmt,
        ProcDefStmt,
        ProcedureDef,
        PythonExprNativeSegment,
        TextSegment,
    )

    builder = virtual_builder_module.VirtualBuilder("", None)
    empty_if = IfStmt(condition_code="", body=[], else_body=[])
    virtual_statements_module._emit_if_stmt(
        builder, virtual_builder_module.RegionCursor([]), empty_if, {}, "", {}
    )
    assert "cast(bool, _kedi_placeholder())" in builder.finish()
    assert builder.lines.count("    pass") == 2

    no_else = IfStmt(condition_code="", body=[])
    virtual_statements_module._emit_if_stmt(
        builder, virtual_builder_module.RegionCursor([]), no_else, {}, "", {}
    )
    assert list(virtual_scope_module._iter_control_statements([no_else])) == [no_else]

    loop_scope = {"n": virtual_builder_module.ScopeName(typ="str", source_range=None)}
    empty_loop = LoopStmt(binder="n", iterable_code="", body=[])
    virtual_statements_module._emit_loop_stmt(
        builder,
        virtual_builder_module.RegionCursor([]),
        empty_loop,
        loop_scope,
        "",
        {},
    )
    text = builder.finish()
    assert "cast(Iterable[Any], _kedi_placeholder())" in text
    assert "n: str = cast(str, _kedi_placeholder())" in text
    assert loop_scope["n"].typ == "str"

    empty_conditional_loop = ConditionalLoopStmt(condition_code="", body=[])
    virtual_statements_module._emit_conditional_loop_stmt(
        builder,
        virtual_builder_module.RegionCursor([]),
        empty_conditional_loop,
        {},
        "",
        {},
    )
    text = builder.finish()
    assert "cast(bool, _kedi_placeholder())" in text
    assert text.endswith("    pass\n")

    nested = ProcedureDef(name="nested", params=[])
    statements = list(
        virtual_scope_module._iter_control_statements(
            [IfStmt(condition_code="", body=[], else_body=[ProcDefStmt(proc=nested)])]
        )
    )
    assert any(isinstance(stmt, ProcDefStmt) for stmt in statements)

    nested_loop = LoopStmt(
        binder="item",
        body=[ProcDefStmt(proc=nested)],
        map_body=[AssignmentStmt(name="value", expr=[TextSegment(text="x")])],
    )
    conditional = ConditionalLoopStmt(
        condition_code="",
        body=[ProcDefStmt(proc=nested)],
    )
    walked = list(virtual_scope_module._iter_control_statements([nested_loop, conditional]))
    assert len(walked) == 5
    outer = ProcedureDef(name="outer", params=[], body=[nested_loop])
    assert [proc.name for proc in virtual_scope_module._all_procedures([outer])] == [
        "outer",
        "nested",
    ]

    fallback_builder = virtual_builder_module.VirtualBuilder("", None)
    fallback_cursor = virtual_builder_module.RegionCursor([])
    virtual_statements_module._emit_assignment_block(
        fallback_builder,
        fallback_cursor,
        AssignmentBlockStmt(name="value", code="return 1"),
        {"value": virtual_builder_module.ScopeName(typ="int", source_range=None)},
        "",
    )
    virtual_statements_module._emit_assignment_stmt(
        fallback_builder,
        fallback_cursor,
        AssignmentStmt(
            name="value",
            expr=[PythonExprNativeSegment(code="1")],
        ),
        {"value": virtual_builder_module.ScopeName(typ="int", source_range=None)},
        "",
    )
    assert "_kedi_placeholder" in fallback_builder.finish()


def test_python_virtual_document_maps_directive_inline_python() -> None:
    src = (
        "```\n"
        "endpoint: str = 'https://example.test/mcp'\n"
        "limit: int = 128\n"
        "```\n"
        "> mcp:\n"
        "    transport: http\n"
        "    url: `endpoint`\n"
        "> settings:\n"
        "    max_tokens: `limit`\n"
        "> profile: analyst:\n"
        "    > settings:\n"
        "        max_tokens: `limit`\n"
    )

    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")

    assert [region["text"] for region in virtual["ranges"]] == [
        "endpoint: str = 'https://example.test/mcp'\nlimit: int = 128",
        "limit",
        "endpoint",
        "limit",
    ]


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
        ">> Fake me a [person: Person]\n"
        "\n"
        "= <`person`>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "import this" in virtual["text"]
    assert "class Person(BaseModel):" in virtual["text"]
    assert "def __kedi_toplevel__() -> Any:" in virtual["text"]
    assert "    person: Person = cast(Person, _kedi_placeholder())" in virtual["text"]


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
    assert "    expected: Any = _kedi_placeholder()" in virtual["text"]
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
    assert (
        "def make(name: str, salary: int = 0) -> Person: return cast(Person, _kedi_placeholder())"
    ) in virtual["text"]
    assert "def __kedi_scope_make(name: str, salary: int = 0) -> Person:" in virtual["text"]
    assert any(region["text"] == "0" for region in virtual["ranges"])


def test_python_virtual_document_defaults_untyped_proc_params_to_str():
    src = "@greet(name):\n  = ```\n  return name.upper()\n  ```\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")
    ast.parse(virtual["text"])
    assert "def greet(name: str) -> str: return cast(str, _kedi_placeholder())" in virtual["text"]
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


def test_python_virtual_document_preserves_annotated_metadata():
    src = (
        '~Person(name: Annotated[str, "Full name"])\n'
        "> show: ok\n\n"
        '@p(name: Annotated[str, "Full name"]) -> str:\n'
        "  = <name>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/annotated.kedi")

    assert 'name: Annotated[str, "Full name"]' in virtual["text"]
    assert "from typing import Annotated" in virtual["text"]


def test_python_virtual_document_imports_datetime_types():
    src = "@p(started_at: datetime) -> timedelta:\n  = <started_at>\n"
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")

    assert "from datetime import datetime, timedelta" in virtual["text"]
    assert "started_at: datetime" in virtual["text"]
    assert "-> timedelta" in virtual["text"]


def test_python_virtual_document_imports_runtime_type_aliases():
    src = (
        "@p(day: date, clock: time, matcher: Regex, email: Email, site: HttpUrl, file: FileUrl):\n"
        "  = <email>\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")

    assert "from datetime import date, time" in virtual["text"]
    assert "from re import Pattern as Regex" in virtual["text"]
    assert "from pydantic import EmailStr as Email, FileUrl, HttpUrl" in virtual["text"]
    assert "matcher: Regex" in virtual["text"]
    assert "email: Email" in virtual["text"]
    assert "site: HttpUrl" in virtual["text"]
    assert "file: FileUrl" in virtual["text"]


def test_python_virtual_document_imports_approval_types():
    src = (
        "@approve(request: ApprovalRequest) -> ApprovalDecision:\n  = `ApprovalDecision.allow()`\n"
    )
    virtual = compute_python_virtual_document(src, "/tmp/x.kedi")

    assert "from kedi import ApprovalDecision, ApprovalRequest" in virtual["text"]
    assert "request: ApprovalRequest" in virtual["text"]
    assert "-> ApprovalDecision" in virtual["text"]


def test_python_virtual_document_models_kedi_proc_stubs_for_metrics():
    src = (
        "@get_country(city):\n"
        "  >> <city> is located in [country: str].\n"
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
    assert ("def get_country(city: str) -> str: return cast(str, _kedi_placeholder())") in virtual[
        "text"
    ]
    assert "def __kedi_metric_get_country_accuracy() -> Any:" in virtual["text"]
    assert 'result = get_country("France")' in virtual["text"]
    assert any(r["kind"] == "fenced" for r in virtual["ranges"])
    assert any(
        s["kind"] == "procedure"
        and s["name"] == "get_country"
        and s["sourceRange"]["start"] == {"line": 0, "character": 1}
        for s in virtual["symbols"]
    )
    assert "    city: str = cast(str, _kedi_placeholder())" in virtual["text"]
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
    assert "    problems: Any = _kedi_placeholder()" in virtual["text"]
    assert "    capacity: str = cast(str, _kedi_placeholder())" in virtual["text"]
    assert "    items_str, capacity = problems" in virtual["text"]
    assert "    pred = solve_knapsack(items_str, capacity)" in virtual["text"]


def test_python_virtual_helper_edges(monkeypatch):
    from kedi.lang.ast import (
        BacktickLineStmt,
        CallSegment,
        EvalMetric,
        EvalSuite,
        LiteralArg,
        OutputSegment,
        ProcedureDef,
        ProgramWithValidation,
        PythonBlock,
        PythonExprNativeSegment,
        PythonExprSegment,
        RawInvokeStmt,
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
        VarArg,
        VariableInitStmt,
    )

    source = "```\n    # comment\n    value = 1\n```\n"
    builder = virtual_builder_module.VirtualBuilder(source, "/tmp/x.kedi")
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

    empty_region = virtual_builder_module.PythonRegion(
        kind="inline",
        range={
            "start": {"line": 99, "character": 0},
            "end": {"line": 99, "character": 0},
        },
        text="",
    )
    assert builder.add_mapped_region(empty_region)["start"]["line"] == 0
    assert "..." in builder.finish()

    comment_region = virtual_builder_module.PythonRegion(
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

    reversed_region = virtual_builder_module.PythonRegion(
        kind="inline",
        range={
            "start": {"line": 2, "character": 9},
            "end": {"line": 2, "character": 2},
        },
        text="",
    )
    builder.add_mapped_region(reversed_region)

    region_cursor = virtual_builder_module.RegionCursor(
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

    stub_builder = virtual_builder_module.VirtualBuilder("", None)
    virtual_stubs_module._emit_python_module_stub(stub_builder, "if")
    virtual_stubs_module._emit_python_module_stub(stub_builder, "1 + 1")
    virtual_stubs_module._emit_python_module_stub(
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
        "class ExplicitInit:\n"
        "    name: str\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "def fn(x: int) -> str:\n"
        "    return str(x)\n"
        "a = b = 1\n"
        "(c, [d, obj.attr]) = (1, [2, 3])\n",
    )
    text = stub_builder.finish()
    assert "os: Any = cast(Any, _kedi_placeholder())" in text
    assert (
        "def Field(*args: Any, **kwargs: Any) -> Any: return cast(Any, _kedi_placeholder())"
    ) in text
    assert "class User: pass" in text
    assert "class Empty:" in text and "    pass" in text
    assert "async def run" in text
    assert text.count("def __init__(self, name: str) -> None") == 1
    assert "c: Any = cast(Any, _kedi_placeholder())" in text
    assert "d: Any = cast(Any, _kedi_placeholder())" in text
    field_call = cast(ast.Call, ast.parse("obj.Field(...)", mode="eval").body)
    factory_call = cast(ast.Call, ast.parse("factory()", mode="eval").body)
    assert virtual_stubs_module._python_call_name(field_call.func) == "Field"
    assert virtual_stubs_module._python_call_name(factory_call.func) == "factory"
    assert virtual_scope_module._unparse_or_any(None) == "Any"

    class BadNode(ast.AST):
        _fields = ("bad",)

        @property
        def bad(self):
            raise RuntimeError("bad")

    assert virtual_scope_module._unparse_or_any(BadNode()) == "Any"
    required_field = cast(ast.AnnAssign, ast.parse("x: int = Field(...)\n").body[0])
    defaulted_field = cast(ast.AnnAssign, ast.parse("x: int = Field(1)\n").body[0])
    assert virtual_stubs_module._python_field_has_default(required_field.value) is False
    assert virtual_stubs_module._python_field_has_default(defaulted_field.value) is True
    assert virtual_stubs_module._python_field_has_default(ast.Constant(value=1)) is True
    assert virtual_stubs_module._assigned_python_names(ast.parse("x: int\n").body[0]) == ["x"]
    multi_assign_class = cast(ast.ClassDef, ast.parse("class C:\n    a = b = 1\n").body[0])
    mixed_assign_class = cast(ast.ClassDef, ast.parse("class C:\n    self.a = b = 1\n").body[0])
    assert virtual_stubs_module._python_class_fields(multi_assign_class) == [
        ("a", "Any", True),
        ("b", "Any", True),
    ]
    assert virtual_stubs_module._python_class_fields(mixed_assign_class) == [("b", "Any", True)]
    assert (
        virtual_statements_module._single_python_segment(
            [TextSegment(text="x"), PythonExprNativeSegment(code="1")]
        )
        is None
    )
    assert virtual_statements_module._single_python_segment([TextSegment(text="x")]) is None
    assert virtual_stubs_module._type_init_params(cast(Any, SimpleNamespace(fields=[]))) == "self"
    assert virtual_builder_module._indent_width("\t  x") == 6
    assert virtual_stubs_module._python_call_name(ast.Constant(value=1)) == ""

    typing_symbols = virtual_stubs_module._typing_symbols_in_type_expr(
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
    assert (
        "builtins",
        "str",
    ) in virtual_stubs_module._typing_symbols_in_type_expr(
        TypeApply(name="Custom", args=[TypeRef(name="str")])
    )
    assert (
        virtual_stubs_module._typing_symbols_in_type_expr(cast(Any, TypeField(name="not-type")))
        == set()
    )
    assert virtual_document_module._path_from_uri(None) is None
    assert virtual_document_module._path_from_uri("untitled:test") is None
    assert virtual_document_module._path_from_uri("file:///tmp/a%20b.kedi") == "/tmp/a b.kedi"

    empty_type_builder = virtual_builder_module.VirtualBuilder("", None)
    virtual_stubs_module._emit_type_stub(empty_type_builder, TypeDef(name="Empty", fields=[]), {})
    assert "    pass" in empty_type_builder.finish()

    nested_builder = virtual_builder_module.VirtualBuilder(
        "@outer():\n  @inner():\n    ###\n    >> doc\n    ###\n", None
    )
    inner = ProcedureDef(
        name="inner",
        params=[],
        body=[],
        docstring="doc",
        loc=Span(0, 1, 2, 3, 5, 8),
    )
    virtual_statements_module._emit_nested_procedure(
        nested_builder,
        virtual_builder_module.RegionCursor([]),
        inner,
        {},
        "    ",
        {},
    )
    empty_inner = ProcedureDef(name="empty", params=[], body=[], loc=Span(0, 1, 1, 1, 1, 10))
    virtual_statements_module._emit_nested_procedure(
        nested_builder,
        virtual_builder_module.RegionCursor([]),
        empty_inner,
        {},
        "    ",
        {},
    )
    assert '"""doc"""' in nested_builder.finish()
    assert "        return cast(str, _kedi_placeholder())" in nested_builder.finish()
    assert virtual_stubs_module._docstring_lines("one line", "  ") == ['  """one line"""']

    assign = VariableInitStmt(
        name="x",
        expr=[PythonExprNativeSegment(code="1", loc=Span(0, 1, 1, 12, 1, 13))],
        loc=Span(0, 1, 1, 2, 1, 3),
    )
    emit_builder = virtual_builder_module.VirtualBuilder("[x: int] = `1`\n", None)
    virtual_statements_module._emit_variable_init_stmt(
        emit_builder, virtual_builder_module.RegionCursor([]), assign, {}, "    "
    )
    assert "x: str = (" in emit_builder.finish()
    plain_assign = VariableInitStmt(
        name="y",
        expr=[TextSegment(text="hello")],
        loc=Span(0, 1, 1, 2, 1, 3),
    )
    virtual_statements_module._emit_variable_init_stmt(
        emit_builder,
        virtual_builder_module.RegionCursor([]),
        plain_assign,
        {},
        "    ",
    )
    assert "y: str = cast(str, _kedi_placeholder())" in emit_builder.finish()
    region_for_assign = {
        "kind": "inline",
        "range": {
            "start": {"line": 0, "character": 11},
            "end": {"line": 0, "character": 12},
        },
        "text": "1",
    }
    virtual_statements_module._emit_variable_init_stmt(
        emit_builder,
        virtual_builder_module.RegionCursor([region_for_assign]),
        assign,
        {},
        "    ",
    )

    import_builder = virtual_builder_module.VirtualBuilder("", None)
    assert virtual_stubs_module._emit_python_stub_stmt(
        import_builder, ast.parse("import typing").body[0]
    )
    assert (
        virtual_stubs_module._emit_python_stub_stmt(
            import_builder,
            ast.parse("from package import *").body[0],
        )
        is False
    )
    assert "import typing" in import_builder.finish()
    assert (
        virtual_stubs_module._checker_safe_value_stub("not valid Python !") == "not valid Python !"
    )

    no_region_block = kedi_ast.VariableInitBlockStmt(
        name="blocked",
        code="return 1",
        type_annotation="int",
        loc=Span(0, 1, 1, 1, 1, 10),
    )
    virtual_statements_module._emit_variable_init_block(
        emit_builder,
        virtual_builder_module.RegionCursor([]),
        no_region_block,
        {},
        "    ",
    )
    assert "blocked: int" in emit_builder.finish()

    nested_call = CallSegment(
        name="outer",
        args=[
            VarArg(name="plain"),
            LiteralArg(native_backtick="native", loc=Span(0, 1, 1, 1, 1, 2)),
            LiteralArg(
                python_segment=PythonExprSegment(
                    code="python_value",
                    loc=Span(0, 1, 1, 3, 1, 15),
                )
            ),
            LiteralArg(
                call_segment=CallSegment(
                    name="inner",
                    args=[LiteralArg(native_backtick="deep", loc=Span(0, 1, 1, 16, 1, 20))],
                )
            ),
            LiteralArg(
                render_segments=[
                    PythonExprNativeSegment(code="rendered", loc=Span(0, 1, 1, 21, 1, 29)),
                    CallSegment(
                        name="render_call",
                        args=[
                            LiteralArg(
                                python_segment=PythonExprSegment(
                                    code="nested_render",
                                    loc=Span(0, 1, 1, 30, 1, 43),
                                )
                            )
                        ],
                    ),
                ]
            ),
        ],
    )
    assert len(list(virtual_statements_module._call_inline_python_locations(nested_call))) == 5
    virtual_statements_module._emit_segments(
        emit_builder,
        virtual_builder_module.RegionCursor([]),
        kedi_ast.TemplateLine(segments=[nested_call]),
        {},
        "    ",
    )
    virtual_statements_module._emit_segments(
        emit_builder,
        virtual_builder_module.RegionCursor([]),
        kedi_ast.TemplateLine(
            segments=[PythonExprSegment(code="missing_region", loc=Span(0, 1, 1, 1, 1, 5))]
        ),
        {},
        "    ",
    )

    raw_scope = virtual_scope_module._collect_toplevel_scope(
        [RawInvokeStmt(name="response", segments=[TextSegment(text="hello")])]
    )
    assert raw_scope["response"].typ == "str"
    assert (
        virtual_scope_module._index_header_ranges([{"ownerKind": None, "containerLine": None}])
        == {}
    )

    stmt_builder = virtual_builder_module.VirtualBuilder("[a] = ok\n`x = 1`\n", None)
    virtual_statements_module._emit_statements(
        stmt_builder,
        virtual_builder_module.RegionCursor(
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
            VariableInitStmt(name="a", expr=[TextSegment(text="ok")], loc=Span(0, 1, 1, 2, 1, 3)),
            kedi_ast.TemplateLine(
                segments=[OutputSegment(name="template_output", type_annotation="int")],
                loc=Span(0, 1, 1, 1, 1, 1),
            ),
            PythonBlock(code="x = 1", loc=Span(0, 1, 2, 1, 2, 7)),
            BacktickLineStmt(code="x = 1", loc=Span(0, 1, 2, 1, 2, 7)),
            BacktickLineStmt(code="y = 2", loc=Span(0, 1, 2, 1, 2, 7)),
        ],
        {},
        "    ",
        {},
    )
    assert "a: str = cast(str, _kedi_placeholder())" in stmt_builder.finish()
    assert "template_output: int = cast(int, _kedi_placeholder())" in stmt_builder.finish()
    assert "x = 1" in stmt_builder.finish()
    virtual_statements_module._emit_statements(
        stmt_builder,
        virtual_builder_module.RegionCursor([]),
        [
            BacktickLineStmt(code="z = 3", loc=Span(0, 1, 2, 1, 2, 7)),
            kedi_ast.ReturnBlockStmt(code="return 1", loc=Span(0, 1, 2, 1, 2, 8)),
            kedi_ast.Stmt(),
        ],
        {},
        "    ",
        {},
    )

    validation_builder = virtual_builder_module.VirtualBuilder(
        "`assert True`\n```\nDATA = []\n```\n", None
    )
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
                    "train": kedi_ast.EvalDataset(
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
    virtual_statements_module._emit_validation(
        validation_builder,
        virtual_builder_module.RegionCursor(
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
    metric_builder = virtual_builder_module.VirtualBuilder("", None)
    virtual_statements_module._emit_metric(
        metric_builder,
        virtual_builder_module.RegionCursor([]),
        EvalSuite(procedure_name="solve"),
        EvalMetric(name="score", code="return 1.0"),
        None,
    )
    assert "    pass" in metric_builder.finish()

    symbol_builder = virtual_builder_module.VirtualBuilder("", None)
    virtual_scope_module._add_scope_param_symbols(
        symbol_builder,
        {"x": virtual_builder_module.ScopeName("str", None)},
        0,
        "no params",
    )
    virtual_scope_module._add_scope_param_symbols(
        symbol_builder,
        {"missing": virtual_builder_module.ScopeName("str", None)},
        0,
        "fn(x)",
    )
    assert symbol_builder.symbols == []
    assert not virtual_builder_module._range_within_span(
        {"start": {"line": 0, "character": 0}, "end": {"line": 3, "character": 0}},
        Span(0, 1, 2, 1, 2, 5),
    )
    assert not virtual_builder_module._range_within_span(
        {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 99}},
        Span(0, 1, 2, 2, 2, 5),
    )
    assert virtual_scope_module._proc_name_range(ProcedureDef(name="p", params=[])) is None
    assert virtual_scope_module._type_name_range(TypeDef(name="T", fields=[])) is None
    assert virtual_scope_module._binding_name_range(VariableInitStmt(name="x")) is None
    assert virtual_scope_module._output_name_range(OutputSegment(name="x")) is None
    assert (
        virtual_scope_module._param_name_range(
            ["@p(x):"],
            ProcedureDef(name="p", params=["x"]),
            "x",
        )
        is None
    )
    assert (
        virtual_scope_module._param_name_range(
            [],
            ProcedureDef(name="p", params=["x"], loc=Span(0, 1, 1, 1, 1, 5)),
            "x",
        )
        is None
    )
    assert (
        virtual_scope_module._param_name_range(
            ["@p x):"],
            ProcedureDef(name="p", params=["x"], loc=Span(0, 1, 1, 1, 1, 5)),
            "x",
        )
        is None
    )
    assert (
        virtual_scope_module._param_name_range(
            ["@p(x"],
            ProcedureDef(name="p", params=["y"], loc=Span(0, 1, 1, 1, 1, 5)),
            "y",
        )
        is None
    )
    assert (
        virtual_scope_module._param_name_range(
            ["@p(x):"],
            ProcedureDef(name="p", params=["y"], loc=Span(0, 1, 1, 1, 1, 6)),
            "y",
        )
        is None
    )
    assert virtual_scope_module._container_line(ProcedureDef(name="p", params=[])) == -1
    virtual_scope_module._add_proc_header_regions(
        symbol_builder,
        ProcedureDef(
            name="p",
            params=["x"],
            param_defaults={"x": "1"},
            loc=Span(0, 1, 1, 1, 1, 6),
        ),
        signature_line=0,
        signature="def p(y: str = 1):",
        header_ranges={
            "param": {
                0: [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 1},
                        },
                        "expressionKind": "default",
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(
        virtual_scope_module,
        "_proc_param_to_python",
        lambda proc, name: "x: str",
    )
    virtual_scope_module._add_proc_header_regions(
        symbol_builder,
        ProcedureDef(
            name="p",
            params=["x"],
            param_defaults={"x": "1"},
            loc=Span(0, 1, 1, 1, 1, 6),
        ),
        signature_line=0,
        signature="def p(x: str):",
        header_ranges={
            "param": {
                0: [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 1},
                        },
                        "expressionKind": "default",
                    }
                ]
            }
        },
    )
    monkeypatch.undo()
    dynamic_header_builder = virtual_builder_module.VirtualBuilder("", None)
    virtual_scope_module._add_proc_header_regions(
        dynamic_header_builder,
        ProcedureDef(
            name="dynamic",
            params=["value"],
            param_type_annotations={"value": "int"},
            param_type_exprs={"value": TypePython(code="int")},
            return_type_annotation="list[int]",
            return_type_expr=TypePython(code="list[int]"),
            loc=Span(0, 1, 1, 1, 1, 30),
        ),
        signature_line=0,
        signature="def dynamic(value: int) -> list[int]:",
        header_ranges={
            "param": {
                0: [
                    {
                        "range": {
                            "start": {"line": 0, "character": 16},
                            "end": {"line": 0, "character": 19},
                        },
                        "expressionKind": "type",
                    }
                ]
            },
            "return_type": {
                0: [
                    {
                        "range": {
                            "start": {"line": 0, "character": 26},
                            "end": {"line": 0, "character": 35},
                        },
                        "expressionKind": "type",
                    }
                ]
            },
        },
    )
    assert len(dynamic_header_builder.mappings) == 2
    dynamic_proc = ProcedureDef(
        name="dynamic",
        params=["value"],
        param_type_annotations={"value": "int"},
        param_type_exprs={"value": TypePython(code="int")},
        return_type_annotation="list[int]",
        return_type_expr=TypePython(code="list[int]"),
        loc=Span(0, 1, 1, 1, 1, 30),
    )
    missing_header_builder = virtual_builder_module.VirtualBuilder("", None)
    virtual_scope_module._add_proc_header_regions(
        missing_header_builder,
        dynamic_proc,
        signature_line=0,
        signature="def dynamic(value: object):",
        header_ranges={
            "param": {
                0: [
                    {
                        "range": {
                            "start": {"line": 0, "character": 16},
                            "end": {"line": 0, "character": 19},
                        },
                        "expressionKind": "type",
                    }
                ]
            }
        },
    )
    virtual_scope_module._add_proc_header_regions(
        missing_header_builder,
        dynamic_proc,
        signature_line=0,
        signature="def dynamic(value: int) -> object:",
        header_ranges={
            "return_type": {
                0: [
                    {
                        "range": {
                            "start": {"line": 0, "character": 26},
                            "end": {"line": 0, "character": 35},
                        },
                        "expressionKind": "type",
                    }
                ]
            }
        },
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            virtual_scope_module,
            "_proc_param_to_python",
            lambda _proc, _name: "value: object",
        )
        virtual_scope_module._add_proc_header_regions(
            missing_header_builder,
            dynamic_proc,
            signature_line=0,
            signature="def dynamic(value: object):",
            header_ranges={
                "param": {
                    0: [
                        {
                            "range": {
                                "start": {"line": 0, "character": 16},
                                "end": {"line": 0, "character": 19},
                            },
                            "expressionKind": "type",
                        }
                    ]
                }
            },
        )
    virtual_scope_module._add_default_expr_region(
        symbol_builder,
        source_range={
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
        line=0,
        line_text="x: int = 1",
        expr="missing",
    )

    assert (
        virtual_scope_module._find_deepest_procedure_at_line([ProcedureDef(name="p", params=[])], 1)
        is None
    )
    first = ProcedureDef(name="first", params=[])
    second = ProcedureDef(name="second", params=[])
    assert virtual_scope_module._top_level_procedure_for([first, second], second) is second
    assert virtual_scope_module._top_level_procedure_for([first], second) is None
    assert virtual_scope_module._should_emit_full_module_python(None, 1) is False
    assert virtual_builder_module._indent_width("") == 0
    virtual_document_module._fallback_virtual_document(
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

    monkeypatch.setattr(virtual_document_module, "compute_python_ranges", lambda src: [])
    outside_focus = virtual_document_module.compute_python_virtual_document(
        "@p():\n  = ok\n\n```\nSIDE = 2\n```\n",
        focus_line=99,
    )
    assert "SIDE: Any = cast(Any, _kedi_placeholder())" in outside_focus["text"]
    fallback = virtual_document_module.compute_python_virtual_document(
        "```\nvalue = 1\n```\n@bad(:\n"
    )
    assert fallback["focusKey"] is None

    monkeypatch.setattr(virtual_document_module, "compute_python_ranges", lambda src: [])
    no_region = virtual_document_module.compute_python_virtual_document(
        "```\nPRELUDE = 1\n```\n```\nSIDE = 2\n```\n"
    )
    assert "PRELUDE = 1" in no_region["text"]
    assert "SIDE = 2" in no_region["text"]

    stdin = io.StringIO('{"text":"= ok\\n","uri":"file:///tmp/x.kedi","focusLine":0}')
    stdout = io.StringIO()
    monkeypatch.setattr(virtual_document_module.sys, "stdin", stdin)
    monkeypatch.setattr(virtual_document_module.sys, "stdout", stdout)
    virtual_document_module.main()
    assert json.loads(stdout.getvalue())["focusKey"] is None

    stdin = io.StringIO('\n{"id":1,"text":"= ok\\n"}\nnot-json\n')
    stdout = io.StringIO()
    monkeypatch.setattr(virtual_document_module.sys, "stdin", stdin)
    monkeypatch.setattr(virtual_document_module.sys, "stdout", stdout)
    virtual_document_module.main_loop()
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is False


# ---------------------------------------------------------------------
# Server construction smoke
# ---------------------------------------------------------------------
def test_hook_handler_expressions_are_mapped_into_virtual_python() -> None:
    source = """
```
def redact(event):
    return None
```
> hooks:
    user_prompt_submit: `redact`
    pre_tool_use: `[redact]`
"""

    virtual = compute_python_virtual_document(source, "/tmp/hooks.kedi")

    assert "def redact(event):" in virtual["text"]
    assert "        redact" in virtual["text"]
    assert "        [redact]" in virtual["text"]
    inline = [region for region in virtual["ranges"] if region["kind"] == "inline"]
    assert [region["text"] for region in inline] == ["redact", "[redact]"]


def test_virtual_document_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import io
    import json
    import runpy
    import sys

    entry = importlib.import_module("kedi.lsp.python_virtual.__main__")
    assert entry.main is virtual_document_module.main
    monkeypatch.delitem(sys.modules, entry.__name__)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"text":"= ok\\n"}'))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    runpy.run_module("kedi.lsp.python_virtual", run_name="__main__")
    assert json.loads(output.getvalue())["focusKey"] is None


def test_python_virtual_document_reads_bindings_that_templates_use():
    # Without the reads, Python tooling reports `count` and `who` as "not accessed".
    src = (
        "@greet(name: str) -> str:\n"
        "  = Hello <name>\n"
        "\n"
        "[count: int] = `3`\n"
        ">> There are <count> items.\n"
        "[who: str] = `'Kedi'`\n"
        "[literal: str] = `'unused'`\n"
        "= <greet(<who>)> <greet(literal)> <missing>\n"
    )
    text = compute_python_virtual_document(src, source_path=None)["text"]
    reads = [
        line.split(" = ", 1)[1]
        for line in text.splitlines()
        if line.strip().startswith("__kedi_template_read_")
    ]

    assert sorted(reads) == ["count", "name", "who"]
    ast.parse(text)
