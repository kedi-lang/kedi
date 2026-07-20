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


def test_python_virtual_document_wraps_template_python_substitutions_in_assignments():
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
    from kedi.lsp import python_virtual as pv

    program = parse_program('~Request(note: Annotated[str, "Human note"])\n')

    assert pv._typing_import_lines(program, None) == ["from typing import Annotated, Any"]
    assert (
        pv._typing_import_lines(
            program,
            None,
            ("from typing import *", "this is not valid Python:"),
        )
        == []
    )
    assert pv._typing_symbols_in_python_stub("not valid Python:") == set()
    assert ("typing", "Optional") in pv._typing_symbols_in_python_stub(
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
        "= ok\n\n"
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
        AssignStmt,
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

    nested_builder = pv.VirtualBuilder(
        "@outer():\n  @inner():\n    ###\n    >> doc\n    ###\n", None
    )
    inner = ProcedureDef(
        name="inner",
        params=[],
        body=[],
        docstring="doc",
        loc=Span(0, 1, 2, 3, 5, 8),
    )
    pv._emit_nested_procedure(nested_builder, pv.RegionCursor([]), inner, {}, "    ", {})
    empty_inner = ProcedureDef(name="empty", params=[], body=[], loc=Span(0, 1, 1, 1, 1, 10))
    pv._emit_nested_procedure(nested_builder, pv.RegionCursor([]), empty_inner, {}, "    ", {})
    assert '"""doc"""' in nested_builder.finish()
    assert "        return cast(str, _kedi_placeholder())" in nested_builder.finish()
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
    assert "y: str = cast(str, _kedi_placeholder())" in emit_builder.finish()
    region_for_assign = {
        "kind": "inline",
        "range": {
            "start": {"line": 0, "character": 11},
            "end": {"line": 0, "character": 12},
        },
        "text": "1",
    }
    pv._emit_assign_stmt(emit_builder, pv.RegionCursor([region_for_assign]), assign, {}, "    ")

    import_builder = pv.VirtualBuilder("", None)
    assert pv._emit_python_stub_stmt(import_builder, ast.parse("import typing").body[0])
    assert (
        pv._emit_python_stub_stmt(
            import_builder,
            ast.parse("from package import *").body[0],
        )
        is False
    )
    assert "import typing" in import_builder.finish()
    assert pv._checker_safe_value_stub("not valid Python !") == "not valid Python !"

    no_region_block = pv.AssignBlockStmt(
        name="blocked",
        code="return 1",
        type_annotation="int",
        loc=Span(0, 1, 1, 1, 1, 10),
    )
    pv._emit_assign_block(
        emit_builder,
        pv.RegionCursor([]),
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
    assert len(list(pv._call_inline_python_locations(nested_call))) == 5
    pv._emit_segments(
        emit_builder,
        pv.RegionCursor([]),
        pv.TemplateLine(segments=[nested_call]),
        {},
        "    ",
    )
    pv._emit_segments(
        emit_builder,
        pv.RegionCursor([]),
        pv.TemplateLine(
            segments=[PythonExprSegment(code="missing_region", loc=Span(0, 1, 1, 1, 1, 5))]
        ),
        {},
        "    ",
    )

    raw_scope = pv._collect_toplevel_scope(
        [RawInvokeStmt(name="response", segments=[TextSegment(text="hello")])]
    )
    assert raw_scope["response"].typ == "str"
    assert pv._index_header_ranges([{"ownerKind": None, "containerLine": None}]) == {}

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
            pv.TemplateLine(
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
    pv._add_proc_header_regions(
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
    monkeypatch.setattr(pv, "_proc_param_to_python", lambda proc, name: "x: str")
    pv._add_proc_header_regions(
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
    dynamic_header_builder = pv.VirtualBuilder("", None)
    pv._add_proc_header_regions(
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
    missing_header_builder = pv.VirtualBuilder("", None)
    pv._add_proc_header_regions(
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
    pv._add_proc_header_regions(
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
        patch.setattr(pv, "_proc_param_to_python", lambda _proc, _name: "value: object")
        pv._add_proc_header_regions(
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
    first = ProcedureDef(name="first", params=[])
    second = ProcedureDef(name="second", params=[])
    assert pv._top_level_procedure_for([first, second], second) is second
    assert pv._top_level_procedure_for([first], second) is None
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
    assert "SIDE: Any = cast(Any, _kedi_placeholder())" in outside_focus["text"]
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
