from __future__ import annotations

from typing import Any

import pytest

import kedi.lsp.features.scope as lsp_scope_module
import kedi.lsp.features.semantic_tokens as lsp_semantic_tokens_module
import kedi.lsp.features.symbols as lsp_symbols_module
import kedi.lsp.python_virtual.scope as virtual_scope_module
from kedi.agent_adapter.request.prompts import GROUNDING_INSTRUCTION
from kedi.artifacts import artifact_instructions
from kedi.core.expressions import o
from kedi.core.runtime import KediRuntime
from kedi.engine import ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import (
    InputSegment,
    OutputSegment,
    RawInvokeStmt,
    Span,
    TemplateBlock,
    TemplateLine,
    TextSegment,
    VariableInitStmt,
)
from kedi.lang.compiler.evaluation import merge_template_block_lines
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features.diagnostics import compute_diagnostics
from kedi.lsp.features.hover import compute_hover
from kedi.lsp.features.semantic_tokens import compute_semantic_tokens_full
from kedi.lsp.python_virtual.document import compute_python_virtual_document
from kedi.resolved_types import ResolvedKediType
from tests.lsp_helpers import decode_semantic_tokens, hover_at, hover_value, position_of
from tests.mock_adapter import MockAdapter


def _run(src: str) -> str:
    rt = compile_program(parse_program(src), adapter=MockAdapter())
    out = rt.run_main()
    assert isinstance(out, str)
    return out


def test_parse_template_block_multiline() -> None:
    src = ">> What's the [capital] of Turkey?\nWhat's the [population: int] of Ankara?\n= done\n"
    prog = parse_program(src)
    assert len(prog.toplevel_body) == 1
    block = prog.toplevel_body[0]
    assert isinstance(block, TemplateBlock)
    assert len(block.lines) == 2
    assert all(isinstance(line, TemplateLine) for line in block.lines)


def test_parse_template_block_preserves_blank_continuation_line() -> None:
    src = ">> First paragraph [answer].\n\nSecond paragraph adds context.\n= <answer>\n"
    prog = parse_program(src)
    block = prog.toplevel_body[0]
    assert isinstance(block, TemplateBlock)
    assert len(block.lines) == 3
    assert block.lines[1].segments == []

    merged = merge_template_block_lines(block.lines)
    text = "".join(segment.text for segment in merged.segments if isinstance(segment, TextSegment))
    assert "\n\nSecond paragraph" in text


def test_optimize_span_executes_its_lines_as_one_template_block() -> None:
    src = (
        "@summarize(topic: str) -> str:\n"
        "  > optimize: prompt:\n"
        "    >> Give a concise summary of <topic>.\n"
        "    Return it as [summary].\n"
        "  = <summary>\n"
        "\n"
        "= <summarize(Kedi)>\n"
    )
    adapter = MockAdapter()
    program = parse_program(src)
    proc = program.procedures[0]

    assert len(proc.optimize_spans) == 1
    assert len(proc.optimize_spans[0].template_lines) == 2
    assert isinstance(proc.body[0], TemplateBlock)
    assert proc.body[0].lines is proc.optimize_spans[0].template_lines

    runtime = compile_program(program, adapter=adapter)
    assert runtime.run_main() == "summary-value"
    assert len(adapter.instruction_calls) == 1


def test_optimize_span_accepts_explicit_template_block_syntax() -> None:
    src = (
        "@summarize(topic: str) -> str:\n"
        "  > optimize: prompt:\n"
        "    >> Give a concise summary of <topic>.\n"
        "    Return it as [summary].\n"
        "  = <summary>\n"
        "\n"
        "= <summarize(Kedi)>\n"
    )
    adapter = MockAdapter()
    program = parse_program(src)
    proc = program.procedures[0]

    assert len(proc.optimize_spans) == 1
    assert len(proc.optimize_spans[0].template_lines) == 2
    assert isinstance(proc.body[0], TemplateBlock)
    assert proc.body[0].lines is proc.optimize_spans[0].template_lines

    runtime = compile_program(program, adapter=adapter)
    assert runtime.run_main() == "summary-value"
    assert len(adapter.instruction_calls) == 1


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("padding", ["\n", "    \n", "\n\n", "\n  # Next statement\n"])
def test_optimize_block_allows_layout_whitespace_and_preserves_paragraphs(ending, padding):
    source = (
        "@summarize(topic: str) -> str:\n"
        "  > optimize: prompt:\n"
        "\n"
        "    # The prompt begins below.\n"
        "    >> Give a concise summary of <topic>.\n"
        "\n"
        "    The summary is [summary].\n" + padding + "  = <summary>\n"
        "\n"
        "= <summarize(Kedi)>\n"
    ).replace("\n", ending)
    program = parse_program(source)
    lines = program.procedures[0].optimize_spans[0].template_lines
    assert len(lines) == 3
    assert lines[1].segments == []
    adapter = MockAdapter()
    assert compile_program(program, adapter=adapter).run_main() == "summary-value"
    assert len(adapter.instruction_calls) == 1
    assert "summary of Kedi.\n\nThe summary is [summary]." in adapter.request_calls[0]


@pytest.mark.parametrize(
    "body",
    [
        "    The result is [answer].\n",
        "\n    The result is [answer].\n",
        "    # A comment is not a template marker.\n    The result is [answer].\n",
        "    Read <topic>.\n    The result is [answer].\n",
        "    The result is [answer].\n    >> A later marker cannot repair the first line.\n",
    ],
)
def test_optimize_rejects_bare_prompt_in_grammar_and_runtime_parser(body):
    import tree_sitter_kedi
    from tree_sitter import Language, Parser

    source = "@summarize(topic):\n  > optimize: prompt:\n" + body + "  = <answer>\n"
    tree = Parser(Language(tree_sitter_kedi.language())).parse(source.encode())
    assert tree.root_node.has_error
    with pytest.raises(KediParseError):
        parse_program(source)


def test_optimize_rejects_multiple_template_blocks_in_one_span():
    with pytest.raises(KediParseError):
        parse_program(
            "@summarize():\n"
            "  > optimize: prompt:\n"
            "    >> The first result is [first].\n"
            "    >> The second result is [second].\n"
            "  = <second>\n"
        )


def test_optimize_span_cannot_read_its_own_pending_output() -> None:
    src = (
        "@plan() -> str:\n"
        "  > optimize: prompt:\n"
        "    >> Pick a European [city].\n"
        "    Describe <city> as [description].\n"
        "  = <description>\n"
    )

    with pytest.raises(KediParseError) as exc:
        parse_program(src)

    assert any(d.code == "K-TEMPLATE-BLOCK-FORWARD-OUTPUT" for d in exc.value.diagnostics)


def test_merge_template_block_lines_joins_with_newline() -> None:
    block = parse_program(">> line one [x]\nline two <seed>\n").toplevel_body[0]
    assert isinstance(block, TemplateBlock)
    merged = merge_template_block_lines(block.lines)
    assert any(isinstance(seg, TextSegment) and seg.text == "\n" for seg in merged.segments)


def test_merge_template_block_lines_edge_cases() -> None:
    assert merge_template_block_lines([]).segments == []

    single = TemplateLine(segments=[TextSegment(text="only")])
    assert merge_template_block_lines([single]) is single

    block_loc = Span(
        start_byte=0,
        end_byte=10,
        start_line=1,
        start_col=1,
        end_line=2,
        end_col=1,
    )
    relocated = merge_template_block_lines([single], loc=block_loc)
    assert relocated.loc == block_loc
    assert relocated.segments is single.segments

    merged = merge_template_block_lines(
        [
            TemplateLine(
                segments=[OutputSegment(name="a")],
                optimized_output_override={"a": 1},
            ),
            TemplateLine(
                segments=[InputSegment(name="a")],
                optimized_output_override={"b": 2},
            ),
        ]
    )
    assert merged.optimized_output_override == {"a": 1, "b": 2}
    assert any(isinstance(seg, InputSegment) and seg.name == "a" for seg in merged.segments)


def test_template_block_cannot_substitute_output_from_same_block() -> None:
    src = (
        ">> [city] is a city in Europe.\n"
        "Based on delivery result. Delivery estimate for <city> is [estimate: int] days.\n"
    )
    with pytest.raises(KediParseError) as exc:
        parse_program(src)
    assert any(d.code == "K-TEMPLATE-BLOCK-FORWARD-OUTPUT" for d in exc.value.diagnostics)
    diagnostics = compute_diagnostics(src, None)
    assert any(d.code == "K-TEMPLATE-BLOCK-FORWARD-OUTPUT" for d in diagnostics)


def test_raw_invoke_parses_capture_statement() -> None:
    prog = parse_program("[answer] << Hello <name>\n")
    stmt = prog.toplevel_body[0]
    assert isinstance(stmt, RawInvokeStmt)
    assert stmt.name == "answer"
    assert any(isinstance(seg, InputSegment) and seg.name == "name" for seg in stmt.segments)


def test_raw_invoke_capture_runs_adapter_invoke() -> None:
    src = "[name] = Ada\n[answer] << Hello <name>\n= <answer>\n"
    adapter = MockAdapter()
    rt = compile_program(parse_program(src), adapter=adapter)

    assert rt.run_main() == "response:Hello Ada"
    assert len(adapter.instruction_calls) == 1


def test_raw_invoke_supports_adapter_without_profile_override() -> None:
    class InvokeOnlyAdapter:
        def invoke_sync(self, *, prompt: str, **_kwargs: Any) -> str:
            assert prompt == "Hello"
            return "ok"

    rt = compile_program(
        parse_program("[answer] << Hello\n= <answer>\n"),
        adapter=InvokeOnlyAdapter(),
    )

    assert rt.run_main() == "ok"


def test_raw_invoke_capture_runs_inside_procedure_body() -> None:
    src = "@p(name):\n  [answer] << Hello <name>\n  = <answer>\n\n= <p(Ada)>\n"
    adapter = MockAdapter()
    rt = compile_program(parse_program(src), adapter=adapter)

    assert rt.run_main() == "response:Hello Ada"
    assert len(adapter.instruction_calls) == 1


def test_raw_invoke_direct_runtime_guards(monkeypatch: Any) -> None:
    rt = KediRuntime(adapter=MockAdapter())
    with pytest.raises(RuntimeError, match="outside of a kedi procedure"):
        rt.invoke(["prompt"])

    token = rt._current_env.set({})
    try:
        assert rt.invoke(["prompt"]) is None
        with pytest.raises(ValueError, match="output fields"):
            rt.invoke([o("answer")])

        def fake_evaluate_expression(_expr: object, _env: dict[str, Any]) -> tuple[None, None]:
            return None, None

        monkeypatch.setattr(rt, "_evaluate_expression", fake_evaluate_expression)
        assert rt.invoke(["ignored"]) is None
    finally:
        rt._current_env.reset(token)


def test_raw_invoke_uses_profile_tools_system_and_parallel_engine() -> None:
    src = (
        "> model: raw-model\n"
        "> instructions:\n"
        "  Keep it short.\n"
        "\n"
        "@lookup() -> str:\n"
        "  = ok\n"
        "\n"
        "> use: lookup\n"
        "[answer] << Hello\n"
        "= <answer>\n"
    )
    adapter = MockAdapter()
    rt = compile_program(parse_program(src), adapter=adapter)

    assert rt.run_main() == "response:Hello"
    assert adapter.model_calls == ["raw-model"]
    assert adapter.instruction_calls == [
        f"Keep it short.\n\n<ToolCallInstructions>\n{GROUNDING_INSTRUCTION}\n</ToolCallInstructions>\n\n{artifact_instructions()}"
    ]
    assert adapter.tool_calls == [
        (
            "lookup",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]

    parallel_adapter = MockAdapter()
    parallel_rt = compile_program(
        parse_program(">> Give me [name]\n[answer] << Hello <name>\n= <answer>\n"),
        adapter=parallel_adapter,
        engine=ThreadEngine(max_workers=1),
    )
    try:
        assert parallel_rt.run_main() == "response:Hello name-value"
    finally:
        parallel_rt._engine.shutdown()

    discard_adapter = MockAdapter()
    discard_rt = compile_program(
        parse_program(">> Hello\n= done\n"),
        adapter=discard_adapter,
        engine=ThreadEngine(max_workers=1),
    )
    try:
        assert discard_rt.run_main() == "done"
    finally:
        discard_rt._engine.shutdown()


def test_parallel_raw_invoke_preserves_kedi_execution_errors() -> None:
    class FailingAdapter(MockAdapter):
        def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
            raise KediExecutionError("adapter failed")

    rt = compile_program(
        parse_program("[answer] << Hello\n= <answer>\n"),
        adapter=FailingAdapter(),
        engine=ThreadEngine(max_workers=1),
    )
    try:
        with pytest.raises(KediExecutionError, match="adapter failed"):
            rt.run_main()
    finally:
        rt._engine.shutdown()


def test_parallel_raw_invoke_wraps_unexpected_adapter_errors() -> None:
    class FailingAdapter(MockAdapter):
        def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
            raise ValueError("bad invoke")

    rt = compile_program(
        parse_program("[answer] << Hello\n= <answer>\n"),
        adapter=FailingAdapter(),
        engine=ThreadEngine(max_workers=1),
    )
    try:
        with pytest.raises(KediExecutionError) as exc:
            rt.run_main()
        assert "bad invoke" in str(exc.value)
        assert isinstance(exc.value.original, ValueError)
    finally:
        rt._engine.shutdown()


def test_outputless_template_block_runs_adapter_invoke_and_discards_response() -> None:
    src = ">> Hello raw\n= done\n"
    adapter = MockAdapter()
    rt = compile_program(parse_program(src), adapter=adapter)

    assert rt.run_main() == "done"
    assert len(adapter.instruction_calls) == 1


def test_raw_invoke_is_visible_in_virtual_python() -> None:
    src = "@p():\n  [answer] << Hello\n  ```\n  print(answer)\n  ```\n  = <answer>\n"
    doc = compute_python_virtual_document(src, "/tmp/raw.kedi")

    assert "answer: str = cast(str, _kedi_placeholder())" in doc["text"]
    assert any(symbol["name"] == "answer" for symbol in doc["symbols"])
    stmt = parse_program("[answer] << Hello <name>\n").toplevel_body[0]
    assert isinstance(stmt, RawInvokeStmt)
    assert lsp_scope_module._segments_of(stmt)
    assert virtual_scope_module._stmt_segments(stmt)
    assert virtual_scope_module._raw_invoke_name_range(RawInvokeStmt(name="x")) is None


def test_raw_invoke_rejects_output_fields_and_non_string_capture_type() -> None:
    with pytest.raises(KediParseError) as typed:
        parse_program("[answer: int] << Hello\n")
    assert any(d.code == "K-RAW-INVOKE-TYPE" for d in typed.value.diagnostics)

    with pytest.raises(KediParseError) as output:
        parse_program("[answer] << Hello [field]\n")
    assert any(d.code == "K-RAW-INVOKE-OUTPUT" for d in output.value.diagnostics)


def test_raw_invoke_str_capture_type_is_lsp_warning() -> None:
    diags = compute_diagnostics("[answer: str] << Hello\n", "example.kedi")

    assert any(
        d.code == "K-RAW-INVOKE-REDUNDANT-STR"
        and d.severity == lsp_symbols_module.lsp.DiagnosticSeverity.Warning
        for d in diags
    )


def test_template_block_single_agent_run() -> None:
    calls: list[str] = []

    class CountingAdapter(MockAdapter):
        def produce_sync(
            self,
            *,
            template: str,
            output_schema: dict[str, ResolvedKediType] | None = None,
            **_: Any,
        ) -> Any:
            calls.append(template)
            return super().produce_sync(template=template, output_schema=output_schema)

    src = "@p():\n  >> Foo bar [baz]\n  Baz foo [bar]\n  Bar baz [foo]\n  = ok\n\n= <p()>\n"
    rt = compile_program(parse_program(src), adapter=CountingAdapter())
    rt.run_main()
    assert len(calls) == 1
    assert "Foo bar [baz]" in calls[0]
    assert "Baz foo [bar]" in calls[0]
    assert "Bar baz [foo]" in calls[0]


@pytest.mark.parametrize(
    "declarations",
    (
        "[value] and [value]",
        "[value: str] and [value]",
        "[value] and [value: str]",
        "[value: str] and [value: int]",
        "[value]\nrepeat [value]",
    ),
)
def test_template_block_rejects_duplicate_output_fields(declarations: str) -> None:
    source = f">> {declarations}\n= ok\n"

    with pytest.raises(KediParseError) as exc:
        parse_program(source)

    assert any(
        diagnostic.code == "K-TEMPLATE-DUPLICATE-OUTPUT" for diagnostic in exc.value.diagnostics
    )
    diagnostics = compute_diagnostics(source, "/tmp/duplicate-output.kedi")
    assert any(diagnostic.code == "K-TEMPLATE-DUPLICATE-OUTPUT" for diagnostic in diagnostics)


def test_separate_template_blocks_may_reuse_an_output_name() -> None:
    parse_program(">> First [value].\n>> Second [value].\n= ok\n")


def test_optimize_template_rejects_duplicate_output_fields() -> None:
    source = "@p():\n  > optimize: prompt:\n    >> Put [value] and [value: str].\n  = ok\n= <p()>\n"

    with pytest.raises(KediParseError) as exc:
        parse_program(source)

    assert any(
        diagnostic.code == "K-TEMPLATE-DUPLICATE-OUTPUT" for diagnostic in exc.value.diagnostics
    )


def test_template_block_stateful_run(monkeypatch: Any) -> None:
    src = (
        "@p():\n"
        "  >> Foo bar [baz]\n"
        "  >> Baz foo [bar]\n"
        "  >> Bar baz [foo]\n"
        "  = <foo>-<bar>-<baz>\n\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out == "foo-value-bar-value-baz-value"


def test_bare_template_line_at_top_level_is_syntax_error() -> None:
    src = "The [the]\n= ok\n"
    with pytest.raises(KediParseError) as exc:
        parse_program(src)
    assert any(d.code == "K-BARE-TEMPLATE" for d in exc.value.diagnostics)
    diags = compute_diagnostics(src, None)
    assert any(d.code == "K-BARE-TEMPLATE" for d in diags)


def test_bare_template_line_in_procedure_body_is_syntax_error() -> None:
    src = "@p():\n  The [the]\n  = ok\n"
    with pytest.raises(KediParseError) as exc:
        parse_program(src)
    assert any(d.code == "K-BARE-TEMPLATE" for d in exc.value.diagnostics)


def test_bare_template_line_inside_block_is_continuation() -> None:
    src = ">> First [a]\nThe [the]\n= <the>\n"
    prog = parse_program(src)
    block = prog.toplevel_body[0]
    assert isinstance(block, TemplateBlock)
    assert len(block.lines) == 2
    assert any(
        isinstance(seg, TextSegment) and seg.text == "The " for seg in block.lines[1].segments
    )
    out = _run(src)
    assert out == "the-value"


def test_template_block_hover_and_diagnostics() -> None:
    src = ">> What's the [capital] of Turkey?\nWhat's the [population: int] of Ankara?\n"
    hover = compute_hover(src, None, position_of(src, ">>"))
    assert hover is not None
    assert "Template block" in hover_value(hover)

    capital_hover = hover_at(src, "/tmp/example.kedi", "capital", occurrence=1)
    assert "Output" in hover_value(capital_hover)


def test_template_block_disambiguates_variable_init_stmt() -> None:
    src = "@p():\n  >> line one [x]\n  [y: int] = `1`\n  = <y>\n\n= <p()>\n"
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert isinstance(proc.body[0], TemplateBlock)
    assert len(proc.body[0].lines) == 1
    assign = proc.body[1]
    assert isinstance(assign, VariableInitStmt)
    assert assign.name == "y"


def test_top_level_template_block_executes() -> None:
    src = ">> line one [x]\nline two is static\n= <x>\n"
    out = _run(src)
    assert out == "x-value"


def test_template_block_lsp_semantic_tokens_and_virtual_python() -> None:
    src = "@p(seed):\n  >> first [a]\n  second <seed>\n  = <a>\n"
    tokens = compute_semantic_tokens_full(src, "/tmp/tb.kedi")
    assert tokens.data
    virtual = compute_python_virtual_document(src, "/tmp/tb.kedi", focus_line=1)
    assert "first" in virtual["text"] or "a" in virtual["text"]


def test_top_level_template_block_semantic_tokens_match_screenshot_shape() -> None:
    """Regression: stale parser turns the whole file into ERROR → zero tokens (all gray)."""
    src = ">> Hello [id: int]\n\n\n@proc(id: int):\n    = done\n"
    tokens = compute_semantic_tokens_full(src, "/tmp/tb.kedi")
    assert tokens.data, "template_block_stmt grammar must yield highlight captures"
    decoded = decode_semantic_tokens(tokens.data)
    keyword = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("keyword")
    string_t = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("string")
    variable = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("variable")
    type_t = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("type")
    function = lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("function")
    assert any(ln == 0 and token_type == keyword for ln, _, _, token_type, _ in decoded)
    assert any(ln == 0 and token_type == string_t for ln, _, _, token_type, _ in decoded)
    assert any(ln == 0 and token_type == variable for ln, _, _, token_type, _ in decoded)
    assert any(ln == 0 and token_type == type_t for ln, _, _, token_type, _ in decoded)
    assert any(ln == 3 and token_type == function for ln, _, _, token_type, _ in decoded)


def test_multiple_template_blocks_in_procedure() -> None:
    src = "@p(seed):\n  >> first [a]\n  second <seed>\n\n  >> other [b]\n  = <b>\n\n= <p(x)>\n"
    prog = parse_program(src)
    blocks = [s for s in prog.procedures[0].body if isinstance(s, TemplateBlock)]
    assert len(blocks) == 2
    assert len(blocks[0].lines) == 2
    assert len(blocks[1].lines) == 1
