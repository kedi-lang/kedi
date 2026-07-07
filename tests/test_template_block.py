from __future__ import annotations

from typing import Any

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import (
    AssignStmt,
    InputSegment,
    OutputSegment,
    Span,
    TemplateBlock,
    TemplateLine,
    TextSegment,
)
from kedi.lang.compiler.evaluation import merge_template_block_lines
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp import features
from kedi.lsp.features import compute_diagnostics, compute_hover, compute_semantic_tokens_full
from kedi.lsp.python_virtual import compute_python_virtual_document
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


def test_template_block_disambiguates_assign_stmt() -> None:
    src = "@p():\n  >> line one [x]\n  [y: int] = `1`\n  = <y>\n\n= <p()>\n"
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert isinstance(proc.body[0], TemplateBlock)
    assert len(proc.body[0].lines) == 1
    assign = proc.body[1]
    assert isinstance(assign, AssignStmt)
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
    keyword = features.SEMANTIC_TOKEN_TYPES.index("keyword")
    string_t = features.SEMANTIC_TOKEN_TYPES.index("string")
    variable = features.SEMANTIC_TOKEN_TYPES.index("variable")
    type_t = features.SEMANTIC_TOKEN_TYPES.index("type")
    function = features.SEMANTIC_TOKEN_TYPES.index("function")
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
