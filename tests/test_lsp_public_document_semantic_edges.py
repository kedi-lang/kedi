"""LSP document symbols, semantic tokens, and python range edge cases."""

from __future__ import annotations

import pytest

import kedi.lsp.features.document as lsp_document_module
import kedi.lsp.features.parser_state as lsp_parser_state_module
import kedi.lsp.features.python_regions as python_regions_module
import kedi.lsp.features.semantic_tokens as lsp_semantic_tokens_module
from kedi.lang.ast import EvalDataset, EvalMetric, EvalSuite


def test_document_symbols_include_eval_suite() -> None:
    eval_symbols = lsp_document_module.compute_document_symbols(
        "@eval: p:\n"
        "  > data: train:\n"
        "    = `[]`\n"
        "  > test_data: holdout:\n"
        "    = `[]`\n"
        "  > metric: score(train):\n"
        "    = `1.0`\n",
        "/tmp/x.kedi",
    )
    assert any(symbol.name == "@eval: p" for symbol in eval_symbols)
    direct_eval = lsp_document_module._eval_suite_symbol(
        EvalSuite(
            "p",
            datasets={"train": EvalDataset("train", "[]")},
            test_datasets={"holdout": EvalDataset("holdout", "[]")},
            metrics=[EvalMetric("score", "return 1")],
        )
    )
    assert direct_eval.range.start.line == 0


def test_semantic_token_capture_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(lsp_semantic_tokens_module._CAPTURE_TO_LSP, "custom.deep", ("variable", 0))
    assert lsp_semantic_tokens_module._capture_to_token("custom.deep.extra") == (
        lsp_semantic_tokens_module.SEMANTIC_TOKEN_TYPES.index("variable"),
        0,
    )
    assert lsp_semantic_tokens_module._capture_to_token("missing.capture") is None

    original_py_parser = lsp_parser_state_module._PY_PARSER
    original_py_query = lsp_parser_state_module._PY_HIGHLIGHTS_QUERY
    lsp_parser_state_module._PY_PARSER = original_py_parser
    lsp_parser_state_module._PY_HIGHLIGHTS_QUERY = original_py_query
    py_tokens = lsp_semantic_tokens_module.compute_semantic_tokens_full(
        "= <`len([1, 2])`>\n", "/tmp/x.kedi"
    )
    assert len(py_tokens.data) % 5 == 0
    monkeypatch.setattr(lsp_semantic_tokens_module, "_KEDI_HIGHLIGHTS_QUERY", None)
    monkeypatch.setattr(lsp_semantic_tokens_module, "_KEDI_INJECTIONS_QUERY", None)
    assert (
        lsp_semantic_tokens_module.compute_semantic_tokens_full(
            "@p():\n  = ok\n", "/tmp/x.kedi"
        ).data
        == []
    )


def test_python_range_helpers() -> None:
    all_ranges = python_regions_module.compute_python_ranges("@p(x: int = `1`):\n  = <`x`>\n")
    header_ranges = python_regions_module.compute_header_python_ranges(
        "@p(x: int = `1`):\n  = <`x`>\n"
    )
    assert any(r.get("ownerKind") == "param" for r in header_ranges)
    assert all(r["text"] == "1" for r in header_ranges)
    assert any(r["text"] == "x" for r in all_ranges)
    regular_only = python_regions_module._collect_python_ranges(
        "@p(x: int = `1`):\n  = <`x`>\n",
        include_header_inline=False,
    )
    assert [item["text"] for item in regular_only] == ["x"]

    return_ranges = python_regions_module.compute_header_python_ranges(
        "@p() -> `list[int]`:\n  = ok\n"
    )
    assert return_ranges == [
        {
            "kind": "inline",
            "range": {
                "start": {"line": 0, "character": 9},
                "end": {"line": 0, "character": 18},
            },
            "text": "list[int]",
            "ownerKind": "return_type",
            "containerKind": "procedure_def",
            "containerLine": 0,
            "expressionKind": "type",
        }
    ]
