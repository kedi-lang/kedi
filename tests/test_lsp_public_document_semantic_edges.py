"""LSP document symbols, semantic tokens, and python range edge cases."""

from __future__ import annotations

import pytest

from kedi.lang.ast import EvalDataset, EvalMetric, EvalSuite
from kedi.lsp import features


def test_document_symbols_include_eval_suite() -> None:
    eval_symbols = features.compute_document_symbols(
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
    direct_eval = features._eval_suite_symbol(
        EvalSuite(
            "p",
            datasets={"train": EvalDataset("train", "[]")},
            test_datasets={"holdout": EvalDataset("holdout", "[]")},
            metrics=[EvalMetric("score", "return 1")],
        )
    )
    assert direct_eval.range.start.line == 0


def test_semantic_token_capture_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(features._CAPTURE_TO_LSP, "custom.deep", ("variable", 0))
    assert features._capture_to_token("custom.deep.extra") == (
        features.SEMANTIC_TOKEN_TYPES.index("variable"),
        0,
    )
    assert features._capture_to_token("missing.capture") is None

    original_py_parser = features._PY_PARSER
    original_py_query = features._PY_HIGHLIGHTS_QUERY
    features._PY_PARSER = original_py_parser
    features._PY_HIGHLIGHTS_QUERY = original_py_query
    py_tokens = features.compute_semantic_tokens_full("= <`len([1, 2])`>\n", "/tmp/x.kedi")
    assert len(py_tokens.data) % 5 == 0
    monkeypatch.setattr(features, "_KEDI_HIGHLIGHTS_QUERY", None)
    monkeypatch.setattr(features, "_KEDI_INJECTIONS_QUERY", None)
    assert features.compute_semantic_tokens_full("@p():\n  = ok\n", "/tmp/x.kedi").data == []


def test_python_range_helpers() -> None:
    all_ranges = features.compute_python_ranges("@p(x: int = `1`):\n  = <`x`>\n")
    header_ranges = features.compute_header_python_ranges("@p(x: int = `1`):\n  = <`x`>\n")
    assert any(r.get("ownerKind") == "param" for r in header_ranges)
    assert all(r["text"] == "1" for r in header_ranges)
    assert any(r["text"] == "x" for r in all_ranges)
    regular_only = features._collect_python_ranges(
        "@p(x: int = `1`):\n  = <`x`>\n",
        include_header_inline=False,
    )
    assert [item["text"] for item in regular_only] == ["x"]

    return_ranges = features.compute_header_python_ranges("@p() -> `list[int]`:\n  = ok\n")
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
