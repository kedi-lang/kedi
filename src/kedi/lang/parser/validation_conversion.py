from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from tree_sitter import Node

from kedi.lang.ast import (
    EvalDataset,
    EvalMetric,
    EvalSuite,
    OptimizeSpan,
    TemplateBlock,
    TemplateLine,
    TestCase,
    TestSuite,
)
from kedi.lang.parser.cst_helpers import _field, _identifier_name, _named_children, _src_text

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker


def _walk_optimize_directive(self: _Walker, node: Node) -> tuple[OptimizeSpan, TemplateBlock]:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    body_node = _field(node, "body")
    template_lines: List[TemplateLine] = []
    if body_node is not None:
        body_children = _named_children(body_node)
        block_node = next(
            (child for child in body_children if child.type == "template_block_stmt"),
            None,
        )
        if block_node is not None:
            template_lines = self._walk_template_block_stmt(block_node).lines
            self._validate_template_block_scope(template_lines)
        else:
            for tline in body_children:
                if tline.type != "template_line":
                    continue
                template_lines.append(self._walk_template_line(tline))
            self._validate_template_block_scope(template_lines)
    if not template_lines:
        # Legacy invariant — preserve the exact substring so
        # ``pytest.raises(ValueError, match="must have an indented block")``
        # keeps matching.
        self._error(
            node,
            f"Optimize span {name!r} must have an indented block after it",
            code="K-OPT-EMPTY",
        )
    # The optimize span's template lines also execute during a normal run.
    # Keep them in one TemplateBlock so the whole span is one LLM call,
    # matching ordinary `>>` continuation semantics. The optimizer still
    # owns the individual line objects and mutates them in place.
    loc = self._span(node)
    return (
        OptimizeSpan(name=name, template_lines=template_lines, loc=loc),
        TemplateBlock(lines=template_lines, loc=loc),
    )


def _walk_validation_block(self: _Walker, node: Node) -> Union[TestSuite, EvalSuite, None]:
    """Walk a `@test:` or `@eval:` block.

    The grammar accepts ``@<kw>:<procedure>:`` with a uniform body
    rule that allows ``> case:`` (test) or ``> data:`` / ``>
    test_data:`` / ``> metric:`` (eval) directives. We dispatch
    here based on the kw text, and emit a structured "Unknown
    directive" error for any body entry that does not match the
    chosen suite type.
    """
    kw_node = _field(node, "kw")
    if kw_node is None:
        self._error(node, "validation block missing keyword")
        return None
    kw_text = _src_text(self.source, kw_node)
    proc_node = _field(node, "procedure")
    if proc_node is None:
        self._error(node, "validation block missing procedure name")
        return None
    procedure_name = _identifier_name(self.source, proc_node)
    body = _field(node, "body")
    body_children = _named_children(body) if body is not None else []

    if kw_text == "test":
        suite_t = TestSuite(procedure_name=procedure_name, cases=[], loc=self._span(node))
        for c in body_children:
            if c.type == "test_case":
                suite_t.cases.append(self._walk_test_case(c))
            elif c.type in ("eval_data", "eval_test_data", "eval_metric"):
                self._error(
                    c,
                    f"Unknown directive in @test body: {c.type!r}",
                    code="K-VALID-DIR",
                )
            elif c.type == "ERROR" or c.is_error:
                self._error(
                    c,
                    "Unknown directive in @test body",
                    code="K-VALID-DIR",
                )
        return suite_t

    if kw_text == "eval":
        suite_e = EvalSuite(procedure_name=procedure_name, loc=self._span(node))
        metric_count = 0
        for c in body_children:
            if c.type == "test_case":
                self._error(
                    c,
                    "Unknown directive in @eval body: 'case' is only valid in @test",
                    code="K-VALID-DIR",
                )
            elif c.type == "eval_data":
                ds = self._walk_eval_dataset(c, is_test=False)
                suite_e.datasets[ds.name] = ds
            elif c.type == "eval_test_data":
                ds = self._walk_eval_dataset(c, is_test=True)
                suite_e.test_datasets[ds.name] = ds
            elif c.type == "eval_metric":
                metric_count += 1
                suite_e.metrics.append(self._walk_eval_metric(c))
            elif c.type == "ERROR" or c.is_error:
                self._error(
                    c,
                    "Unknown directive in @eval body",
                    code="K-VALID-DIR",
                )
        if metric_count > 1:
            self._error(
                node,
                "Multiple metrics defined for the same procedure",
                code="K-EVAL-MULTI-METRIC",
            )
        return suite_e

    self._error(
        kw_node,
        f"Unknown validation block keyword {kw_text!r}; expected 'test' or 'eval'",
        code="K-VALID-KW",
    )
    return None


def _walk_test_case(self: _Walker, node: Node) -> TestCase:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    body = _field(node, "code")
    code = ""
    is_block = False
    if body is not None:
        for inner in _named_children(body):
            if inner.type == "backtick_line_stmt":
                # Single-line backtick: extract the inline python
                # body (without backticks).
                bn = next(
                    (c for c in _named_children(inner) if c.type == "inline_python_expr"),
                    None,
                )
                if bn is not None:
                    code = self._walk_inline_python_expr(bn)
                is_block = False
            elif inner.type == "python_block":
                code = self._walk_python_block_code(inner)
                is_block = True
    return TestCase(name=name, code=code, is_block=is_block, loc=self._span(node))


def _walk_eval_dataset(self: _Walker, node: Node, *, is_test: bool) -> EvalDataset:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    code, is_block = self._walk_eval_entry_body(_field(node, "body"))
    return EvalDataset(name=name, code=code, is_block=is_block, loc=self._span(node))


def _walk_eval_metric(self: _Walker, node: Node) -> EvalMetric:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    dataset_node = _field(node, "dataset")
    dataset_name = _identifier_name(self.source, dataset_node) if dataset_node is not None else None
    code, is_block = self._walk_eval_entry_body(_field(node, "body"))
    return EvalMetric(
        name=name,
        code=code,
        is_block=is_block,
        dataset_name=dataset_name,
        loc=self._span(node),
    )


def _walk_eval_entry_body(self: _Walker, body: Optional[Node]) -> tuple[str, bool]:
    if body is None:
        return ("", False)
    for inner in _named_children(body):
        if inner.type == "return_stmt":
            # The return's value is either inline_python_expr or
            # template_expr — eval entries always use the former
            # in practice.
            value = _field(inner, "value")
            if value is not None and value.type == "inline_python_expr":
                return (self._walk_inline_python_expr(value), False)
            if value is not None:
                return (_src_text(self.source, value), False)
            return ("", False)
        if inner.type == "return_block_stmt":
            code_node = _field(inner, "code")
            code = self._dedent_fenced(code_node) if code_node is not None else ""
            return (code, True)
    return ("", False)
