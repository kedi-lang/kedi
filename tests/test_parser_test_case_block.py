"""Focused parser tests for @test case body shapes."""

from __future__ import annotations

import importlib
from typing import Any, cast

cst_module = importlib.import_module("kedi.lang.parser.cst_to_ast")


class _FakeNode:
    def __init__(
        self,
        type_: str,
        *,
        children: list["_FakeNode"] | None = None,
        fields: dict[str, "_FakeNode"] | None = None,
        start: int = 0,
        end: int = 4,
    ) -> None:
        self.type = type_
        self.children = children or []
        self._fields = fields or {}
        self.is_named = True
        self.start_byte = start
        self.end_byte = end
        self.start_point = (0, start)
        self.end_point = (0, end)

    def child_by_field_name(self, name: str) -> _FakeNode | None:
        return self._fields.get(name)


def _fake_walker(source: bytes = b"test") -> Any:
    from types import SimpleNamespace

    tree = cast(Any, SimpleNamespace(root_node=_FakeNode("source_file")))
    return cst_module._Walker(source, tree, None, None)


def test_walk_test_case_ignores_unrecognized_code_children() -> None:
    walker = _fake_walker(b"test")
    case = walker._walk_test_case(
        _FakeNode(
            "test_case",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "code": _FakeNode(
                    "code",
                    children=[
                        _FakeNode("line_comment"),
                        _FakeNode(
                            "python_block",
                            fields={"code": _FakeNode("python_code", start=0, end=4)},
                        ),
                    ],
                ),
            },
        )
    )
    assert case.is_block is True
    assert case.code == "test"


def test_walk_test_case_python_block_body() -> None:
    walker = _fake_walker(b"test")
    case = walker._walk_test_case(
        _FakeNode(
            "test_case",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "code": _FakeNode(
                    "code",
                    children=[
                        _FakeNode(
                            "python_block",
                            fields={"code": _FakeNode("python_code", start=0, end=4)},
                        )
                    ],
                ),
            },
        )
    )
    assert case.is_block is True
    assert case.code == "test"
