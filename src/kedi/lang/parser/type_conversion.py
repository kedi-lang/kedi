from __future__ import annotations

import ast as py_ast
import keyword
from typing import TYPE_CHECKING, List, Optional

from tree_sitter import Node

from kedi.lang.ast import (
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypeLiteral,
    TypePython,
    TypeRef,
    TypeUnion,
)
from kedi.lang.parser.cst_helpers import (
    _decode_type_python_body,
    _field,
    _identifier_name,
    _named_children,
    _src_text,
)

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker


def _walk_type_expr(self: _Walker, node: Node) -> Optional[TypeExpr]:
    # `type_expr` is a thin wrapper; its sole named child is the
    # actual term (type_ref / type_apply / type_union / type_python).
    inner = _named_children(node)
    if not inner:
        return None
    target = inner[0]
    if target.type == "type_ref":
        name = _identifier_name(self.source, _field(target, "name") or _named_children(target)[0])
        return TypeRef(name=name, loc=self._span(target))
    if target.type == "type_apply":
        name_node = _field(target, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        args = [
            self._walk_type_arg(arg) for arg in _named_children(target) if arg.type == "type_arg"
        ]
        args = [a for a in args if a is not None]
        return TypeApply(name=name, args=args, loc=self._span(target))
    if target.type == "type_union":
        left_node = _field(target, "left")
        right_node = _field(target, "right")
        left = self._walk_type_term(left_node) if left_node is not None else None
        right = self._walk_type_term(right_node) if right_node is not None else None
        return TypeUnion(left=left, right=right, loc=self._span(target))
    if target.type == "type_python":
        body_node = _field(target, "code")
        code = _decode_type_python_body(self.source, body_node) if body_node is not None else ""
        return TypePython(code=code, loc=self._span(target))
    # Unknown type_expr shape
    self._error(target, f"unexpected type_expr inner {target.type!r}")
    return None


def _walk_type_arg(self: _Walker, node: Node) -> Optional[TypeExpr]:
    """Walk a ``type_arg`` wrapper inside generic type applications."""
    target = _named_children(node)[0]
    if target.type == "type_python":
        body_node = _field(target, "code")
        code = _decode_type_python_body(self.source, body_node) if body_node is not None else ""
        return TypePython(code=code, loc=self._span(target))
    if target.type == "type_string":
        source = _src_text(self.source, target).strip()
        value = py_ast.literal_eval(source)
        return TypeLiteral(value=value, source=source, loc=self._span(target))
    return self._walk_type_term(target)


def _walk_type_term(self: _Walker, node: Node) -> Optional[TypeExpr]:
    """Walk a node that is a `_type_expr_term` alternative (no `type_expr` wrapper)."""
    t = node.type
    if t == "type_ref":
        name = _identifier_name(self.source, _field(node, "name") or _named_children(node)[0])
        return TypeRef(name=name, loc=self._span(node))
    if t == "type_apply":
        name_node = _field(node, "name")
        assert name_node is not None
        name = _identifier_name(self.source, name_node)
        args = [self._walk_type_arg(arg) for arg in _named_children(node) if arg.type == "type_arg"]
        args = [a for a in args if a is not None]
        return TypeApply(name=name, args=args, loc=self._span(node))
    if t == "type_union":
        left_node = _field(node, "left")
        right_node = _field(node, "right")
        left = self._walk_type_term(left_node) if left_node is not None else None
        right = self._walk_type_term(right_node) if right_node is not None else None
        return TypeUnion(left=left, right=right, loc=self._span(node))
    self._error(node, f"unexpected type term {t!r}")
    return None


def _type_expr_to_str(self: _Walker, node: Node) -> str:
    """Render a type_expr / inner term back to its source slice.

    Used to populate the legacy ``type_annotation`` string field for
    consumers (compiler / codegen / optimizer) that have not yet
    migrated to the structured ``type_expr`` form. Returning the
    raw source slice means the rendering is byte-perfect — including
    the backticks for ``type_python`` cases, matching the legacy
    ``KediType.resolve`` expectation.
    """
    return _src_text(self.source, node).strip()


def _walk_type_def(self: _Walker, node: Node) -> TypeDef:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    if keyword.iskeyword(name):
        self._error(name_node, f"type name {name!r} is a Python keyword", code="K-TYPE-NAME")
    fields: List[TypeField] = []
    seen_names: set[str] = set()
    seen_default = False
    field_list = _field(node, "fields")
    if field_list is not None:
        for f in _named_children(field_list):
            if f.type != "type_field":
                continue
            fname_node = _field(f, "name")
            assert fname_node is not None
            fname = _identifier_name(self.source, fname_node)
            if keyword.iskeyword(fname) or fname.startswith("_"):
                self._error(
                    fname_node,
                    f"field {fname!r} must not be a Python keyword or start with an underscore",
                    code="K-TYPE-FIELD-NAME",
                )
            if fname in seen_names:
                self._error(
                    f,
                    f"duplicate field {fname!r} in type {name!r}",
                    code="K-TYPE-DUP",
                )
            seen_names.add(fname)
            ftype_node = _field(f, "type")
            ftype_anno = self._type_expr_to_str(ftype_node) if ftype_node is not None else None
            ftype_expr = self._walk_type_expr(ftype_node) if ftype_node is not None else None
            default_node = _field(f, "default")
            default_expr = None
            if default_node is not None:
                seen_default = True
                if ftype_node is None:
                    self._error(
                        f,
                        "defaulted type field requires a type annotation",
                        code="K-TYPE-DEFAULT-ANNOTATION",
                    )
                default_expr = self._walk_inline_python_expr(default_node)
            elif seen_default:
                self._error(
                    f,
                    "required field cannot follow a defaulted field",
                    code="K-TYPE-FIELD-ORDER",
                )
            fields.append(
                TypeField(
                    name=fname,
                    type_annotation=ftype_anno,
                    type_expr=ftype_expr,
                    default_expr=default_expr,
                    loc=self._span(f),
                )
            )
    return TypeDef(name=name, fields=fields, loc=self._span(node))
