"""Lexical adapter selection without evaluating program expressions."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from textwrap import dedent

from tree_sitter import Node


def selected_adapter(source: bytes, node: Node) -> str | None:
    root = node
    while root.parent is not None:
        root = root.parent
    python_names = _python_names(source, root.named_children)
    definitions = {
        source[name.start_byte : name.end_byte].decode("utf-8"): definition
        for definition in root.named_children
        if definition.type in {"profile_directive", "capability_directive"}
        and (name := definition.child_by_field_name("name")) is not None
    }
    # Procedure tools take precedence over profiles regardless of their order.
    definitions.update(
        {
            source[name.start_byte : name.end_byte].decode("utf-8"): definition
            for definition in root.named_children
            if definition.type == "procedure_def"
            and (name := definition.child_by_field_name("name")) is not None
        }
    )
    owner: Node | None = node
    while owner is not None:
        if owner.type in {"profile_directive", "capability_directive"}:
            body = owner.child_by_field_name("body")
            assert body is not None
            # A profile is an independent patch, not a lexical snapshot of the
            # backend outside its definition. Its final selection owns options.
            return _selection(
                source,
                reversed(body.named_children),
                definitions,
                frozenset(),
                python_names,
                focus=node,
            )[1]
        owner = owner.parent
    cursor: Node | None = node
    while cursor is not None:
        previous = cursor.prev_named_sibling
        siblings = []
        while previous is not None:
            siblings.append(previous)
            previous = previous.prev_named_sibling
        found, adapter = _selection(source, siblings, definitions, frozenset(), python_names)
        if found:
            return adapter
        cursor = cursor.parent
    return None


def _selection(
    source: bytes,
    nodes: Iterable[Node],
    definitions: dict[str, Node],
    active: frozenset[str],
    python_names: frozenset[str] | None,
    *,
    focus: Node | None = None,
) -> tuple[bool, str | None]:
    """Resolve newest-first contributions; unknown imports stay unknown."""
    for node in nodes:
        if node.type in {"profile_if_stmt", "capability_if_stmt"}:
            branches = [node.child_by_field_name(name) for name in ("consequence", "alternative")]
            if focus is not None:
                branches = [
                    branch
                    for branch in branches
                    if branch is not None
                    and branch.start_byte <= focus.start_byte < branch.end_byte
                ] or branches
            selections = [
                _selection(
                    source,
                    reversed(branch.named_children),
                    definitions,
                    active,
                    python_names,
                    focus=focus,
                )
                if branch is not None
                else (False, None)
                for branch in branches
            ]
            if any(found for found, _ in selections):
                return (
                    selections[0]
                    if all(item == selections[0] for item in selections)
                    else (True, None)
                )
            continue
        if node.type in {"adapter_directive", "agent_directive"}:
            value = node.child_by_field_name("value")
            if value is not None and value.type != "inline_python_expr":
                return True, source[value.start_byte : value.end_byte].decode("utf-8").strip()
            return True, None
        if node.type != "use_directive":
            continue
        name = node.child_by_field_name("name")
        if name is not None:
            targets = [name]
        else:
            body = node.child_by_field_name("body")
            assert body is not None
            targets = [
                name
                for item in body.named_children
                if (name := item.child_by_field_name("name")) is not None
            ]
        for name in reversed(targets):
            target = source[name.start_byte : name.end_byte].decode("utf-8")
            if python_names is None or target in python_names:
                return True, None
            definition = definitions.get(target)
            if definition is None or target in active:
                return True, None
            if definition.type != "profile_directive":
                continue
            body = definition.child_by_field_name("body")
            assert body is not None
            found, adapter = _selection(
                source, reversed(body.named_children), definitions, active | {target}, python_names
            )
            if found:
                return True, adapter
    return False, None


def _python_names(source: bytes, nodes: Iterable[Node]) -> frozenset[str] | None:
    """Python bindings may shadow profiles; never execute them to guess their kind."""
    names: set[str] = set()
    for node in nodes:
        if node.type != "python_block":
            continue
        code = node.child_by_field_name("code")
        assert code is not None
        try:
            module = ast.parse(dedent(source[code.start_byte : code.end_byte].decode("utf-8")))
        except SyntaxError:
            return None
        for binding in ast.walk(module):
            if isinstance(binding, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(binding.name)
            elif isinstance(binding, ast.Name) and isinstance(binding.ctx, ast.Store):
                names.add(binding.id)
            elif isinstance(binding, ast.alias):
                if binding.name == "*":
                    return None
                names.add(binding.asname or binding.name.partition(".")[0])
            elif (
                isinstance(binding, ast.Call)
                and isinstance(binding.func, ast.Name)
                and binding.func.id == "exec"
            ):
                return None
    return frozenset(names)
