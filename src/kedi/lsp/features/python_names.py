"""Where embedded Python reads or writes a Kedi binding."""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterator

from tree_sitter import Node

_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
_SCOPES = (*_FUNCTION_SCOPES, ast.ClassDef, *_COMPREHENSIONS)


def python_regions(root: Node, source: bytes) -> Iterator[tuple[str, list[int]]]:
    """Yield each embedded Python region as dedented code and its lines' source byte offsets."""
    pending = [root]
    while pending:
        node = pending.pop()
        if node.type not in ("python_code", "python_inline_body"):
            pending.extend(reversed(node.children))
            continue
        lines = source[node.start_byte : node.end_byte].split(b"\n")
        indent = min(
            (len(line) - len(line.lstrip(b" \t")) for line in lines if line.strip()),
            default=0,
        )
        offsets: list[int] = []
        start = node.start_byte
        for line in lines:
            offsets.append(start + min(indent, len(line)))
            start += len(line) + 1
        yield b"\n".join(line[indent:] for line in lines).decode("utf-8"), offsets


def binding_uses(code: str, name: str) -> list[tuple[int, int]]:
    """Return ``(line index, byte column)`` of each ``name`` that refers to the module binding.

    Kedi bindings are the Python globals of embedded code, so a name refers to the binding
    unless a function, lambda, comprehension or class scope around it makes the name local.
    ``global`` declarations of the name are included, because a rename must keep them.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    uses: list[tuple[int, int]] = []
    declarations: list[ast.Global] = []

    def visit(node: ast.AST, shadowed: bool) -> None:
        if isinstance(node, ast.Name) and node.id == name and not shadowed:
            uses.append((node.lineno - 1, node.col_offset))
        if isinstance(node, _FUNCTION_SCOPES):
            for outer in _evaluated_outside(node):
                visit(outer, shadowed)
            declared = [
                stmt
                for stmt in _own_nodes(node)
                if isinstance(stmt, ast.Global) and name in stmt.names
            ]
            declarations.extend(declared)
            inner = False if declared else shadowed or name in _local_names(node)
            for child in _scope_body(node):
                visit(child, inner)
            return
        if isinstance(node, ast.ClassDef):
            for outer in [*node.decorator_list, *node.bases, *node.keywords]:
                visit(outer, shadowed)
            local = shadowed or name in _local_names(node)
            for stmt in node.body:
                # Functions in a class body do not see the class scope.
                visit(stmt, shadowed if isinstance(stmt, _FUNCTION_SCOPES) else local)
            return
        if isinstance(node, _COMPREHENSIONS):
            first, *rest = node.generators
            visit(first.iter, shadowed)
            inner = shadowed or name in _local_names(node)
            visit(first.target, inner)
            for condition in first.ifs:
                visit(condition, inner)
            for generator in rest:
                visit(generator, inner)
            elements = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            for element in elements:
                visit(element, inner)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, shadowed)

    visit(tree, False)
    uses.extend(_declared_name_tokens(code, declarations, name))
    return sorted(set(uses))


def _evaluated_outside(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.AST]:
    """Decorators, defaults and annotations run in the enclosing scope."""
    arguments = node.args
    outside: list[ast.AST] = [*arguments.defaults, *(d for d in arguments.kw_defaults if d)]
    if not isinstance(node, ast.Lambda):
        outside.extend(node.decorator_list)
        outside.extend(arg.annotation for arg in _parameters(node) if arg.annotation)
        if node.returns is not None:
            outside.append(node.returns)
    return outside


def _scope_body(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.AST]:
    return [node.body] if isinstance(node, ast.Lambda) else list(node.body)


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.arg]:
    arguments = node.args
    extra = [arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]
    return [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, *extra]


def _own_nodes(
    scope: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef,
) -> Iterator[ast.AST]:
    """Nodes a scope evaluates itself: nested scopes are yielded but not entered."""
    pending: list[ast.AST] = (
        list(scope.body) if isinstance(scope, ast.ClassDef) else _scope_body(scope)
    )
    while pending:
        current = pending.pop()
        yield current
        if not isinstance(current, _SCOPES):
            pending.extend(ast.iter_child_nodes(current))


def _local_names(node: ast.AST) -> set[str]:
    """Names a scope binds for itself."""
    if isinstance(node, _COMPREHENSIONS):
        return {
            target.id
            for generator in node.generators
            for target in ast.walk(generator.target)
            if isinstance(target, ast.Name)
        }
    assert isinstance(node, (*_FUNCTION_SCOPES, ast.ClassDef))
    names = {arg.arg for arg in _parameters(node)} if isinstance(node, _FUNCTION_SCOPES) else set()
    for current in _own_nodes(node):
        if isinstance(current, ast.Name) and isinstance(current.ctx, (ast.Store, ast.Del)):
            names.add(current.id)
        elif isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(current.name)
        elif isinstance(current, ast.alias):
            names.add((current.asname or current.name).split(".")[0])
        elif isinstance(current, ast.ExceptHandler) and current.name:
            names.add(current.name)
        elif isinstance(current, ast.Nonlocal):
            names.update(current.names)
    return names


def _declared_name_tokens(
    code: str, declarations: list[ast.Global], name: str
) -> list[tuple[int, int]]:
    """Positions of ``name`` inside ``global`` statements, found with the tokenizer."""
    if not declarations:
        return []
    lines = {
        line for stmt in declarations for line in range(stmt.lineno, (stmt.end_lineno or 0) + 1)
    }
    found: list[tuple[int, int]] = []
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        row, column = token.start
        if token.type == tokenize.NAME and token.string == name and row in lines:
            line = code.split("\n")[row - 1]
            found.append((row - 1, len(line[:column].encode("utf-8"))))
    return found
