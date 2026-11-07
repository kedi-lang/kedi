"""Forms: bodies that replace a group's source body, and the gate each one passes first.

A form is Kedi text for one group, named by the group's qualified name in the root module
(``solve.planner``) or by ``<module>:<qualified name>`` in an imported one. Before the program
runs, each form must:

1. parse as the group's body, under the same body rules, and as exactly one template block
   for an ``> optimize:`` span that no optimizer result already replaces;
2. contain no Python: no Python blocks, backtick expressions or types, Python conditions, or
   loops, whose iterables are Python;
3. keep the group's interface: bind every name the source body binds and later code reads,
   with the same declared type; read only names visible at the group; and bind or reassign no
   name visible before the group unless the source body does too.

A form that breaks a rule is rejected with that rule, and the program does not run.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import textwrap
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kedi.groups import form_fingerprint
from kedi.lang.ast import (
    ActiveForm,
    AssignmentBlockStmt,
    AssignmentStmt,
    AwaitStmt,
    BacktickLineStmt,
    CallSegment,
    ConditionalLoopStmt,
    FormProducer,
    GroupStmt,
    IfStmt,
    InputSegment,
    LiteralArg,
    LoopStmt,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    Program,
    PythonBlock,
    PythonExprNativeSegment,
    PythonExprSegment,
    RawInvokeStmt,
    ReturnBlockStmt,
    ReturnStmt,
    Segment,
    SendStmt,
    ShowStmt,
    Stmt,
    TaskGroupStmt,
    TaskStmt,
    TemplateBlock,
    TypeApply,
    TypeExpr,
    TypePython,
    TypeUnion,
    VariableInitBlockStmt,
    VariableInitStmt,
)

if TYPE_CHECKING:
    from kedi.core import KediRuntime

_FINGERPRINT = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z_]\w*")
_BUILTINS = frozenset(dir(builtins))
_PYTHON_STATEMENTS = (
    PythonBlock,
    BacktickLineStmt,
    VariableInitBlockStmt,
    ReturnBlockStmt,
    AssignmentBlockStmt,
)


class FormError(ValueError):
    """A form that cannot run: an invalid forms file, an unknown group, or a broken gate rule."""


@dataclass(frozen=True, slots=True)
class Form:
    """A body for one group.

    ``parent`` is the fingerprint of the form this one derives from; ``note`` is the producer's
    reason for it. Neither is part of the form's identity. ``origin`` names the text's source in
    diagnostics.
    """

    text: str
    parent: str | None = None
    note: str | None = None
    origin: str = "<form>"

    def __post_init__(self) -> None:
        if self.parent is not None and not _FINGERPRINT.fullmatch(self.parent):
            raise FormError(
                f"Form parent must be a 64-character lowercase hex fingerprint, got {self.parent!r}"
            )


def load_forms(path: str | Path) -> dict[str, Form]:
    """Read a forms file: ``{"<group>": {"body": <file> | "text": <kedi>, "parent", "note"}}``.

    ``body`` names a file relative to the forms file; ``parent`` and ``note`` are optional.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FormError(f"Cannot read forms file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FormError(f"Forms file {path} must hold an object keyed by group name")
    return {key: _form_entry(path, key, entry) for key, entry in raw.items()}


def _form_entry(path: Path, key: str, entry: Any) -> Form:
    where = f"Forms file {path}, group '{key}'"
    if not isinstance(entry, dict):
        raise FormError(f"{where}: expected an object")
    unknown = sorted(set(entry) - {"body", "text", "parent", "note"})
    if unknown:
        raise FormError(f"{where}: unknown fields {', '.join(unknown)}")
    if ("body" in entry) == ("text" in entry):
        raise FormError(f"{where}: give exactly one of 'body' and 'text'")
    if not all(isinstance(value, str) for value in entry.values()):
        raise FormError(f"{where}: every field must be a string")
    if "text" in entry:
        return Form(entry["text"], entry.get("parent"), entry.get("note"), f"<form {key}>")
    body = path.parent / entry["body"]
    try:
        text = body.read_text(encoding="utf-8")
    except OSError as exc:
        raise FormError(f"{where}: cannot read body {body}: {exc}") from exc
    return Form(text, entry.get("parent"), entry.get("note"), str(body))


def propose_form(runtime: KediRuntime, group: str, form: Form) -> str:
    """Gate ``form`` for a group of a compiled program and make it the group's active form.

    The form runs from the group's next execution and is recorded with producer ``runtime``.
    Returns its fingerprint; a rejected form raises and leaves the active form in place.
    """
    target, _, qualified = group.rpartition(":")
    module = runtime._module_runtimes.get(target or runtime.module_name)
    if module is None:
        raise FormError(f"Form for group '{group}': the program has no module '{target}'")
    # Compilation sets the program before it registers the module's runtime.
    program = cast(Program, module._program)
    names = frozenset(provided_names(module)) | _program_names(program)
    active = _install(
        program,
        group,
        _group(program, group, qualified, module.module_name),
        form,
        names,
        producer="runtime",
    )
    return form_fingerprint(active.canonical)


def provided_names(runtime: KediRuntime) -> set[str]:
    """Names a module's runtime provides besides its own definitions: imports, prelude, globals."""
    return {
        *runtime._initial_globals_env,
        *runtime._procedures,
        *runtime._type_env_stack[0],
        *runtime._profile_registry,
        *runtime._import_env,
        *(runtime._prelude_env or {}),
    }


def _group(program: Program, key: str, qualified: str, module: str) -> GroupStmt:
    group = program.groups.get(qualified)
    if group is None:
        known = ", ".join(sorted(program.groups)) or "none"
        raise FormError(f"Form for unknown group '{key}': module '{module}' has groups {known}")
    return group


class FormActivation:
    """The forms of one program run, gated and installed module by module as each compiles."""

    def __init__(self, forms: Mapping[str, Form]) -> None:
        self._pending = dict(forms)

    def apply(
        self, program: Program, *, module: str, root: bool, module_names: Iterable[str]
    ) -> None:
        """Gate and install the forms for ``module``; raise on the first rejected one.

        ``module_names`` are the names the compiled module already provides: imports, prelude
        values, and runtime globals.
        """
        names: frozenset[str] | None = None
        for key in list(self._pending):
            target, _, qualified = key.rpartition(":")
            if target != module and not (root and not target):
                continue
            form = self._pending.pop(key)
            group = _group(program, key, qualified, module)
            if names is None:
                names = frozenset(module_names) | _program_names(program)
            _install(program, key, group, form, names, producer="supplied")

    def finish(self) -> None:
        """Reject forms whose module the program never imported."""
        if self._pending:
            raise FormError(
                "Forms for groups in modules the program does not import: "
                + ", ".join(sorted(self._pending))
            )


def _install(
    program: Program,
    key: str,
    group: GroupStmt,
    form: Form,
    module_names: frozenset[str],
    *,
    producer: FormProducer,
) -> ActiveForm:
    """Gate ``form`` against the group's source body, then make it the group's active form."""
    body, canonical = _parse(key, group, form)
    if group.kind == "optimize":
        if len(body) != 1 or not isinstance(body[0], TemplateBlock):
            raise _rejected(key, "parse", "an '> optimize:' form is exactly one template block")
        if _optimizer_result(group):
            raise _rejected(
                key, "optimizer", "the span already runs an optimizer result (.optimized.json)"
            )
    python = _python_use(body)
    if python is not None:
        raise _rejected(key, "python", python)
    site = _site(program, group)
    _check_interface(key, group, body, site, module_names)
    active = ActiveForm(body, canonical, producer, form.parent, form.note)
    block = body[0]
    if group.kind == "optimize" and site.procedure is not None and isinstance(block, TemplateBlock):
        span = next(span for span in site.procedure.optimize_spans if span.name == group.name)
        span.template_lines = block.lines
    group.form = active
    return active


def _rejected(key: str, rule: str, detail: str) -> FormError:
    return FormError(f"Form for group '{key}' rejected ({rule}): {detail}")


def _parse(key: str, group: GroupStmt, form: Form) -> tuple[list[Stmt], str]:
    from kedi.lang.parser import parse_program
    from kedi.lang.parser.control_flow_validation import validate_control_flow
    from kedi.lang.parser.diagnostics import KediParseError

    text = textwrap.dedent(form.text).strip("\n")
    if not text.strip():
        raise _rejected(key, "parse", "the form is empty")
    # The header puts the form where a group body goes; diagnostics count the form's own lines.
    source = f"> evolve: {group.name}:\n{textwrap.indent(text, '  ')}\n"
    try:
        parsed = parse_program(source, source_path=form.origin)
        validate_control_flow(parsed)
    except KediParseError as exc:
        diagnostic = exc.diagnostics[0]
        line = "" if diagnostic.span is None else f"line {diagnostic.span.start_line - 1}: "
        raise _rejected(key, "parse", f"{line}{diagnostic.message}") from exc
    wrapped = parsed.groups[group.name]
    return wrapped.body, wrapped.canonical


def _optimizer_result(group: GroupStmt) -> bool:
    block = group.body[0] if group.body else None
    return isinstance(block, TemplateBlock) and any(
        line.optimized_prompt_prefix or line.optimized_output_override for line in block.lines
    )


# ---------------------------------------------------------------------------------------------
# Python


def _python_use(statements: Sequence[Stmt]) -> str | None:
    for statement in _statements(statements):
        what = _python_statement(statement)
        if what is not None:
            line = "" if statement.loc is None else f"line {statement.loc.start_line - 1}: "
            return f"{line}{what}; forms cannot contain Python"
    return None


def _python_statement(statement: Stmt) -> str | None:
    if isinstance(statement, _PYTHON_STATEMENTS):
        return "a Python block or backtick statement"
    if (
        isinstance(statement, (IfStmt, ConditionalLoopStmt))
        and statement.condition_segments is None
    ):
        return "a Python condition"
    if isinstance(statement, LoopStmt):
        return "a loop, whose iterable is Python"
    types = (
        [statement.type_expr] if isinstance(statement, (RawInvokeStmt, VariableInitStmt)) else []
    )
    for segment in _all_segments(statement):
        if isinstance(segment, (PythonExprSegment, PythonExprNativeSegment)):
            return "a backtick expression"
        if isinstance(segment, OutputSegment):
            types.append(segment.type_expr)
    return "a backtick type" if any(map(_python_type, types)) else None


def _python_type(expr: TypeExpr | None) -> bool:
    if isinstance(expr, TypeApply):
        return any(map(_python_type, expr.args))
    if isinstance(expr, TypeUnion):
        return _python_type(expr.left) or _python_type(expr.right)
    return isinstance(expr, TypePython)


# ---------------------------------------------------------------------------------------------
# Interface


@dataclass(frozen=True, slots=True)
class _Site:
    """Where a group runs: its procedure, the local names before it, and what runs after it."""

    procedure: ProcedureDef | None
    before: frozenset[str]
    after: tuple[Stmt, ...]


def _check_interface(
    key: str, group: GroupStmt, body: list[Stmt], site: _Site, module_names: frozenset[str]
) -> None:
    source_binds = _bound(group.body)
    form_binds = _bound(body)
    read_after = _reads(site.after)
    for name, declared in source_binds.items():
        if name not in read_after:
            continue
        if name not in form_binds:
            raise _rejected(
                key, "interface", f"the form must bind '{name}', which code after the group reads"
            )
        if _type_text(form_binds[name]) != _type_text(declared):
            raise _rejected(
                key,
                "interface",
                f"'{name}' is declared as {_type_text(declared)} in the source but "
                f"{_type_text(form_binds[name])} in the form",
            )
    unknown = _free_reads(body, site.before) - module_names - _BUILTINS
    if unknown:
        raise _rejected(
            key,
            "interface",
            "the form reads names the group cannot see: " + ", ".join(sorted(unknown)),
        )
    changed = set(form_binds) | _assigned(body)
    outside = (changed - set(source_binds) - _assigned(group.body)) & (site.before | module_names)
    if outside:
        raise _rejected(
            key,
            "interface",
            "the form rebinds names visible before the group that the source does not: "
            + ", ".join(sorted(outside)),
        )


def _type_text(annotation: str | None) -> str:
    return "".join((annotation or "str").split())


def _site(program: Program, group: GroupStmt) -> _Site:
    found = _search(program.ordered_toplevel, frozenset(), None, group)
    assert found is not None, "Program.groups holds only groups placed in the program"
    if found.procedure is None:
        # Top-level names are globals: every procedure may read them after the group runs.
        bodies = tuple(stmt for procedure in program.procedures for stmt in procedure.body)
        found = _Site(None, found.before, found.after + bodies)
    return found


def _search(
    block: Sequence[Any], before: frozenset[str], procedure: ProcedureDef | None, group: GroupStmt
) -> _Site | None:
    for index, item in enumerate(block):
        if item is group:
            return _Site(procedure, before | set(_bound(block[:index])), tuple(block[index + 1 :]))
        nested = item.proc if isinstance(item, ProcDefStmt) else item
        if isinstance(nested, ProcedureDef):
            found = _search(nested.body, frozenset(nested.params), nested, group)
        else:
            found = None
            visible = before | set(_bound(block[:index]))
            for child, extra in _child_blocks(item):
                found = found or _search(child, visible | extra, procedure, group)
        if found is not None:
            return found
    return None


def _child_blocks(statement: Any) -> list[tuple[Sequence[Stmt], frozenset[str]]]:
    """A statement's nested blocks, each with the names it adds to the scope it opens."""
    if isinstance(statement, IfStmt):
        return [(statement.body, frozenset()), (statement.else_body or [], frozenset())]
    if isinstance(statement, LoopStmt):
        binder = frozenset({statement.binder})
        mapped = binder | set(_bound(statement.body))
        return [(statement.body, binder), (statement.map_body or [], mapped)]
    if isinstance(statement, ConditionalLoopStmt):
        return [(statement.body, frozenset())]
    if isinstance(statement, TaskGroupStmt):
        return [
            (arm.body, frozenset({arm.wait.binder} if arm.wait.binder else ()))
            for arm in statement.arms
            if arm.body is not None
        ]
    return []


def _bound(statements: Sequence[Any]) -> dict[str, str | None]:
    """Names a block binds in its own scope, with their declared types; groups are transparent."""
    bound: dict[str, str | None] = {}
    for statement in statements:
        if isinstance(statement, GroupStmt):
            bound.update(_bound(statement.body))
        elif isinstance(statement, (RawInvokeStmt, VariableInitStmt, VariableInitBlockStmt)):
            bound[statement.name] = statement.type_annotation
        elif isinstance(statement, TaskStmt):
            bound[statement.binder] = None
        elif isinstance(statement, AwaitStmt) and statement.binder:
            bound[statement.binder] = None
        elif isinstance(statement, ProcDefStmt) and statement.proc is not None:
            bound[statement.proc.name] = None
        elif isinstance(statement, TemplateBlock):
            for segment in _all_segments(statement):
                if isinstance(segment, OutputSegment):
                    bound[segment.name] = segment.type_annotation
    return bound


def _assigned(statements: Sequence[Stmt]) -> set[str]:
    """Names a block reassigns with ``:=``, at any depth: they write to their owning scope."""
    return {
        statement.name
        for statement in _statements(statements)
        if isinstance(statement, (AssignmentStmt, AssignmentBlockStmt))
    }


def _free_reads(statements: Sequence[Stmt], bound: Iterable[str]) -> set[str]:
    """Names a block reads before binding them itself."""
    visible = set(bound)
    free: set[str] = set()
    for statement in statements:
        free |= _own_reads(statement) - visible
        for child, extra in _child_blocks(statement):
            free |= _free_reads(child, visible | extra)
        visible |= set(_bound([statement]))
    return free


def _reads(statements: Sequence[Any]) -> set[str]:
    """Every name a block reads, at any depth, including nested procedure bodies."""
    names: set[str] = set()
    for statement in _statements(statements, procedures=True):
        names |= _own_reads(statement)
    return names


def _own_reads(statement: Any) -> set[str]:
    names = _segment_reads(_all_segments(statement))
    for code in _python_code(statement):
        names |= _python_names(code)
    if isinstance(statement, (AwaitStmt, SendStmt)):
        names.add(statement.handle)
    if isinstance(statement, TaskGroupStmt):
        names |= {arm.wait.handle for arm in statement.arms}
    return names


def _python_code(statement: Any) -> list[str]:
    if isinstance(statement, _PYTHON_STATEMENTS):
        return [statement.code]
    if (
        isinstance(statement, (IfStmt, ConditionalLoopStmt))
        and statement.condition_segments is None
    ):
        return [statement.condition_code]
    if isinstance(statement, LoopStmt):
        return [statement.iterable_code]
    return []


def _segment_reads(segments: Iterable[Segment]) -> set[str]:
    names: set[str] = set()
    for segment in segments:
        if isinstance(segment, (InputSegment, CallSegment)):
            match = _NAME.match(segment.name)
            names |= set() if match is None else {match.group()}
        elif isinstance(segment, (PythonExprSegment, PythonExprNativeSegment)):
            names |= _python_names(segment.code)
    return names


def _python_names(code: str) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(code).strip())
    except SyntaxError:
        return set()  # Reported where the code is compiled.
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


# ---------------------------------------------------------------------------------------------
# Traversal


def _statements(statements: Sequence[Any], *, procedures: bool = False) -> Iterator[Any]:
    """Every statement in a block, depth first; nested procedures only when asked."""
    for statement in statements:
        yield statement
        for child, _ in _child_blocks(statement):
            yield from _statements(child, procedures=procedures)
        if isinstance(statement, GroupStmt):
            yield from _statements(statement.body, procedures=procedures)
        elif procedures and isinstance(statement, ProcDefStmt) and statement.proc is not None:
            yield from _statements(statement.proc.body, procedures=procedures)


def _all_segments(statement: Any) -> Iterator[Segment]:
    """Every segment of a statement's own expressions, including call arguments."""
    for segments in _expressions(statement):
        yield from _nested_segments(segments)


def _expressions(statement: Any) -> list[Sequence[Segment]]:
    if isinstance(statement, TemplateBlock):
        return [line.segments for line in statement.lines]
    if isinstance(statement, (TaskStmt, SendStmt)):
        return [line.segments for line in statement.template.lines]
    if isinstance(statement, (RawInvokeStmt, ShowStmt, ReturnStmt)):
        return [statement.segments]
    if isinstance(statement, (VariableInitStmt, AssignmentStmt)):
        return [statement.expr]
    if isinstance(statement, (IfStmt, ConditionalLoopStmt)) and statement.condition_segments:
        return [statement.condition_segments]
    return []


def _nested_segments(segments: Sequence[Segment]) -> Iterator[Segment]:
    for segment in segments:
        yield segment
        if not isinstance(segment, CallSegment):
            continue
        for argument in [arg for arg in segment.args if isinstance(arg, LiteralArg)]:
            if argument.native_backtick is not None:
                yield PythonExprNativeSegment(code=argument.native_backtick)
            inner = [argument.python_segment, argument.call_segment, *argument.render_segments]
            yield from _nested_segments([part for part in inner if part is not None])


def _program_names(program: Program) -> frozenset[str]:
    """Module-level names the program itself declares."""
    names = {procedure.name for procedure in program.procedures}
    names |= {typedef.name for typedef in program.type_defs}
    names |= set(program.profile_defs)
    names |= set(_bound(program.toplevel_body))
    names |= _assigned(program.toplevel_body)
    for module_import in program.imports:
        names |= set(module_import.names)
    return frozenset(names)


__all__ = ["Form", "FormActivation", "FormError", "load_forms", "propose_form", "provided_names"]
