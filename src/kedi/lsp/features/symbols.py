from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lsprotocol import types as lsp

from kedi.lang.ast import (
    EvalSuite,
    InputSegment,
    OutputSegment,
    ProcedureDef,
    ProfileDefStmt,
    Program,
    ProgramWithValidation,
    Span,
    TestSuite,
    UseDirectiveStmt,
    iter_profile_bodies,
)
from kedi.lsp.ast_spans import name_token_span
from kedi.lsp.coordinates import span_to_range as _span_to_range
from kedi.lsp.module_surface import ImportedSurface


@dataclass
class SymbolRef:
    """The symbol under the cursor, used by references / rename / hover."""

    kind: str
    name: str
    declaration_span: Optional[Span]
    scope_proc: Optional[ProcedureDef]
    # The span of the identifier token under the cursor (for prepareRename).
    cursor_span: Span


@dataclass(frozen=True)
class _ValueDeclaration:
    kind: str
    name: str
    declaration_span: Span
    scope_proc: Optional[ProcedureDef]


def _merged_profile_defs(
    prog: Program | ProgramWithValidation,
    imported_surface: ImportedSurface,
) -> dict[str, ProfileDefStmt]:
    merged = dict(prog.profile_defs)
    merged.update(imported_surface.profiles)
    return merged


def _validation_suites(
    prog: Program | ProgramWithValidation,
) -> tuple[TestSuite | EvalSuite, ...]:
    if not isinstance(prog, ProgramWithValidation):
        return ()
    return (*prog.tests, *prog.evals)


def _output_name_span(seg: OutputSegment) -> Optional[Span]:
    if seg.loc is None:
        return None
    return name_token_span(seg.loc, seg.name, prefix_len=1)


def _input_name_span(seg: InputSegment) -> Optional[Span]:
    if seg.loc is None:
        return None
    return name_token_span(seg.loc, seg.name, prefix_len=1)


def _program_line_text(
    prog: Program | ProgramWithValidation,
    line_number: int,
    source_path: Optional[str] = None,
) -> Optional[str]:
    if prog.source_text:
        lines = prog.source_text.splitlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
    source_map = prog.source_map
    if source_map is not None:
        path = source_path or (prog.loc.source_path if prog.loc is not None else None)
        if path:
            return source_map.get_line(path, line_number)
    return None


def _line_start_byte(
    prog: Program | ProgramWithValidation,
    line_number: int,
) -> Optional[int]:
    if not prog.source_text:
        return None
    offset = 0
    for index, line in enumerate(prog.source_text.splitlines(keepends=True), start=1):
        if index == line_number:
            return offset
        offset += len(line.encode("utf-8"))
    return None


def _line_start_byte_for_program(
    prog: Program | ProgramWithValidation,
    line_number: int,
) -> Optional[int]:
    """Return the UTF-8 byte offset of a source line in ``prog``."""
    return _line_start_byte(prog, line_number)


def _name_token_span_on_line(
    prog: Program | ProgramWithValidation,
    *,
    line_number: int,
    source_path: Optional[str],
    name: str,
    search_start: int = 0,
) -> Optional[Span]:
    line = _program_line_text(prog, line_number, source_path)
    line_start = _line_start_byte(prog, line_number)
    if line is None or line_start is None:
        return None
    for candidate in (f"`{name}`", name):
        name_index = line.find(candidate, search_start)
        if name_index < 0:
            continue
        start_col = name_index + 1
        start_byte = line_start + len(line[:name_index].encode("utf-8"))
        return Span(
            start_byte=start_byte,
            end_byte=start_byte + len(candidate.encode("utf-8")),
            start_line=line_number,
            start_col=start_col,
            end_line=line_number,
            end_col=start_col + len(candidate),
            source_path=source_path,
        )
    return None


def _use_tool_entry_spans(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
) -> list[tuple[str, Span]]:
    if stmt.loc is None or not stmt.tool_names:
        return []
    spans: list[tuple[str, Span]] = []
    seen: set[tuple[str, int]] = set()
    for line_number in range(stmt.loc.start_line, stmt.loc.end_line + 1):
        for tool_name in stmt.tool_names:
            name_span = _name_token_span_on_line(
                prog,
                line_number=line_number,
                source_path=stmt.loc.source_path,
                name=tool_name,
            )
            if name_span is None:
                continue
            key = (tool_name, name_span.start_line)
            if key in seen:
                continue
            seen.add(key)
            spans.append((tool_name, name_span))
    return spans


def _profile_use_entry_spans(
    prog: Program | ProgramWithValidation,
    profile: ProfileDefStmt,
) -> list[tuple[str, Span]]:
    """Return target spans from a profile's ``> use:`` members.

    Single-line targets may resolve to either a tool or another profile.
    Multiline targets are explicit tools. Reconstruct declaration-like spans
    only from directive lines and their indented entries so profile prose
    cannot become a false reference.
    """
    target_names = tuple(
        name
        for body in iter_profile_bodies(profile)
        for name in (*body.use_targets, *body.tool_names)
    )
    if profile.loc is None or not target_names:
        return []

    spans: list[tuple[str, Span]] = []
    seen: set[tuple[str, int, int]] = set()
    in_use_body = False
    for line_number in range(profile.loc.start_line, profile.loc.end_line + 1):
        line = _program_line_text(prog, line_number, profile.loc.source_path)
        if line is None:
            continue
        stripped = line.strip()
        if stripped.startswith("> use:"):
            in_use_body = True
            search_start = line.find(":") + 1
        elif stripped.startswith(">"):
            in_use_body = False
            continue
        elif not in_use_body or not stripped:
            continue
        else:
            search_start = len(line) - len(line.lstrip())

        for target_name in target_names:
            name_span = _name_token_span_on_line(
                prog,
                line_number=line_number,
                source_path=profile.loc.source_path,
                name=target_name,
                search_start=search_start,
            )
            if name_span is None:
                continue
            key = (target_name, name_span.start_line, name_span.start_col)
            if key not in seen:
                seen.add(key)
                spans.append((target_name, name_span))
    return spans


def _use_single_target_span(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
) -> Optional[Span]:
    if stmt.loc is None or not stmt.name:
        return None
    line = _program_line_text(prog, stmt.loc.start_line, stmt.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    return _name_token_span_on_line(
        prog,
        line_number=stmt.loc.start_line,
        source_path=stmt.loc.source_path,
        name=stmt.name,
        search_start=search_start,
    )


def _validation_procedure_span(
    prog: Program | ProgramWithValidation,
    suite: TestSuite | EvalSuite,
) -> Optional[Span]:
    """Return the procedure-name token in a validation-suite header."""
    if suite.loc is None:
        return None
    line = _program_line_text(prog, suite.loc.start_line, suite.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    return _name_token_span_on_line(
        prog,
        line_number=suite.loc.start_line,
        source_path=suite.loc.source_path,
        name=suite.procedure_name,
        search_start=search_start,
    )


def _directive_name_span(
    prog: Program | ProgramWithValidation,
    loc: Span,
    name: str,
) -> Optional[Span]:
    if loc is None:
        return None
    line = _program_line_text(prog, loc.start_line, loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    name_index = line.find(name, search_start)
    if name_index < 0:
        return None
    prefix = line[:name_index]
    start_col = name_index + 1
    line_start = _line_start_byte_for_program(prog, loc.start_line)
    if line_start is None:
        return None
    start_byte = line_start + len(prefix.encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(name.encode("utf-8")),
        start_line=loc.start_line,
        start_col=start_col,
        end_line=loc.start_line,
        end_col=start_col + len(name),
        source_path=loc.source_path,
    )


def _import_name_span(
    prog: Program | ProgramWithValidation,
    module_import,
) -> Optional[Span]:
    if module_import.loc is None:
        return None
    line = _program_line_text(prog, module_import.loc.start_line, module_import.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    name_index = line.find(module_import.module, search_start)
    if name_index < 0:
        return None
    prefix = line[:name_index]
    start_col = name_index + 1
    line_start = _line_start_byte_for_program(prog, module_import.loc.start_line)
    if line_start is None:
        return None
    start_byte = line_start + len(prefix.encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(module_import.module.encode("utf-8")),
        start_line=module_import.loc.start_line,
        start_col=start_col,
        end_line=module_import.loc.start_line,
        end_col=start_col + len(module_import.module),
        source_path=module_import.loc.source_path,
    )


def _approval_value_span(
    prog: Program | ProgramWithValidation,
    loc: Span,
    plain_value: str | None,
) -> Span | None:
    if plain_value:
        return _directive_name_span(prog, loc, plain_value)
    return _directive_value_span(prog, loc)


def _directive_value_span(
    prog: Program | ProgramWithValidation,
    loc: Span,
) -> Span | None:
    """Return the non-whitespace value portion of one single-line directive."""
    line = _program_line_text(prog, loc.start_line, loc.source_path)
    line_start = _line_start_byte(prog, loc.start_line)
    if line is None or line_start is None:
        return None
    colon_index = line.find(":")
    if colon_index < 0:
        return None
    start_index = colon_index + 1
    while start_index < len(line) and line[start_index].isspace():
        start_index += 1
    end_index = len(line)
    while end_index > start_index and line[end_index - 1].isspace():
        end_index -= 1
    if start_index == end_index:
        return None
    start_byte = line_start + len(line[:start_index].encode("utf-8"))
    end_byte = line_start + len(line[:end_index].encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=loc.start_line,
        start_col=start_index + 1,
        end_line=loc.start_line,
        end_col=end_index + 1,
        source_path=loc.source_path,
    )


def _directive_keyword_span(
    prog: Program | ProgramWithValidation,
    loc: Span | None,
    keyword: str,
) -> Span | None:
    if loc is None:
        return None
    line = _program_line_text(prog, loc.start_line, loc.source_path)
    line_start = _line_start_byte(prog, loc.start_line)
    if line is None or line_start is None:
        return None
    name_index = line.find(keyword)
    if name_index < 0:
        return None
    start_byte = line_start + len(line[:name_index].encode("utf-8"))
    start_col = name_index + 1
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(keyword.encode("utf-8")),
        start_line=loc.start_line,
        start_col=start_col,
        end_line=loc.start_line,
        end_col=start_col + len(keyword),
        source_path=loc.source_path,
    )


def _location_for_span(uri: str, source_path: Optional[str], span: Span) -> lsp.Location:
    target_uri = uri
    if span.source_path:
        span_path = Path(span.source_path).resolve()
        if source_path is None or span_path != Path(source_path).resolve():
            target_uri = span_path.as_uri()
    return lsp.Location(uri=target_uri, range=_span_to_range(span))


def _is_identifier(s: str) -> bool:

    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s))
