"""Docstring parsing helpers for Kedi procedures."""

from __future__ import annotations

from dataclasses import dataclass, field

from docstring_parser import parse as parse_docstring


@dataclass(frozen=True)
class ParsedProcedureDocstring:
    description: str | None = None
    parameter_descriptions: dict[str, str] = field(default_factory=dict)
    return_description: str | None = None


def parse_procedure_docstring(docstring: str | None) -> ParsedProcedureDocstring:
    """Parse a procedure docstring into structured descriptions.

    If parsing fails, the raw docstring is preserved as the procedure description
    and parameter / return descriptions are omitted.
    """
    if not docstring:
        return ParsedProcedureDocstring()

    stripped = docstring.strip()
    if not stripped:
        return ParsedProcedureDocstring()

    try:
        parsed = parse_docstring(stripped)
    except Exception:
        return ParsedProcedureDocstring(description=stripped)

    description = _combine_description(
        parsed.short_description,
        parsed.long_description,
    )
    parameter_descriptions = {
        param.arg_name: param.description.strip()
        for param in parsed.params
        if param.arg_name and param.description and param.description.strip()
    }
    return_description = (
        parsed.returns.description.strip()
        if parsed.returns and parsed.returns.description
        else None
    )
    return ParsedProcedureDocstring(
        description=description,
        parameter_descriptions=parameter_descriptions,
        return_description=return_description,
    )


def _combine_description(
    short_description: str | None,
    long_description: str | None,
) -> str | None:
    parts = []
    if short_description and short_description.strip():
        parts.append(short_description.strip())
    if long_description and long_description.strip():
        parts.append(long_description.strip())
    if not parts:
        return None
    return "\n".join(parts)
