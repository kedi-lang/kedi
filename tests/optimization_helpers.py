"""Shared helpers for optimization-related tests."""

from __future__ import annotations

from typing import Any

from kedi.lang import parse_program
from kedi.lang.ast import (
    OptimizeSpan,
    OutputSegment,
    ProgramWithValidation,
    TemplateLine,
    TextSegment,
)


def parse_validation_program(src: str) -> ProgramWithValidation:
    prog = parse_program(src)
    assert isinstance(prog, ProgramWithValidation)
    return prog


def make_optimize_span(name: str = "prompt", segments: list[Any] | None = None) -> OptimizeSpan:
    return OptimizeSpan(
        name=name,
        template_lines=[
            TemplateLine(
                segments=(
                    segments
                    if segments is not None
                    else [
                        TextSegment(text="Answer "),
                        OutputSegment(name="answer", type_annotation="str"),
                    ]
                )
            )
        ],
    )
