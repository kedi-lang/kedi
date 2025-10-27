"""Kedi DSL Parser Package.

This package provides clean, modular parsing of Kedi source code.

Public API:
    parse_program(source) -> Program
"""

from __future__ import annotations

from .program_parser import parse_program

__all__ = ["parse_program"]
