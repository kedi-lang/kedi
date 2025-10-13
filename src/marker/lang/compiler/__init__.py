"""Marker DSL Compiler Package.

This package provides clean, modular compilation of Marker programs.

Public API:
    compile_program(prog, adapter=None) -> MarkerRuntime
"""
from __future__ import annotations

from .program_compiler import compile_program
from .evaluation import evaluate_python_inline as _eval_python_inline, execute_python_block as _execute_python_block

__all__ = ["compile_program", "_eval_python_inline", "_execute_python_block"]