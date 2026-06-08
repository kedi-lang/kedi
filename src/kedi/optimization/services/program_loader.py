"""Service for loading programs with optimized prompts injected."""

from __future__ import annotations

import sys
from pathlib import Path

from ...lang import Program, parse_program
from ...reporter import CliReporter, NullReporter, Reporter
from ...source_loading import load_kedi_source_bundle
from .prompt_loader import OptimizedPromptLoader


def load_program_with_optimized_prompts(
    source_path: Path,
    *,
    debug: bool = False,
    reporter: Reporter | None = None,
) -> Program:
    """Load and parse a program, injecting optimized prompts if available.

    This is the central function for loading programs that should use
    optimized prompts. It handles:
    1. Reading the source file
    2. Merging .val.kedi if it exists (for validation/tests)
    3. Merging .cache.kedi if it exists (for AI-generated code)
    4. Parsing the program
    5. Injecting optimized prompts from .optimized.json if available

    Args:
        source_path: Path to the .kedi source file
        debug: If True, emit debug information about prompt injection
        reporter: Optional reporter for debug output (defaults to stderr)

    Returns:
        Parsed Program object with optimized prompts injected (if available)
    """
    _reporter = NullReporter() if not debug else (reporter or CliReporter(file=sys.stderr))

    loaded = load_kedi_source_bundle(source_path)
    program = parse_program(
        loaded.text,
        source_path=str(source_path.resolve()),
        source_map=loaded.source_map,
    )

    # Load and inject optimized prompts if available
    prompt_loader = OptimizedPromptLoader(source_path)

    if debug:
        has_prompts = prompt_loader.has_optimized_prompts()
        _reporter.emit(f"[DEBUG] Checking for optimized prompts: {has_prompts}")
        if has_prompts:
            _reporter.emit(f"[DEBUG] Optimized path: {prompt_loader.optimized_path}")

    if prompt_loader.has_optimized_prompts():
        if debug:
            from ..utils import span_to_instruction

            for proc in program.procedures:
                if proc.optimize_spans:
                    for span in proc.optimize_spans:
                        original = span_to_instruction(span)
                        _reporter.emit(
                            f"[DEBUG] BEFORE injection - {proc.name}::{span.name}: "
                            f"{original[:80]}..."
                        )

        prompt_loader.inject_optimized_prompts(program)

        if debug:
            from ..utils import span_to_instruction

            for proc in program.procedures:
                if proc.optimize_spans:
                    for span in proc.optimize_spans:
                        injected = span_to_instruction(span)
                        _reporter.emit(
                            f"[DEBUG] AFTER injection - {proc.name}::{span.name}: "
                            f"{injected[:80]}..."
                        )

    return program
