"""Service for loading programs with optimized prompts injected."""

from __future__ import annotations

import os
from pathlib import Path

from ...lang import Program, parse_program
from .prompt_loader import OptimizedPromptLoader


def load_program_with_optimized_prompts(
    source_path: Path,
    *,
    debug: bool = False,
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
        debug: If True, print debug information about prompt injection

    Returns:
        Parsed Program object with optimized prompts injected (if available)
    """
    # Read source file
    src = source_path.read_text(encoding="utf-8")

    # Merge companion .val.kedi file if it exists (for validation/tests)
    val_path = source_path.with_suffix(".val.kedi")
    if val_path.exists():
        src = src + "\n\n" + val_path.read_text(encoding="utf-8")

    # Merge cache file if it exists (for AI-generated procedures)
    cache_path = source_path.with_suffix(".cache.kedi")
    if cache_path.exists():
        src = src + "\n\n" + cache_path.read_text(encoding="utf-8")

    # Parse program
    program = parse_program(src)

    # Load and inject optimized prompts if available
    prompt_loader = OptimizedPromptLoader(source_path)

    if debug:
        has_prompts = prompt_loader.has_optimized_prompts()
        print(f"[DEBUG] Checking for optimized prompts: {has_prompts}", file=os.sys.stderr)
        if has_prompts:
            print(
                f"[DEBUG] Optimized path: {prompt_loader.optimized_path}",
                file=os.sys.stderr,
            )

    if prompt_loader.has_optimized_prompts():
        if debug:
            # Show prompts before injection
            from ..utils import span_to_instruction

            for proc in program.procedures:
                if proc.optimize_spans:
                    for span in proc.optimize_spans:
                        original = span_to_instruction(span)
                        print(
                            f"[DEBUG] BEFORE injection - {proc.name}::{span.name}: "
                            f"{original[:80]}...",
                            file=os.sys.stderr,
                        )

        prompt_loader.inject_optimized_prompts(program)

        if debug:
            # Verify injection happened
            from ..utils import span_to_instruction

            for proc in program.procedures:
                if proc.optimize_spans:
                    for span in proc.optimize_spans:
                        injected = span_to_instruction(span)
                        print(
                            f"[DEBUG] AFTER injection - {proc.name}::{span.name}: "
                            f"{injected[:80]}...",
                            file=os.sys.stderr,
                        )

    return program
