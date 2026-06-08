"""Service for loading optimized prompts from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from ...lang.ast import Program, TemplateLine
from ..utils import normalize_optimized_prompt_prefix


class OptimizedPromptLoader:
    """Service for loading optimized prompts from .optimized.json files.

    Handles loading and injecting optimized prompts into procedure definitions.
    """

    def __init__(self, source_path: Path):
        """Initialize prompt loader.

        Args:
            source_path: Path to the source .kedi file (can be relative or absolute)
        """
        # Always resolve to absolute path to ensure we look in the correct directory
        # regardless of current working directory
        self.source_path = Path(source_path).resolve()

        # Build optimized path in the same directory as source file
        # Format: source.kedi -> source.kedi.optimized.json
        self.optimized_path = self.source_path.with_suffix(
            self.source_path.suffix + ".optimized.json"
        )

        # Verify optimized path is in same directory as source
        if self.source_path.parent != self.optimized_path.parent:
            raise RuntimeError(
                f"Path resolution error: source parent ({self.source_path.parent}) "
                f"!= optimized parent ({self.optimized_path.parent})"
            )

    def load_optimized_prompts(self) -> dict[str, dict[str, str]]:
        """Load optimized prompts from JSON file.

        Returns:
            Dictionary mapping procedure_name -> span_name -> optimized_prompt
            Returns empty dict if file doesn't exist or is invalid
        """
        if not self.optimized_path.exists():
            return {}

        try:
            with open(self.optimized_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def inject_optimized_prompts(self, program: Program) -> None:
        """Inject optimized prompts into program's optimize spans.

        Modifies the program's ProcedureDef objects in-place, replacing
        template_lines in OptimizeSpan objects with optimized versions.

        Args:
            program: Program to modify
        """
        optimized_prompts = self.load_optimized_prompts()

        if not optimized_prompts:
            return

        for proc in program.procedures:
            if proc.name not in optimized_prompts:
                continue

            proc_optimized = optimized_prompts[proc.name]

            for span in proc.optimize_spans:
                if span.name not in proc_optimized:
                    continue

                optimized_prompt = normalize_optimized_prompt_prefix(
                    span, proc_optimized[span.name]
                )
                # NEW APPROACH: Store optimized prompt as prefix, keep original seed prompt intact
                # This way:
                # 1. GEPA can output plain text without needing correct type names
                # 2. Original prompt with proper type annotations is preserved
                # 3. At evaluation time, optimized text is prepended to seed prompt

                # Store the optimized prompt text in the first template line's prefix.
                # Keep all original template lines so typed output placeholders remain
                # available for DSPy signature extraction and execution.
                if span.template_lines:
                    # Modify the existing TemplateLine object in place
                    # This automatically updates proc.body since they reference the same object
                    span.template_lines[0].optimized_prompt_prefix = optimized_prompt or None
                else:
                    # This shouldn't happen - optimize spans should have template lines
                    # But if it does, create a new one with empty segments and the prefix
                    span.template_lines = [
                        TemplateLine(
                            segments=[],
                            optimized_prompt_prefix=optimized_prompt or None,
                        )
                    ]

    def has_optimized_prompts(self) -> bool:
        """Check if optimized prompts file exists.

        Returns:
            True if .optimized.json file exists and is readable
        """
        return self.optimized_path.exists()
