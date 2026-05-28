"""Service for writing optimization results to files."""

from __future__ import annotations

import json
from pathlib import Path


class OptimizationResultWriter:
    """Service for writing optimized prompts and scores to files.

    Handles writing optimized prompts to .optimized.json files and
    training scores to .optimized_scores.json files.
    """

    def __init__(self, source_path: Path):
        """Initialize result writer.

        Args:
            source_path: Path to source .kedi file
        """
        self.source_path = Path(source_path).resolve()

    def write_optimized_prompts(
        self,
        optimized_prompts: dict[str, dict[str, str]],
    ) -> Path:
        """Write optimized prompts to .optimized.json file.

        Args:
            optimized_prompts: Dict mapping procedure_name -> span_name -> prompt

        Returns:
            Path to written file
        """
        # Create filename: source.kedi -> source.kedi.optimized.json
        optimized_path = self.source_path.with_suffix(self.source_path.suffix + ".optimized.json")

        # Serialize to JSON with indentation
        json_str = json.dumps(optimized_prompts, indent=2, ensure_ascii=False)

        # Write to file
        optimized_path.write_text(json_str, encoding="utf-8")

        return optimized_path

    def write_training_scores(
        self,
        train_scores: dict[str, float],
    ) -> Path:
        """Write training scores to .optimized_scores.json file.

        Args:
            train_scores: Dict mapping procedure_name -> score

        Returns:
            Path to written file
        """
        # Create filename: source.kedi -> source.kedi.optimized_scores.json
        scores_path = self.source_path.with_suffix(
            self.source_path.suffix + ".optimized_scores.json"
        )

        # Load existing scores if file exists
        existing_scores: dict[str, float] = {}
        if scores_path.exists():
            try:
                scores_content = scores_path.read_text(encoding="utf-8")
                existing_scores = json.loads(scores_content)
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, start fresh
                existing_scores = {}

        # Merge new scores with existing
        existing_scores.update(train_scores)

        # Serialize to JSON with indentation
        json_str = json.dumps(existing_scores, indent=2, ensure_ascii=False)

        # Write to file
        scores_path.write_text(json_str, encoding="utf-8")

        return scores_path

    def format_prompts_for_display(
        self,
        optimized_prompts: dict[str, dict[str, str]],
        optimized_path: Path,
    ) -> str:
        """Format optimized prompts for CLI display.

        Args:
            optimized_prompts: Dict mapping procedure_name -> span_name -> prompt
            optimized_path: Path to optimized file

        Returns:
            Formatted string for display
        """
        lines = [
            f"Optimization complete. Wrote optimized prompts to {optimized_path.name}",
            "",
        ]

        for proc_name, spans in optimized_prompts.items():
            lines.append(f"Optimized prompts for '{proc_name}':")
            for span_name, prompt in spans.items():
                lines.append(f"  [{span_name}]:")
                # Display prompt with indentation
                for prompt_line in prompt.split("\n"):
                    lines.append(f"    {prompt_line}")
                lines.append("")

        return "\n".join(lines)

    def delete_existing_optimized_files(self) -> None:
        """Delete existing optimized files to ensure fresh optimization."""
        optimized_path = self.source_path.with_suffix(self.source_path.suffix + ".optimized.json")
        scores_path = self.source_path.with_suffix(
            self.source_path.suffix + ".optimized_scores.json"
        )

        if optimized_path.exists():
            optimized_path.unlink()
        if scores_path.exists():
            scores_path.unlink()
