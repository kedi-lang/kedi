"""Utility functions for optimizer implementations."""

from __future__ import annotations

from typing import cast

from ..lang.ast import (
    REFERENCE_SEGMENTS,
    CallSegment,
    InputSegment,
    OptimizeSpan,
    OutputSegment,
    PythonExprNativeSegment,
    PythonExprSegment,
    TemplateLine,
    TextSegment,
)
from ..lang.compiler.environment import KediRuntime
from ..lang.compiler.evaluation import (
    evaluate_call,
    evaluate_python_inline,
    lookup_value,
)

__all__ = [
    "extract_input_fields",
    "extract_output_fields",
    "normalize_metric_result",
    "normalize_optimized_prompt_prefix",
    "render_instruction_lines",
    "span_to_instruction",
    "template_line_to_string",
]


def normalize_metric_result(val: object) -> tuple[float, str | None]:
    """Normalize metric return values to ``(score, feedback)``."""
    if isinstance(val, bool):
        return (1.0 if val else 0.0, None)
    if isinstance(val, tuple) and len(val) == 2:
        score = float(cast(float | int | str, val[0]))
        feedback = None if val[1] is None else str(val[1])
        return (score, feedback)
    if isinstance(val, tuple):
        raise TypeError("Metric tuple result must be (score, feedback)")
    return (float(cast(float | int | str, val)), None)


def template_line_to_string(template_line: TemplateLine) -> str:
    """Convert TemplateLine to prompt string.

    Args:
        template_line: The template line to convert

    Returns:
        String representation with placeholders:
        - TextSegment -> literal text
        - InputSegment -> <name>
        - OutputSegment -> [name] or [name: type]
    """
    parts = []
    for segment in template_line.segments:
        if isinstance(segment, TextSegment):
            parts.append(segment.text)
        elif isinstance(segment, REFERENCE_SEGMENTS):
            parts.append(f"<<{segment.name}>>")
        elif isinstance(segment, InputSegment):
            parts.append(f"<{segment.name}>")
        elif isinstance(segment, OutputSegment):
            # Include type annotation in output placeholder if present
            if segment.type_annotation:
                parts.append(f"[{segment.name}: {segment.type_annotation}]")
            else:
                parts.append(f"[{segment.name}]")
        elif isinstance(segment, (PythonExprSegment, PythonExprNativeSegment)):
            # Preserve visibility of dynamic hints in instructions; do not evaluate here
            # Represent as inline code within angle brackets
            code = getattr(segment, "code", "")
            if code:
                parts.append(f"<`{code}`>")
            else:
                parts.append("<expr>")
    return "".join(parts)


def span_to_instruction(
    span: OptimizeSpan,
    *,
    include_optimized_prefix: bool = True,
) -> str:
    """Convert OptimizeSpan to instruction string.

    If ``include_optimized_prefix`` is true and the first template line has
    ``optimized_prompt_prefix`` (from a prior run's ``.optimized.json``), it is
    prepended so DSPy/GEPA seed from the last saved prompt.

    Args:
        span: The optimize span to convert
        include_optimized_prefix: Whether to prepend saved optimized prefix text

    Returns:
        Newline-joined string of all template lines in the span
    """
    lines = [template_line_to_string(tl) for tl in span.template_lines]
    base = "\n".join(lines)
    if include_optimized_prefix and span.template_lines:
        prefix = getattr(span.template_lines[0], "optimized_prompt_prefix", None)
        if prefix and str(prefix).strip():
            p = str(prefix).rstrip()
            return f"{p}\n{base}" if base.strip() else p
    return base


def normalize_optimized_prompt_prefix(span: OptimizeSpan, instructions: str | None) -> str:
    """Convert a full DSPy instruction string into the stored prefix form.

    Kedi persists only the optimized prefix and keeps the original span lines as the
    canonical typed seed prompt. GEPA/DSPy often surface a full instruction string
    that ends with that canonical seed prompt. Strip any trailing copies of the seed
    so reloading the saved prompt does not duplicate the base instructions.
    """
    if instructions is None:
        return ""

    current = str(instructions).rstrip()
    if not current:
        return ""

    base = span_to_instruction(span, include_optimized_prefix=False).rstrip()
    if not base:
        return current

    while True:
        if current == base:
            return ""
        if not current.endswith(base):
            return current

        prefix = current[: -len(base)]
        if prefix and not prefix[-1].isspace():
            return current
        current = prefix.rstrip()


def render_instruction_lines(
    span: OptimizeSpan,
    rt: KediRuntime,
    kwargs: dict[str, object],
) -> str:
    """Render optimize span template lines into a concrete instruction string.

    Substitutes inputs (<name>), evaluates inline python (both `<`expr`>` and native),
    and evaluates calls. Output placeholders are preserved as textual markers
    (e.g., `[name: type]`) so the instruction still guides the LLM to produce fields.

    Args:
        span: Optimize span whose template lines will be rendered
        rt: Active KediRuntime for evaluation
        kwargs: Values to use for substitutions and python evaluation

    Returns:
        Concrete instruction string with inputs and expressions resolved while
        preserving output placeholders.
    """
    rendered_lines: list[str] = []
    for template_line in span.template_lines:
        parts: list[str] = []
        for segment in template_line.segments:
            if isinstance(segment, TextSegment):
                parts.append(segment.text)
            elif isinstance(segment, REFERENCE_SEGMENTS):
                # A reference reaches the model as an attachment, never as text.
                parts.append(f"<<{segment.name}>>")
            elif isinstance(segment, InputSegment):
                try:
                    value = lookup_value(rt, segment.name, kwargs)
                    parts.append(str(value))
                except Exception:
                    # Leave placeholder if value not available in this context
                    parts.append(f"<{segment.name}>")
            elif isinstance(segment, CallSegment):
                try:
                    value = evaluate_call(rt, segment, kwargs)
                    parts.append(str(value))
                except Exception:
                    # Preserve call placeholder if evaluation isn't possible here
                    parts.append(f"<{segment.name}()>")
            elif isinstance(segment, OutputSegment):
                if segment.type_annotation:
                    parts.append(f"[{segment.name}: {segment.type_annotation}]")
                else:
                    parts.append(f"[{segment.name}]")
            elif isinstance(segment, (PythonExprSegment, PythonExprNativeSegment)):
                try:
                    value = evaluate_python_inline(rt, getattr(segment, "code", ""), kwargs)
                    parts.append(str(value))
                except Exception:
                    code = getattr(segment, "code", "")
                    parts.append(f"<`{code}`>")
        rendered_lines.append("".join(parts))
    return "\n".join(rendered_lines)


def extract_output_fields(span: OptimizeSpan) -> list[tuple[str, str | None]]:
    """Extract output field names and types from an OptimizeSpan.

    Args:
        span: The optimize span to analyze

    Returns:
        List of (field_name, type_annotation) tuples, preserving order
        type_annotation is None if no type specified
    """
    outputs: list[tuple[str, str | None]] = []
    seen_names: set[str] = set()

    for template_line in span.template_lines:
        for segment in template_line.segments:
            if isinstance(segment, OutputSegment):
                # Only add if not already seen (avoid duplicates)
                if segment.name not in seen_names:
                    outputs.append((segment.name, segment.type_annotation))
                    seen_names.add(segment.name)

    return outputs


def extract_input_fields(span: OptimizeSpan) -> list[str]:
    """Extract input field names from an OptimizeSpan.

    Args:
        span: The optimize span to analyze

    Returns:
        List of input field names (e.g., from <name> substitutions)
    """
    inputs: list[str] = []
    seen_names: set[str] = set()

    for template_line in span.template_lines:
        for segment in template_line.segments:
            if isinstance(segment, InputSegment):
                if segment.name not in seen_names:
                    inputs.append(segment.name)
                    seen_names.add(segment.name)

    return inputs
