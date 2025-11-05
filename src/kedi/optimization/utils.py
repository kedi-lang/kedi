"""Utility functions for optimizer implementations."""

from __future__ import annotations

from ..lang.ast import (
    InputSegment,
    OptimizeSpan,
    OutputSegment,
    TemplateLine,
    TextSegment,
    PythonExprSegment,
    PythonExprNativeSegment,
)


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
            code = getattr(segment, 'code', '')
            if code:
                parts.append(f"<`{code}`>")
            else:
                parts.append("<expr>")
    return "".join(parts)


def span_to_instruction(span: OptimizeSpan) -> str:
    """Convert OptimizeSpan to instruction string.
    
    Args:
        span: The optimize span to convert
        
    Returns:
        Newline-joined string of all template lines in the span
    """
    lines = [template_line_to_string(tl) for tl in span.template_lines]
    return "\n".join(lines)


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
