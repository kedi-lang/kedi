from __future__ import annotations

from dataclasses import dataclass

from .models import PromptDraft

_TEMPLATE_FILL_INSTRUCTION = (
    "Fill each unescaped [name] or [name: type] output field with only that field's "
    "replacement value; never return the completed template as a field value. "
    "Escaped square brackets are literal text, not output fields; for example, "
    "\\[ok\\] represents the literal text [ok]. "
    "A replacement value may itself contain delimiters; include them when the template "
    "explicitly requests them. Preserve literal values already present in the template "
    "exactly; do not replace or invent them. The template must remain grammatically "
    "and semantically coherent after substitution."
)
CLAIM_INSTRUCTION = (
    "Determine whether the following claim is true. Consider evidence in this priority order: "
    "(1) the current context, (2) relevant available tools when they are needed, and "
    "(3) reliable knowledge you already possess when the context and tools do not establish "
    "the answer. Before answering, use a relevant tool when it can authoritatively resolve a "
    "time-sensitive, private, or otherwise unestablished claim; do not call tools for stable "
    "facts you already know. Prefer current context and tool results over prior knowledge when "
    "they conflict. Return true if the claim is established; return false if it is contradicted "
    "or you do not know. Missing context alone is not a reason to return false."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RawPromptBuilder:
    content: str

    def build(self) -> PromptDraft:
        return PromptDraft(content=self.content, kind="raw")


@dataclass(frozen=True, slots=True, kw_only=True)
class KediTemplatePromptBuilder:
    template: str
    output_fields: tuple[str, ...]

    def build(self) -> PromptDraft:
        fields = ", ".join(self.output_fields)
        content = (
            f"{_TEMPLATE_FILL_INSTRUCTION}\n"
            "Fields to be substituted into the template string for final construction: "
            f"({fields}).\n\nThe template string:\n{self.template}"
        )
        return PromptDraft(content=content, kind="template")


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimPromptBuilder:
    claim: str

    def build(self) -> PromptDraft:
        return PromptDraft(
            content=f"Claim: {self.claim}",
            kind="claim",
        )


__all__ = [
    "CLAIM_INSTRUCTION",
    "ClaimPromptBuilder",
    "KediTemplatePromptBuilder",
    "RawPromptBuilder",
]
