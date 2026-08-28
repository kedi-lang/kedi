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
_CLAIM_INSTRUCTION = (
    "Return true only if the following claim can be established as true from the available "
    "context; otherwise return false."
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
            content=f"{_CLAIM_INSTRUCTION}\nClaim: {self.claim}",
            kind="claim",
        )


__all__ = ["ClaimPromptBuilder", "KediTemplatePromptBuilder", "RawPromptBuilder"]
