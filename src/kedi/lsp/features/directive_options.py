"""Value completion using the same field contracts as the DSL evaluator."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lsprotocol import types as lsp

from kedi.directive_values import DIRECTIVE_RULES, FIELD_RULES, ValueRule, field_rule
from kedi.lang.parser.directive_context import selected_adapter
from kedi.lang.parser.parser import parse
from kedi.lsp.coordinates import utf16_length, utf16_to_index


@dataclass(frozen=True)
class ValueContext:
    scope: str
    name: str
    start: int
    adapter: str | None


def value_context(source: str, pos: lsp.Position) -> ValueContext | None:
    lines = source.split("\n")
    if pos.line >= len(lines):
        return None
    line = lines[pos.line]
    before = line[: utf16_to_index(line, pos.character)]
    directive = re.fullmatch(r"\s*>\s*(\w+):[ \t]*(.*)", before)
    setting = re.fullmatch(r"\s*(\w+):[ \t]*(.*)", before)
    match = directive or setting
    requirement = False
    if match is None:
        if re.fullmatch(r"\s*[A-Za-z_]*", before):
            depth = len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())
            for previous in reversed(lines[: pos.line]):
                if not previous.strip() or previous.lstrip().startswith("#"):
                    continue
                expanded = previous.expandtabs(4)
                if len(expanded) - len(expanded.lstrip()) < depth:
                    if re.fullmatch(r"\s*>\s*requires:\s*", previous):
                        requirement = True
                    break
        if not requirement:
            return None
    name = "requires" if match is None else match[1]
    start = len(before) - len(before.lstrip()) if match is None else match.start(2)
    encoded = (source + "\n").encode("utf-8")
    offset = len(
        ("\n".join(lines[: pos.line]) + ("\n" if pos.line else "") + before).encode("utf-8")
    )
    node = parse(encoded).root_node.descendant_for_byte_range(
        max(0, offset - 1), max(0, offset - 1)
    )
    assert node is not None
    adapter = selected_adapter(encoded, node)
    ancestor = node
    scope = ""
    while ancestor is not None:
        if ancestor.type in {"line_comment", "block_comment"}:
            return ValueContext("comment", "", start, None)
        candidate = ancestor.type.removesuffix("_body")
        if candidate in FIELD_RULES:
            scope = candidate
            break
        if ancestor.type == "agent_section":
            section = ancestor.child_by_field_name("name")
            assert section is not None
            if encoded[section.start_byte : section.end_byte] == b"auth":
                scope = "auth"
                break
        ancestor = ancestor.parent
    if directive is not None or requirement:
        if name not in DIRECTIVE_RULES and name not in {"adapter", "agent", "requires"}:
            return None
        return ValueContext("directive", name, start, adapter)
    if scope not in FIELD_RULES:
        return None
    return ValueContext(scope, name, start, adapter)


def option_completions(
    source: str,
    pos: lsp.Position,
    *,
    explicit: bool = False,
) -> list[lsp.CompletionItem] | None:
    context = value_context(source, pos)
    if context is None:
        return None
    line = source.split("\n")[pos.line]
    cursor = utf16_to_index(line, pos.character)
    prefix = line[context.start : cursor]
    if not re.fullmatch(r"[\w-]*", prefix) or (not prefix and not explicit):
        return []
    rule = (
        DIRECTIVE_RULES.get(context.name)
        if context.scope == "directive"
        else field_rule(context.scope, context.name, adapter=context.adapter)
    )
    if context.scope == "directive" and context.name in {"adapter", "agent"}:
        from kedi.agent_adapter.registry import adapter_factory, known_adapter_shortnames

        kind = "agent-framework" if context.name == "adapter" else "agent-harness"
        options = tuple(
            name for name in known_adapter_shortnames() if adapter_factory(name).kind == kind
        )
        rule = ValueRule("Registered agent integration", options)
    if context.scope == "directive" and context.name == "requires":
        from kedi.agent_adapter.meta import adapter_capability_names

        rule = ValueRule("Required adapter capability", tuple(adapter_capability_names()))
    if rule is None:
        return []
    suffix = re.match(r"[\w-]*", line[cursor:])
    end = cursor + (len(suffix[0]) if suffix else 0)
    return [
        lsp.CompletionItem(
            label=option,
            kind=lsp.CompletionItemKind.EnumMember,
            detail=rule.description,
            text_edit=lsp.TextEdit(
                range=lsp.Range(
                    start=lsp.Position(
                        line=pos.line, character=utf16_length(line[: context.start])
                    ),
                    end=lsp.Position(line=pos.line, character=utf16_length(line[:end])),
                ),
                new_text=option,
            ),
        )
        for option in rule.completion_values
        if option.startswith(prefix)
    ]
