from __future__ import annotations

from typing import Any

import lsprotocol.types as lsp
import pytest
from pydantic import BaseModel, ValidationError

import kedi.lsp.features.capabilities as lsp_capabilities_module
import kedi.lsp.features.capability_state as capability_state_module
from kedi import force
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_profile import AgentProfile, block_agent_profile_scope
from kedi.core.runtime import KediRuntime
from kedi.engine import ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import RawInvokeStmt
from kedi.lsp.features.diagnostics import compute_diagnostics
from kedi.lsp.python_virtual.document import compute_python_virtual_document
from tests.lsp_helpers import hover_at, hover_value
from tests.mock_adapter import MockAdapter


class TypedAdapter(MockAdapter):
    """Returns a fixed value for typed produce calls and records what it was asked for."""

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value
        self.typed_calls: list[tuple[str, Any]] = []
        self.invoke_prompts: list[str] = []

    def produce_sync(self, *, template: str, output_schema: Any = None, **kwargs: Any) -> Any:
        output_type = kwargs.pop("output_type", None)
        if output_type is None:
            return super().produce_sync(template=template, output_schema=output_schema, **kwargs)
        self.typed_calls.append((template, output_type))
        self.instruction_calls.append(kwargs.get("instructions"))
        return self.value

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        self.invoke_prompts.append(prompt)
        return f"text:{prompt}"


def _run(src: str, adapter: MockAdapter, **kwargs: Any) -> Any:
    return compile_program(parse_program(src), adapter=adapter, **kwargs).run_main()


def test_typed_raw_invoke_parses_without_diagnostics() -> None:
    program = parse_program("[count: int] << How many?\n")

    stmt = program.toplevel_body[0]
    assert isinstance(stmt, RawInvokeStmt)
    assert stmt.type_annotation == "int"
    assert program.parser_warnings == []


def test_typed_raw_invoke_sends_the_prompt_unchanged_with_the_type() -> None:
    adapter = TypedAdapter("3")

    result = _run("[topic] = cats\n[count: int] << How many <topic>?\n= `count + 1`\n", adapter)

    assert result == 4
    assert adapter.typed_calls == [("How many cats?", int)]
    assert adapter.invoke_prompts == []
    assert "TemplateInstructions" not in (adapter.instruction_calls[-1] or "")


def test_typed_raw_invoke_validates_custom_types() -> None:
    adapter = TypedAdapter({"title": "Risky", "score": 2})

    result = _run(
        "~Report(title, score: int)\n[report: Report] << Review it.\n= `report.score * 10`\n",
        adapter,
    )

    assert result == 20
    ((_, output_type),) = adapter.typed_calls
    assert issubclass(output_type, BaseModel)


def test_typed_raw_invoke_rejects_an_invalid_value() -> None:
    with pytest.raises((KediExecutionError, ValidationError), match="int"):
        _run("[count: int] << How many?\n= <count>\n", TypedAdapter("many"))


def test_typed_raw_invoke_binding_keeps_its_type_contract() -> None:
    adapter = TypedAdapter(3)

    assert _run("[count: int] << How many?\n[count] := `count + 1`\n= <count>\n", adapter) == "4"
    with pytest.raises(KediExecutionError):
        _run('[count: int] << How many?\n[count] := `"many"`\n= <count>\n', adapter)


def test_str_raw_invoke_capture_keeps_the_text_path() -> None:
    adapter = TypedAdapter("unused")

    assert _run("[answer: `str`] << Hello\n= <answer>\n", adapter) == "text:Hello"
    assert adapter.typed_calls == []


@pytest.mark.parametrize("annotation", ["str", "`str`"])
def test_str_raw_invoke_capture_keeps_its_type_contract(annotation: str) -> None:
    adapter = TypedAdapter("unused")

    assert _run(f"[a: {annotation}] << Say hi.\n[a] := `'hello'`\n= <a>\n", adapter) == "hello"
    with pytest.raises(KediExecutionError, match="annotated type"):
        _run(f"[a: {annotation}] << Say hi.\n[a] := `5`\n= `repr(a)`\n", adapter)
    assert adapter.typed_calls == []


def test_typed_raw_invoke_runs_on_the_parallel_engine() -> None:
    adapter = TypedAdapter(["a", "b"])
    rt = compile_program(
        parse_program("[tags: list[str]] << Tags?\n= `', '.join(tags)`\n"),
        adapter=adapter,
        engine=ThreadEngine(max_workers=1),
    )
    try:
        assert rt.run_main() == "a, b"
    finally:
        rt._engine.shutdown()
    assert adapter.typed_calls == [("Tags?", list[str])]


def test_typed_raw_invoke_without_artifact_policy() -> None:
    rt = KediRuntime(adapter=TypedAdapter("5"))
    token = rt._current_env.set({})
    try:
        with block_agent_profile_scope(AgentProfile()):
            value = rt.invoke(["How many?"], capture="count", output_type=int)
    finally:
        rt._current_env.reset(token)

    assert force(value) == 5


def test_typed_raw_invoke_is_typed_in_the_virtual_document_and_hover() -> None:
    src = "[count: int] << How many?\n```\nprint(count)\n```\n= <count>\n"

    doc = compute_python_virtual_document(src, "/tmp/typed_raw.kedi")
    assert "count: int = cast(int, _kedi_placeholder())" in doc["text"]

    text = hover_value(hover_at(src, "/tmp/typed_raw.kedi", "count"))
    assert "`count: int`" in text
    assert "structured output" in text


def test_codex_schema_format_diagnostic_applies_to_typed_raw_invoke() -> None:
    src = "> agent: codex\n\n[file: FileUrl] << Which file?\n"

    diags = compute_diagnostics(src, "example.kedi")

    assert [d.code for d in diags] == ["K-ADAPTER-SCHEMA-FORMAT"]
    assert diags[0].range.start.line == 2
    assert diags[0].range.start.character == len("[file: ")


def test_typed_raw_invoke_needs_structured_output_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in (lsp_capabilities_module, capability_state_module):
        monkeypatch.setattr(
            module,
            "adapter_capabilities",
            lambda _name: AdapterCapabilities(supports_structured_output=False),
        )

    typed = compute_diagnostics("> agent: codex\n\n[count: int] << How many?\n", "example.kedi")
    text = compute_diagnostics("> agent: codex\n\n[answer] << Hello\n", "example.kedi")

    assert [d.code for d in typed] == ["K-ADAPTER-CAPABILITY"]
    assert typed[0].severity == lsp.DiagnosticSeverity.Error
    assert "typed raw invoke captures" in typed[0].message
    assert text == []
