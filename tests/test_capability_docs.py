from __future__ import annotations

from pathlib import Path
import re

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
import pytest

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.lang import compile_program, parse_program


@pytest.fixture
def examples():
    path = (
        Path(__file__).resolve().parents[1] / "kedi-docs/docs/agentic-engineering/capabilities.md"
    )
    if not path.is_file():
        pytest.skip("documentation submodule is not checked out")
    blocks = re.findall(r"^(`{3,})kedi\n(.*?)^\1$", path.read_text(), re.M | re.S)
    assert len(blocks) == 6
    return [body for _, body in blocks]


@pytest.mark.parametrize("reset", [False, True])
def test_documented_composition_and_reset_templates_run_unchanged(examples, reset):
    instructions = []
    values = {
        "readiness": "pending",
        "reason": "staging verification is still running",
        "status": "pending",
        "explanation": "verification must finish first",
    }

    def respond(messages, info):
        instructions.append(info.instructions)
        output = info.output_tools[0]
        fields = output.parameters_json_schema["properties"]
        return ModelResponse(
            [ToolCallPart(output.name, {name: values[name] for name in fields}, "output")]
        )

    source = examples[0]
    if reset:
        definitions = source.split("\n> use: release_reviewer", 1)[0]
        source = definitions + "\n" + examples[1]
    runtime = compile_program(
        parse_program(source), adapter=PydanticAdapter(FunctionModel(respond))
    )
    try:
        result = runtime.run_main()
        if not reset:
            assert result == {"readiness": values["readiness"], "reason": values["reason"]}
    finally:
        runtime.artifacts.close()
    if reset:
        assert len(instructions) == 2
        assert all("Use the supplied evidence" not in text for text in instructions)
        assert sum("Keep the explanation" in text for text in instructions) == 1
    else:
        assert len(instructions) == 1
        assert "Use the supplied evidence" in instructions[0]
        assert "Keep the explanation" in instructions[0]


def test_documented_harness_component_reads_a_real_file(examples, tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def divide(a, b):\n    return a / b\n")
    requests = []

    def respond(messages, info):
        requests.append(messages)
        assert "Review code without changing files." in info.instructions
        if len(requests) == 1:
            assert {tool.name for tool in info.function_tools} >= {"read_source"}
            return ModelResponse([ToolCallPart("read_source", {"path": str(path)}, "read")])
        assert "return a / b" in str(messages)
        return ModelResponse(
            [
                ToolCallPart(
                    info.output_tools[0].name, {"finding": "Zero divisor is not handled."}, "output"
                )
            ]
        )

    runtime = compile_program(
        parse_program(examples[2]), adapter=PydanticAdapter(FunctionModel(respond))
    )
    try:
        runtime.run_main()
        assert (
            runtime._invoke_procedure(runtime._get_proc("review_file"), [str(path)])
            == "Zero divisor is not handled."
        )
    finally:
        runtime.artifacts.close()
    assert len(requests) == 2


def test_documented_deferred_bundle_loads_through_native_loader(examples):
    requests = []

    def respond(messages, info):
        requests.append(info)
        if len(requests) == 1:
            assert "read_source" not in {tool.name for tool in info.function_tools}
            identity = re.search(
                r"kedi_source_reader_on_demand_[a-f0-9]{12}", info.instructions
            ).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        assert "read_source" in {tool.name for tool in info.function_tools}
        return ModelResponse([TextPart("loaded")])

    definitions = examples[2].split("\n> instructions: Review code", 1)[0]
    source = definitions + "\n" + examples[3] + "\n[reply] << Load the source reader.\n= <reply>\n"
    runtime = compile_program(
        parse_program(source), adapter=PydanticAdapter(FunctionModel(respond))
    )
    try:
        assert runtime.run_main() == "loaded"
    finally:
        runtime.artifacts.close()
    assert len(requests) == 2


@pytest.mark.parametrize("index", [4, 5])
def test_documented_nested_and_subagent_bundles_run_without_eager_activation(examples, index):
    def respond(messages, info):
        assert "load_capability" in {tool.name for tool in info.function_tools}
        assert "read_source" not in {tool.name for tool in info.function_tools}
        assert "delegate_task" not in {tool.name for tool in info.function_tools}
        assert "kedi_implementation_evidence_" not in info.instructions
        return ModelResponse([TextPart("available")])

    definitions = examples[2].split("\n> instructions: Review code", 1)[0]
    runtime = compile_program(
        parse_program(
            definitions + "\n" + examples[index] + "\n[reply] << Inspect capabilities.\n= <reply>\n"
        ),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    try:
        assert runtime.run_main() == "available"
    finally:
        runtime.artifacts.close()
