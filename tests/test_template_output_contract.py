"""Template output tools describe capture values without rewriting returned data."""

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program


def test_template_tool_requests_replacements_and_keeps_intentional_sentences():
    descriptions = []

    def respond(messages, info):
        descriptions.append(info.output_tools[0].description)
        return ModelResponse(
            [ToolCallPart(info.output_tools[0].name, {"message": "The message is a sentence."})]
        )

    runtime = compile_program(
        parse_program(">> The message is [message: str].\n= `message`\n"),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    try:
        assert runtime.run_main() == "The message is a sentence."
    finally:
        runtime.artifacts.close()
    assert "replacement values" in descriptions[0]
    assert "Preserve full sentences" in descriptions[0]


def test_direct_python_produce_keeps_native_output_description():
    class Result(BaseModel):
        """An ordinary application result."""

        message: str

    def respond(messages, info):
        assert "replacement values" not in (info.output_tools[0].description or "")
        return ModelResponse([ToolCallPart(info.output_tools[0].name, {"message": "ready"})])

    result = PydanticAdapter(FunctionModel(respond)).produce_sync(
        "Write a report.", output_type=Result
    )
    assert result.message == "ready"


def test_decision_template_does_not_receive_completion_output_guidance():
    class DecisionModel(FunctionModel):
        kedi_prompt_mode = "decision"

    def respond(messages, info):
        assert "TemplateInstructions" not in (info.instructions or "")
        assert "replacement values" not in (info.output_tools[0].description or "")
        return ModelResponse([ToolCallPart(info.output_tools[0].name, {"approved": True})])

    runtime = compile_program(
        parse_program(
            ">> Two plus two equals four, so approval is [approved: bool].\n= `approved`\n"
        ),
        adapter=PydanticAdapter(DecisionModel(respond)),
    )
    try:
        assert runtime.run_main() is True
    finally:
        runtime.artifacts.close()
