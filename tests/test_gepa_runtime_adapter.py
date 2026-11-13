"""GEPA predictions and non-optimized model calls use independent backends."""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ImageUrl, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from kedi import app
from kedi.agent_adapter import PydanticAdapter
from tests.dspy_mocks import MockLM, Recorder, install_mock_dspy


def test_gepa_keeps_media_calls_on_the_selected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    install_mock_dspy(monkeypatch)
    monkeypatch.setattr("kedi.optimization.services.dspy_config.ThinkingStrippingLM", MockLM)
    Recorder.reset()
    requests = []

    def respond(messages, info):
        for message in messages:
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    requests.extend(item for item in part.content if isinstance(item, ImageUrl))
        output = info.output_tools[0]
        assert tuple(output.parameters_json_schema["properties"]) == ("animal",)
        return ModelResponse(parts=[ToolCallPart(output.name, {"animal": "cat"})])

    runtime_adapter = PydanticAdapter(FunctionModel(respond))
    selected = []

    def build_runtime(adapter, model):
        selected.append((adapter, model))
        return runtime_adapter

    monkeypatch.setattr(app, "_build_runtime_adapter", build_runtime)
    source = tmp_path / "media.kedi"
    source.write_text(
        "```\n"
        "from pydantic_ai.messages import ImageUrl\n"
        "```\n\n"
        "@photo() -> ImageUrl:\n"
        '  = `ImageUrl("https://example.com/cat.png")`\n\n'
        "@solve(seed: str) -> int:\n"
        "  >> The animal in <<photo()>> is [animal: str].\n"
        "  > optimize: number:\n"
        "    >> The value for <seed> is [count: int].\n"
        '  = `count if animal == "cat" else 0`\n\n'
        "@eval: solve:\n"
        "  > data: cases:\n"
        "    = ```\n"
        '    return [("alpha", None), ("beta", None), ("gamma", None)]\n'
        "    ```\n"
        "  > metric: score(cases):\n"
        "    = ```\n"
        "    value = solve(cases)\n"
        '    return (float(value < 0), "OK")\n'
        "    ```\n",
        encoding="utf-8",
    )
    code, message = app.run_optimization_cli(
        str(source),
        optimizer="gepa",
        adapter="pydantic",
        adapter_model="test:media-runtime",
        optimizer_model="mock/optimizer",
        reflection_model="mock/reflection",
        max_metric_calls=6,
        quiet=True,
    )
    assert code == 0, message
    assert selected == [("pydantic", "test:media-runtime")]
    assert requests and all(image.url == "https://example.com/cat.png" for image in requests)
    assert Recorder.calls, "GEPA must execute its native DSPy predictor"
    assert source.with_suffix(".kedi.optimized.json").exists()
    scores = json.loads(source.with_suffix(".kedi.optimized_scores.json").read_text())
    assert scores["solve"] == 1.0
