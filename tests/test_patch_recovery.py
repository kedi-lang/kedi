from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.lang import compile_program, parse_program


def patch_args(step: int) -> dict[str, str]:
    return {
        "operation": "update",
        "file_path": "data.txt",
        "content": "",
        "old_text": "same" if step < 4 else "same\nsame\n",
        "new_text": "fixed\n",
    }


class PatchModel(BaseChatModel):
    step: int = 0
    results: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "patch-rejection-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> PatchModel:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.results = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
        step = self.step
        self.step += 1
        answer = (
            AIMessage(content="done")
            if step == 5
            else AIMessage(
                content="",
                tool_calls=[
                    {"name": "apply_patch", "args": patch_args(step), "id": f"call_{step}"}
                ],
            )
        )
        return ChatResult(generations=[ChatGeneration(message=answer)])


@pytest.mark.parametrize("framework", ["pydantic", "langchain"])
def test_real_kedi_program_corrects_more_than_three_patch_rejections(
    framework: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.txt"
    data.write_text("same\nsame\n", encoding="utf-8")
    results: list[str] = []
    step = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        results[:] = [
            str(p.content) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
        ]
        index = step
        step += 1
        if index == 5:
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[ToolCallPart("apply_patch", patch_args(index), f"call_{index}")]
        )

    lc_model = PatchModel()
    adapter = (
        PydanticAdapter(FunctionModel(respond))
        if framework == "pydantic"
        else LangChainAdapter(lc_model)
    )
    source = (
        "> import: filesystem\n> use: apply_patch\n> approval: allow\n\n"
        "[result] << Apply the edit and finish.\n= <result>\n"
    )
    program = tmp_path / "patch.kedi"
    program.write_text(source, encoding="utf-8")
    runtime = compile_program(parse_program(source, source_path=str(program)), adapter=adapter)
    assert runtime.run_main() == "done"
    if framework == "langchain":
        results = lc_model.results
    assert len(results) == 5
    assert all("Patch rejected; no changes made:" in result for result in results[:4])
    assert "updated" in results[4]
    assert data.read_text(encoding="utf-8") == "fixed\n"
