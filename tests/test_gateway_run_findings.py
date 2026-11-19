"""Regressions found by running Kedi against OpenCode Go models (paid gateway experiment, 2026-11-20)."""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from pydantic_ai import ModelRetry

from kedi.agent_adapter.adapters import LangChainAdapter
from kedi.agent_adapter.adapters.pydantic import _model_retry_for_tool_error
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.lang import compile_program, parse_program


class OutputToolModel(BaseChatModel):
    """Answers with scripted replies; `None` args mean "call the structured output tool"."""

    replies: list[Any]
    output_args: dict[str, Any] = Field(default_factory=dict)
    tool_names: list[str] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "synthetic-gateway"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OutputToolModel:
        self.tool_names = [
            getattr(t, "name", None) or t.get("name") or t["function"]["name"] for t in tools
        ]
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if reply is None:
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_names[-1],
                        "args": self.output_args,
                        "id": f"out_{len(self.seen)}",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


PROGRAM = """```
from datetime import date
```
~Visit(day: date, guests: int)
>> The visit described here is [visit: Visit].
= `visit`
"""


def run(model: OutputToolModel) -> Any:
    return compile_program(parse_program(PROGRAM), adapter=LangChainAdapter(model)).run_main()


def test_langchain_accepts_json_dates_for_strict_kedi_types() -> None:
    # LangChain validated a Pydantic response format with validate_python, and Kedi types are
    # strict, so the JSON string "2026-04-10" was rejected as a date and the model looped.
    model = OutputToolModel(
        replies=[None], output_args={"visit": {"day": "2026-04-10", "guests": 3}}
    )
    visit = run(model)
    assert visit.day == date(2026, 4, 10) and visit.guests == 3
    assert len(model.seen) == 1


def test_langchain_invalid_structured_output_gets_a_correction() -> None:
    model = OutputToolModel(
        replies=[None, None], output_args={"visit": {"day": "not a date", "guests": 3}}
    )

    def fix_after_first(_: Any) -> None:
        model.output_args = {"visit": {"day": "2026-04-11", "guests": 3}}

    original = model._generate

    def generate(messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        if model.seen:
            fix_after_first(None)
        return original(messages, *args, **kwargs)

    object.__setattr__(model, "_generate", generate)
    visit = run(model)
    assert visit.day == date(2026, 4, 11)
    assert any(
        isinstance(m, HumanMessage) and "does not match the required schema" in str(m.content)
        for m in model.seen[1]
    )


def test_langchain_text_reply_instead_of_output_gets_a_correction() -> None:
    # Some gateway models write the output tool call as text when tool_choice is required.
    text = AIMessage(content="<tool_call>_OutputModel><parameter=visit>...</parameter></tool_call>")
    model = OutputToolModel(
        replies=[text, None], output_args={"visit": {"day": "2026-04-10", "guests": 2}}
    )
    visit = run(model)
    assert visit.guests == 2
    assert any(
        isinstance(m, HumanMessage)
        and "did not include the required structured output" in str(m.content)
        for m in model.seen[1]
    )


def application_tool(fn: Any) -> ToolSpec:
    return ToolSpec(
        fn=fn,
        name="lookup",
        description="Look up a record",
        json_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        risk="read_only",
        metadata={},
        artifact_result="never",
    )


def test_application_tool_input_errors_go_back_to_the_model() -> None:
    def lookup(key: str) -> str:
        raise LookupError(f"no record {key}")

    retry = _model_retry_for_tool_error(
        application_tool(lookup), ValueError("day must be an ISO date")
    )
    assert isinstance(retry, ModelRetry) and "day must be an ISO date" in str(retry)
    assert isinstance(
        _model_retry_for_tool_error(application_tool(lookup), KeyError("O-1")), ModelRetry
    )
    assert isinstance(
        _model_retry_for_tool_error(application_tool(lookup), TimeoutError("slow")), ModelRetry
    )
    # Defects and unknown failures in the tool's own code still end the run.
    assert _model_retry_for_tool_error(application_tool(lookup), TypeError("bad call")) is None
    assert _model_retry_for_tool_error(application_tool(lookup), RuntimeError("bug")) is None


def test_langchain_loop_recovers_from_an_application_tool_error() -> None:
    calls: list[str] = []

    def lookup(key: str) -> str:
        calls.append(key)
        if key != "O-2":
            raise ValueError("unknown order; order ids look like O-2")
        return "shipped"

    replies = [
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"key": "2"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"key": "O-2"}, "id": "c2"}]),
        AIMessage(content="shipped"),
    ]
    model = OutputToolModel(replies=replies)
    adapter = LangChainAdapter(model)
    with adapter.tool_scope([application_tool(lookup)]):
        assert adapter.invoke_sync(prompt="What is the status of order 2?") == "shipped"
    assert calls == ["2", "O-2"]
    assert any(
        isinstance(m, ToolMessage)
        and m.status == "error"
        and "order ids look like" in str(m.content)
        for m in model.seen[1]
    )


def test_langchain_text_replies_stop_after_the_output_retries() -> None:
    import pytest

    text = AIMessage(content="The visit is on April 10 with 2 guests.")
    model = OutputToolModel(replies=[text, AIMessage(content=text.content)])
    with pytest.raises(Exception, match="required key"):
        run(model)
    assert (
        len(model.seen) == 2
    )  # one correction (output_retries=1), then the usual missing-output error


def test_langchain_invalid_output_after_the_retries_reports_validation() -> None:
    import pytest

    model = OutputToolModel(
        replies=[None, None], output_args={"visit": {"day": "soon", "guests": 3}}
    )
    with pytest.raises(Exception, match="valid date"):
        run(model)
    assert len(model.seen) == 2
