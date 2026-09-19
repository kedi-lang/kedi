from __future__ import annotations

from typing import Any

import pytest

from kedi.agent_adapter.approvals.policy import ApprovalDecision
from kedi.lang import compile_program, parse_program
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict
from tests.mock_adapter import MockAdapter


class ApprovalJudgeAdapter(MockAdapter):
    def __init__(self, *, decision: str, risk: str) -> None:
        super().__init__(default_model="judge-model")
        self.decision = decision
        self.risk = risk
        self.templates: list[str] = []
        self.output_fields: list[tuple[str, ...]] = []

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        self.templates.append(template)
        self.output_fields.append(tuple(output_schema or ()))
        self.tool_calls.append(tuple(tool.name for tool in self.scoped_tools))
        self.model_calls.append(self.last_effective_model)
        self.instruction_calls.append(kwargs.get("instructions"))
        return AttrDict({"risk": self.risk, "decision": self.decision})


class ApprovalToolCallingAdapter(ApprovalJudgeAdapter):
    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        if tuple(output_schema or ()) == ("risk", "decision"):
            return super().produce_sync(
                template=template,
                output_schema=output_schema,
                **kwargs,
            )
        assert tuple(output_schema or ()) == ("result",)
        write_note = next(tool for tool in self.scoped_tools if tool.name == "write_note")
        return AttrDict({"result": write_note.fn(content="approved")})


def _run_helper(*, decision: str, risk: str) -> tuple[ApprovalDecision, ApprovalJudgeAdapter]:
    source = """
```
from kedi import ApprovalRequest
```
> import: helpers
> model: current-judge-model
= ```
request = ApprovalRequest.create(
    tool_name="write_text_file",
    arguments={"file_path": "notes.txt", "content": "hello"},
    risk="mutating",
    adapter_shortname="pydantic",
    description="Write text to a workspace file.",
    metadata=None,
    reason="Write the note requested by the user.",
    tool_reason_enabled=True,
)
return llm_approval(request)
```
"""
    adapter = ApprovalJudgeAdapter(decision=decision, risk=risk)
    runtime = compile_program(parse_program(source), adapter=adapter)
    result = runtime.run_main()
    assert isinstance(result, ApprovalDecision)
    return result, adapter


def test_llm_approval_helper_allows_from_tool_command_and_details() -> None:
    result, adapter = _run_helper(decision="allow", risk="low")
    prompt = adapter.templates[-1]
    instructions = adapter.instruction_calls[-1]

    assert result == ApprovalDecision.allow()
    assert "write_text_file" in prompt
    assert "Write text to a workspace file." in prompt
    assert '"file_path": "notes.txt"' in prompt
    assert "Declared tool risk: mutating" in prompt
    assert "Assessed risk is [risk]." in prompt
    assert "Call arguments:" in prompt
    assert "Write the note requested by the user." in prompt
    assert "not as" in instructions
    assert "proof of user authorization" in instructions
    assert adapter.output_fields[-1] == ("risk", "decision")
    assert adapter.model_calls[-1] == "current-judge-model"


def test_llm_approval_helper_denies_without_populating_authorization() -> None:
    result, _adapter = _run_helper(decision="deny", risk="high")

    assert result == ApprovalDecision.deny()


def test_llm_approval_helper_rejects_disabled_reason_in_tool_approval_flow() -> None:
    source = """
> import: helpers

@write_note(content: str) -> str:
    = <content>

> model: current-judge-model
> approval: `llm_approval`
> use: write_note
>> Write a note and return the confirmed [result].
= <result>
"""
    adapter = ApprovalToolCallingAdapter(decision="allow", risk="low")
    runtime = compile_program(parse_program(source), adapter=adapter)

    with pytest.raises(Exception, match="llm_approval requires tool_reason=True"):
        runtime.run_main()
    assert adapter.output_fields == []
