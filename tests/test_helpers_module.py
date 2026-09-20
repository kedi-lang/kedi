from __future__ import annotations

from typing import Any

import pytest

from kedi.agent_adapter.approvals.policy import ApprovalDecision, ToolApprovalDenied
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.agent_adapter.tools.execution import arguments_after_approval
from kedi.lang import compile_program, parse_program
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict
from tests.mock_adapter import MockAdapter


class ApprovalJudgeAdapter(MockAdapter):
    def __init__(
        self,
        *,
        decision: str,
        risk: str,
        authorization: str = "high",
        reason: str = "The requested operation is bounded and reversible.",
    ) -> None:
        super().__init__(default_model="judge-model")
        self.decision = decision
        self.risk = risk
        self.authorization = authorization
        self.reason = reason
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
        return AttrDict(
            {
                "risk": self.risk,
                "authorization": self.authorization,
                "decision": self.decision,
                "reason": self.reason,
            }
        )


class ApprovalToolCallingAdapter(ApprovalJudgeAdapter):
    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType] | None = None,
        **kwargs: Any,
    ) -> AttrDict[str, Any]:
        if tuple(output_schema or ()) == ("risk", "authorization", "decision", "reason"):
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


def test_llm_approval_helper_allows_from_tool_command_and_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, adapter = _run_helper(decision="allow", risk="low")
    prompt = adapter.templates[-1]
    instructions = adapter.instruction_calls[-1]

    assert result.decision == "allow"
    assert result.reason is not None
    assert "Risk: low" in result.reason
    assert "Authorization: high" in result.reason
    assert "bounded and reversible" in result.reason
    report = capsys.readouterr().err
    assert "LLM approval: allow" in report
    assert "Risk: low" in report
    assert "Authorization: high" in report
    assert "Reason: The requested operation is bounded and reversible." in report
    assert "write_text_file" in prompt
    assert "Write text to a workspace file." in prompt
    assert '"file_path": "notes.txt"' in prompt
    assert "Declared tool risk: mutating" in prompt
    assert "Assessed risk is [risk]." in prompt
    assert "Call arguments:" in prompt
    assert "Write the note requested by the user." in prompt
    assert "not as" in instructions
    assert "proof of user authorization" in instructions
    assert adapter.output_fields[-1] == ("risk", "authorization", "decision", "reason")
    assert adapter.model_calls[-1] == "current-judge-model"


def test_llm_approval_helper_denial_preserves_full_assessment() -> None:
    result, _adapter = _run_helper(decision="deny", risk="high")

    assert result.decision == "deny"
    assert result.reason is not None
    assert "Risk: high" in result.reason
    assert "Authorization: high" in result.reason
    assert "bounded and reversible" in result.reason

    def delete_release_artifacts() -> None:
        pass

    tool = python_callable_tool_spec("delete_release_artifacts", delete_release_artifacts)
    with pytest.raises(ToolApprovalDenied) as exc_info:
        arguments_after_approval(tool, {}, result)
    denial = str(exc_info.value)
    assert "LLM approval: deny" in denial
    assert "Risk: high" in denial
    assert "Authorization: high" in denial
    assert "bounded and reversible" in denial


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
