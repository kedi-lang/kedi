"""Exercise the published workflows, not a second implementation of their source."""

import re
import runpy
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

import kedi
from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program
from kedi.tests_runner import run_tests

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
WORKFLOWS = {
    "meeting_actions",
    "release_notes",
    "support_reply",
    "parallel_review",
    "artifact_investigation",
    "python_api",
    "jev_routing",
    "module_imports",
    "knapsack",
    "flappy_kedi",
}


@pytest.fixture(params=["example", "repository", "elsewhere"])
def example_runtime(monkeypatch, request, tmp_path):
    runtimes = []
    monkeypatch.delenv("KEDI_ADAPTER_MODEL", raising=False)
    monkeypatch.delenv("KEDI_AGENT", raising=False)

    def build(name, model=None):
        source = EXAMPLES / name / "main.kedi"
        monkeypatch.chdir(
            {"example": source.parent, "repository": EXAMPLES.parent, "elsewhere": tmp_path}[
                request.param
            ]
        )
        runtime = compile_program(
            parse_program(source.read_text(), source_path=str(source)),
            adapter=PydanticAdapter(model) if model is not None else None,
        )
        runtimes.append(runtime)
        return runtime

    yield build
    for runtime in runtimes:
        runtime.artifacts.close()


@pytest.mark.parametrize("source", sorted(EXAMPLES.rglob("*.kedi")))
def test_every_published_kedi_source_parses(source):
    parse_program(source.read_text(), source_path=str(source))


def test_catalog_has_no_loose_scripts_and_every_workflow_has_a_guide():
    assert not list(EXAMPLES.glob("*.py"))
    assert not list(EXAMPLES.glob("*.kedi"))
    index = (EXAMPLES / "README.md").read_text()
    for name in WORKFLOWS:
        guide = EXAMPLES / name / "README.md"
        assert guide.is_file()
        assert f"{name}/README.md" in index
        assert "## Run" in guide.read_text() or name == "knapsack"


@pytest.mark.parametrize("document", ["README.md", "docs/KEDI_DSL.md"])
def test_repository_introduction_and_specification_have_parseable_kedi_blocks(document):
    blocks = list(
        re.finditer(
            r"(?m)^(?P<fence>`{3,})kedi\n(?P<source>[\s\S]*?)^(?P=fence)\s*$",
            (EXAMPLES.parent / document).read_text(),
        )
    )
    assert blocks
    for block in blocks:
        parse_program(block["source"])


def test_guides_do_not_link_to_deleted_example_files():
    documents = [EXAMPLES.parent / "README.md", EXAMPLES.parent / "docs/README.md"]
    documents.extend(EXAMPLES.rglob("*.md"))
    for document in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("https://", "http://", "#")):
                continue
            assert (document.parent / target.split("#")[0]).exists(), (document, target)


def test_meeting_actions_preserve_nested_types_and_missing_values(example_runtime):
    actions = [
        {
            "task": "Add the duplicate-payment regression test",
            "owner": "Sam",
            "deadline": "2026-09-25",
        },
        {"task": "Update the rollback runbook", "owner": "Alex", "deadline": None},
        {"task": "Schedule a production rehearsal", "owner": None, "deadline": None},
    ]
    model = TestModel(call_tools=[], custom_output_args={"actions": actions})
    result = example_runtime("meeting_actions", model).run_main()
    assert [action.model_dump() for action in result] == actions
    assert all(isinstance(action, BaseModel) for action in result)
    with pytest.raises(ValidationError, match="at least 1 character"):
        type(result[0])(task="", owner=None, deadline=None)
    schema = model.last_model_request_parameters.output_tools[0].parameters_json_schema
    action_schema = schema["$defs"]["Action"]
    assert "not a suggestion" in action_schema["properties"]["task"]["description"]
    assert action_schema["properties"]["task"]["minLength"] == 1
    assert {s["type"] for s in action_schema["properties"]["owner"]["anyOf"]} == {"string", "null"}


def test_release_notes_skip_internal_changes_and_preserve_order(example_runtime):
    calls = []

    def respond(messages, info):
        prompt = str(messages[-1])
        calls.append(prompt)
        tool = info.output_tools[0]
        if "Claim:" in prompt:
            value = {next(iter(tool.parameters_json_schema["properties"])): "fixture" not in prompt}
        elif "CSV" in prompt:
            value = {"note": "CSV exports preserve non-ASCII customer names."}
        else:
            value = {"note": "On Linux, resume interrupted uploads up to 100 MB."}
        return ModelResponse(parts=[ToolCallPart(tool.name, value)])

    assert example_runtime("release_notes", FunctionModel(respond)).run_main() == [
        "CSV exports preserve non-ASCII customer names.",
        "On Linux, resume interrupted uploads up to 100 MB.",
    ]
    assert len(calls) == 5
    assert sum("Claim:" in prompt for prompt in calls) == 3
    assert sum("fixture" in prompt for prompt in calls) == 1


def test_support_profile_calls_real_policy_tool_before_returning_reply(example_runtime):
    calls = []

    def respond(messages, info):
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        calls.append(messages)
        assert "read_delivery_policy" in {tool.name for tool in info.function_tools}
        if not returns:
            return ModelResponse(parts=[ToolCallPart("read_delivery_policy", {})])
        assert returns[-1].content == (EXAMPLES / "support_reply/delivery_policy.md").read_text()
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "reply": {
                            "message": "Please share the order number and latest tracking update so we can investigate.",
                            "policy_sections": ["Delayed parcels", "Refund eligibility"],
                        }
                    },
                )
            ]
        )

    result = example_runtime("support_reply", FunctionModel(respond)).run_main()
    assert result.policy_sections == ["Delayed parcels", "Refund eligibility"]
    assert "order number" in result.message
    assert len(calls) == 2


def test_parallel_review_starts_both_children_and_joins_typed_results(example_runtime):
    rendezvous = Barrier(2, timeout=10)
    calls = []

    def respond(messages, info):
        prompt = str(messages)
        security = "authorization and data exposure" in prompt
        review = {
            "issue": "Account ownership is not checked"
            if security
            else "Connection leaks on error",
            "recommendation": "Verify ownership" if security else "Use a context manager",
        }
        calls.append((prompt, review))
        rendezvous.wait()
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "task_summary": "Reviewed the supplied patch.",
                        "final_result": {"review": review},
                    },
                )
            ]
        )

    result = example_runtime("parallel_review", FunctionModel(respond)).run_main()
    assert result.security.recommendation == "Verify ownership"
    assert result.reliability.recommendation == "Use a context manager"
    assert isinstance(result.security, BaseModel)
    assert len(calls) == 2
    assert all("connection.close()" in prompt for prompt, _ in calls)


def test_incident_tool_becomes_an_artifact_and_can_be_queried(example_runtime):
    calls = []

    def respond(messages, info):
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("read_incident_log", {})])
        last = returns[-1]
        calls.append(last.tool_name)
        if last.tool_name == "read_incident_log":
            assert "DB_POOL_SIZE" not in str(last.content)
            assert "ref_id" in last.content
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "query_artifact",
                        {
                            "ref_id": last.content["ref_id"],
                            "query": "DB_POOL_SIZE connection limit 180",
                            "max_chars": 1800,
                            "top_k": 3,
                        },
                    )
                ]
            )
        assert last.tool_name == "query_artifact"
        assert "DB_POOL_SIZE" in str(last.content)
        assert "180" in str(last.content)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "finding": {
                            "cause": "Eight pools of 40 exceeded the 180-connection limit.",
                            "mitigation": "Restore pool size to 20.",
                            "evidence": [
                                "09:30 pool-size change",
                                "09:55 configuration correction",
                            ],
                        }
                    },
                )
            ]
        )

    runtime = example_runtime("artifact_investigation", FunctionModel(respond))
    result = runtime.run_main()
    assert "180" in result.cause
    assert calls == ["read_incident_log", "query_artifact"]
    events = runtime.artifacts.history.snapshot()
    assert sum(event.kind == "artifact_query" for event in events) == 1


@pytest.mark.parametrize("entrypoint", [False, True])
def test_python_query_and_file_binding_pass_the_native_incident(monkeypatch, capsys, entrypoint):
    monkeypatch.delenv("KEDI_ADAPTER_MODEL", raising=False)
    module = runpy.run_path(str(EXAMPLES / "python_api/main.py"))
    prompts = []

    def respond(messages, info):
        prompts.append(str(messages))
        fields = info.output_tools[0].parameters_json_schema["properties"]
        if "incident" in fields:
            value = {"incident": {"service": "checkout", "symptom": "timeouts", "resolved": False}}
        else:
            assert "checkout" in str(messages) and "False" in str(messages)
            value = {"notice": "Checkout requests are timing out. Recovery has not been confirmed."}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, value)])

    with kedi.context(adapter=PydanticAdapter(FunctionModel(respond)), model=None):
        if entrypoint:
            runpy.run_path(str(EXAMPLES / "python_api/main.py"), run_name="__main__")
            output = capsys.readouterr().out
            assert "resolved=False" in output
            assert "not been confirmed" in output
        else:
            incident = module["extract_incident"](
                "Checkout is timing out; the rollback is only planned."
            )
            assert isinstance(incident, module["Incident"])
            assert incident.resolved is False
            assert "not been confirmed" in module["draft_notice"](incident)
    assert len(prompts) == 2


@pytest.mark.parametrize(
    "probability,action", [(0.95, "Escalate to billing."), (0.80, "Queue for billing.")]
)
def test_jev_routing_uses_one_request_and_exposes_threshold_evidence(
    example_runtime, monkeypatch, capsys, probability, action
):
    from kedi_typesafe import TypeSafeModel
    from typesafe_sdk import Choice, ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage

    class Client:
        calls = 0

        async def system_one(self, state, questions, *, model=None):
            self.calls += 1
            assert "duplicate payments" in str(state)
            assert len(questions) == 2
            return SystemOneResponse(
                model="test-jev",
                answers={
                    name: ChoiceAnswer(
                        choice="billing",
                        confidence=1,
                        probabilities={
                            "billing": 1,
                            "access": 0,
                            "general": 0,
                        },
                    )
                    if isinstance(question, Choice)
                    else NoulAnswer(noul=probability)
                    for name, question in questions.items()
                },
                usage=Usage(input_tokens=50, output_tokens=2),
            )

        async def aclose(self):
            pass

    client = Client()
    model = TypeSafeModel(client=client)
    monkeypatch.setattr(PydanticAdapter, "_resolve_model", lambda self, name: model)
    assert example_runtime("jev_routing", model).run_main() == action
    output = capsys.readouterr().out
    assert str(probability) in output and "0.85" in output
    assert client.calls == 1


def test_module_example_runs_without_a_model_and_its_local_tests_pass(example_runtime):
    quote = example_runtime("module_imports").run_main()
    assert quote.model_dump() == {
        "subtotal_cents": 49500,
        "discount_cents": 4950,
        "total_cents": 44550,
    }
    suites = run_tests(str(EXAMPLES / "module_imports/pricing.kedi"))
    assert len(suites) == 1 and len(suites[0].cases) == 2
    assert all(case.success for case in suites[0].cases)
