import asyncio
import json
import re

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.agent_profile import AgentProfile
from kedi.artifacts import ArtifactPolicy
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("case", ["scope-failure", "history-lanes", "template-raw-template"])
async def test_wire_requests_restore_scoped_config_and_isolate_history(backend, case):
    requests, responses, outputs, lanes, effects = [], {}, [], [], []
    requests_by_response = {}
    codes = ["a" * 32, "b" * 32]
    state = ConversationState()

    async def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        supplied = re.findall(r"The current scoped code is ([0-9a-f]{32})", json.dumps(body))
        code = supplied[-1] if supplied else responses[body["previous_response_id"]]
        response_id = f"resp_fixture_{len(responses)}"
        responses[response_id] = code
        requests_by_response[response_id] = body
        return httpx.Response(
            200,
            json={
                "id": response_id,
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.2",
                "output": [
                    {
                        "id": "msg_fixture",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"code": code}),
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        )

    def inner_probe() -> str:
        effects.append("inner_probe")
        return codes[1]

    inner_probe.__kedi_tool_risk__ = "read_only"
    tool = python_callable_tool_spec("inner_probe", inner_probe)
    outer = AgentProfile(
        system=f"The current scoped code is {codes[0]}. Copy it exactly.",
        effort="low",
        approval=ApprovalPolicy(mode="deny"),
        artifacts=ArtifactPolicy(enabled=False),
    )
    inner = AgentProfile(
        system=f"The current scoped code is {codes[1]}. Copy it exactly.",
        effort="medium",
        approval=ApprovalPolicy(mode="allow"),
        artifacts=ArtifactPolicy(enabled=True),
        tools=(tool.name,),
        tool_bindings={tool.name: tool},
    )

    async def execute(adapter, profile, *, raw=False):
        engine = SequentialEngine()
        runtime = compile_program(
            parse_program(
                "> history: enabled\n"
                + (
                    "[code] << Report the current scoped code.\n"
                    if raw
                    else ">> The current scoped code is [code: str].\n"
                )
                + "= `code`"
            ),
            adapter=adapter,
            engine=engine,
            conversation_state=state,
            default_agent_profile=profile,
        )
        try:
            result = await asyncio.to_thread(runtime.run_main)
            outputs.append(json.loads(result)["code"] if raw else result)
            lanes.append(state.adapter_state_snapshot(adapter.shortname).lane_identity)
        finally:
            await runtime.aclose()
            engine.shutdown()

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async with AsyncOpenAI(api_key="fixture", http_client=http, max_retries=0) as client:
                adapter = (
                    PydanticAdapter(
                        OpenAIResponsesModel(
                            "gpt-5.2", provider=OpenAIProvider(openai_client=client)
                        )
                    )
                    if backend == "pydantic"
                    else LangChainAdapter(
                        ChatOpenAI(
                            model="gpt-5.2",
                            api_key="fixture",
                            root_async_client=client,
                            async_client=client.chat.completions,
                            use_responses_api=True,
                            streaming=False,
                            max_retries=0,
                        )
                    )
                )
                original = adapter.active_profile
                with use_async_owner_loop():
                    if case == "scope-failure":
                        with adapter.profile_override(outer):
                            await execute(adapter, adapter.active_profile)
                            with pytest.raises(LookupError, match="nested failure"):
                                with adapter.profile_override(inner):
                                    await execute(adapter, adapter.active_profile)
                                    raise LookupError("nested failure")
                            assert adapter.active_profile == outer
                            await execute(adapter, adapter.active_profile)
                        assert adapter.active_profile == original
                    elif case == "history-lanes":
                        for profile in (outer, outer, inner, inner):
                            await execute(adapter, profile)
                        assert lanes[0] == lanes[1] and lanes[2] == lanes[3]
                        assert lanes[0] != lanes[2]
                    else:
                        await execute(adapter, outer)
                        await execute(adapter, outer, raw=True)
                        await execute(adapter, outer)
        selected = {
            "scope-failure": [0, 1, 0],
            "history-lanes": [0, 0, 1, 1],
            "template-raw-template": [0, 0, 0],
        }[case]
        assert outputs == [codes[i] for i in selected]
        assert len(requests) == len(selected)
        assert [r["reasoning"]["effort"] for r in requests] == [
            ["low", "medium"][i] for i in selected
        ]
        assert [
            any(t.get("name") == "inner_probe" for t in r.get("tools", [])) for r in requests
        ] == [bool(i) for i in selected]
        assert effects == []

        def effective_instructions(body):
            if "instructions" in body:
                return body["instructions"]
            messages = [
                m
                for m in body.get("input", [])
                if isinstance(m, dict) and m.get("role") in {"system", "developer"}
            ]
            if messages:
                return json.dumps(messages)
            if "previous_response_id" in body:
                return effective_instructions(requests_by_response[body["previous_response_id"]])
            return ""

        for turn, (body, index) in enumerate(zip(requests, selected)):
            serialized = effective_instructions(body)
            completion = not (case == "template-raw-template" and turn == 1)
            assert serialized.count("<TemplateInstructions>") == int(completion)
            assert serialized.count("</TemplateInstructions>") == int(completion)
            assert serialized.count("<ToolCallInstructions>") == int(bool(index))
            assert serialized.count("</ToolCallInstructions>") == int(bool(index))
            if index:
                assert serialized.index("</TemplateInstructions>") < serialized.index(
                    "<ToolCallInstructions>"
                )
            for message in body.get("input", []):
                if isinstance(message, dict) and message.get("role") == "user":
                    assert "<TemplateInstructions>" not in json.dumps(message)
                    assert "Fill each unescaped" not in json.dumps(message)
    finally:
        state.close()
