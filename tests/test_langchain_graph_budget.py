import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.tools.specs import ToolSpec


class ToolLoopModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("excess", [False, True])
def test_real_graph_honors_request_budget_instead_of_premature_recursion_limit(excess):
    calls = []
    model = ToolLoopModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "read_page", "args": {"index": i}, "id": f"call_{i}"}],
                )
                for i in range(20 if excess else 19)
            ]
            + [AIMessage(content="finished")]
        )
    )
    adapter = LangChainAdapter(model, default_usage_limits=UsageLimits(request_limit=20))
    tool = ToolSpec(
        fn=lambda index: calls.append(index) or "page",
        name="read_page",
        description="Read a ledger page.",
        json_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
        risk="read_only",
        artifact_result="never",
    )
    with adapter.tool_scope([tool]):
        if excess:
            with pytest.raises(UsageLimitExceeded, match="request_limit"):
                adapter.invoke_sync(prompt="Read the pages and finish.")
        else:
            assert adapter.invoke_sync(prompt="Read the pages and finish.") == "finished"
    assert calls == list(range(20 if excess else 19))
