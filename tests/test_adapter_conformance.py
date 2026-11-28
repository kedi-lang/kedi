"""The adapter conformance kit (#232 step 3): built-ins pass, adapters that author text fail."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from pydantic import TypeAdapter

from kedi.agent_adapter import (
    AdapterCapabilities,
    AgentAdapterKind,
    LangChainAdapter,
    PydanticAdapter,
    RenderedRequest,
    ToolSpec,
    WebGPUAdapter,
    current_request,
)
from kedi.agent_adapter.conformance import SCENARIOS, ConformanceReport, check_adapter
from kedi.agent_adapter.conformance.check import TOOL_MARKER, _check_run, _Run
from kedi.agent_adapter.conformance.endpoint import Exchange, RecordingEndpoint, Script

Mutate = Callable[[dict[str, Any], RenderedRequest], dict[str, Any]]
NATIVE = AdapterCapabilities(
    supports_structured_output=True, supports_tool_registration=True, supports_artifacts=True
)


class OpenAIChatAdapter:
    """A custom adapter on an OpenAI-compatible endpoint that forwards ``render()``.

    ``mutate`` rewrites each request body before it is sent, to build adapters that break the
    contract. ``capabilities`` selects native fields or Kedi's protocol parts.
    """

    kind: AgentAdapterKind = "agent-framework"
    shortname = "openai-chat"

    def __init__(
        self,
        base_url: str,
        mutate: Mutate | None = None,
        capabilities: AdapterCapabilities = NATIVE,
    ) -> None:
        self.capabilities = capabilities
        self._url = f"{base_url}/v1/chat/completions"
        self._mutate = mutate or (lambda body, rendered: body)
        self._tools: ContextVar[tuple[ToolSpec, ...]] = ContextVar(
            f"openai_chat_tools_{id(self)}", default=()
        )

    @classmethod
    def for_conformance(cls, base_url: str) -> OpenAIChatAdapter:
        return cls(base_url)

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        token = self._tools.set(tuple(tool for tool in tools if isinstance(tool, ToolSpec)))
        try:
            yield
        finally:
            self._tools.reset(token)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.set((*self._tools.get(), tool))

    def invoke_sync(self, **_: Any) -> str:
        return str(self._run(structured=False))

    def produce_sync(self, *, output_type: Any = None, **_: Any) -> Any:
        # Template captures arrive with output_schema, and Kedi validates the mapping.
        data = self._run(structured=True)
        return data if output_type is None else TypeAdapter(output_type).validate_python(data)

    def _run(self, *, structured: bool) -> Any:
        rendered = current_request().render()
        native_tools = self.capabilities.supports_native_tool_calling is not False
        native_output = self.capabilities.supports_native_structured_output is not False
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": rendered.system or ""},
            {"role": "user", "content": rendered.user},
        ]
        specs = {tool.name: tool for tool in self._tools.get()}
        body: dict[str, Any] = {"model": "conformance", "messages": messages}
        if native_tools and rendered.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    },
                }
                for tool in rendered.tools
            ]
        if structured and native_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": rendered.output_json_schema},
            }
        for _ in range(4):
            reply = httpx.post(self._url, json=self._mutate(body, rendered), timeout=30).json()
            message = reply["choices"][0]["message"]
            calls = [
                (call["id"], call["function"]["name"], json.loads(call["function"]["arguments"]))
                for call in message.get("tool_calls") or []
            ]
            if not native_tools:
                action = json.loads(message["content"])
                if action["action"] == "final":
                    return action.get("data", action.get("text"))
                calls = [(action["callId"], action["name"], action["arguments"])]
            elif not calls:
                return json.loads(message["content"]) if structured else message["content"]
            messages.append({"role": "assistant", "content": message["content"]})
            for call_id, name, arguments in calls:
                try:
                    result = specs[name].fn(**arguments)
                except Exception as error:  # noqa: BLE001 - the model reads the failure
                    result = f"error: {error}"
                messages.append({"role": "tool", "tool_call_id": call_id, "content": str(result)})
        raise RuntimeError("tool loop did not finish")


def _statuses(report: ConformanceReport) -> dict[str, str]:
    return {result.name: result.status for result in report.results}


def _system(body: dict[str, Any]) -> dict[str, Any]:
    return body["messages"][0]


def _check(mutate: Mutate, capabilities: AdapterCapabilities = NATIVE) -> ConformanceReport:
    return check_adapter(lambda url: OpenAIChatAdapter(url, mutate, capabilities))


@pytest.mark.parametrize(
    "capabilities",
    [
        NATIVE,
        AdapterCapabilities(
            supports_structured_output=True,
            supports_tool_registration=True,
            supports_artifacts=True,
            supports_native_structured_output=False,
            supports_native_tool_calling=False,
        ),
        AdapterCapabilities(
            supports_structured_output=True,
            supports_tool_registration=True,
            supports_artifacts=True,
            supports_native_structured_output=False,
        ),
    ],
    ids=["native", "protocol", "protocol-output"],
)
def test_a_custom_adapter_that_forwards_render_passes(capabilities: AdapterCapabilities) -> None:
    report = _check(lambda body, rendered: body, capabilities)
    assert report.passed, report.render()
    assert _statuses(report) == {
        "invoke": "passed",
        "produce": "passed",
        "claim": "passed",
        "tool-allow": "passed",
        "tool-deny": "passed",
        "media": "skipped",
        "request-budget": "skipped",
    }
    assert report.harness_text == []
    assert report.as_dict()["not_covered"] == ["subagents"]


def test_request_dependent_adapter_text_fails() -> None:
    def fill(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
        fields = ", ".join(sorted((rendered.output_json_schema or {}).get("properties", {})))
        _system(body)["content"] += f"\n\nFill these fields: {fields or 'none'}."
        return body

    report = _check(fill)
    assert not report.passed
    assert "text the adapter wrote changes with the request" in report.render()


def test_a_fixed_harness_prompt_passes_and_is_listed() -> None:
    def harness(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
        _system(body)["content"] = f"You are Canary Harness.\n\n{_system(body)['content']}"
        return body

    report = _check(harness)
    assert report.passed, report.render()
    assert report.harness_text == ["You are Canary Harness."]
    assert "Request-independent harness text" in report.render()


def _drop(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
    first = rendered.instruction_parts[0].content
    _system(body)["content"] = _system(body)["content"].replace(first, "")
    return body


def _duplicate(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
    _system(body)["content"] += "\n\n" + _system(body)["content"]
    return body


def _inline(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
    system, user, *rest = body["messages"]
    merged = {"role": "user", "content": f"{system['content']}\n\n{user['content']}"}
    return {**body, "messages": [merged, *rest]}


def _prompt_twice(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
    body["messages"][1]["content"] += "\n\n" + body["messages"][1]["content"]
    return body


def _strip_descriptions(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key != "description"}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return {
        **body,
        **{key: strip(body[key]) for key in ("tools", "response_format") if key in body},
    }


def _broken(body: dict[str, Any], rendered: RenderedRequest) -> dict[str, Any]:
    raise RuntimeError("backend unreachable")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop, "reached the wire 0 times"),
        (_duplicate, "reached the wire 2 times"),
        (_inline, "is not in the system text"),
        (_prompt_twice, "the prompt reached the user turn 2 times"),
        (_strip_descriptions, "the output schema description 'Full legal name' did not arrive"),
        (_strip_descriptions, "the description of tool canary_tool did not arrive"),
        (_broken, "the program failed: RuntimeError: backend unreachable"),
    ],
)
def test_adapters_that_change_kedis_text_fail(mutate: Mutate, message: str) -> None:
    report = _check(mutate)
    assert not report.passed
    assert message in report.render()


class ClosingAdapter(OpenAIChatAdapter):
    """Counts the adapters the kit closes."""

    opened = 0
    closed = 0

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        type(self).opened += 1

    def close(self) -> None:
        type(self).closed += 1


class AsyncClosingAdapter(ClosingAdapter):
    opened = 0
    closed = 0

    async def aclose(self) -> None:
        type(self).closed += 1


@pytest.mark.parametrize("adapter", [ClosingAdapter, AsyncClosingAdapter])
def test_the_kit_closes_every_adapter_it_creates(adapter: type[ClosingAdapter]) -> None:
    assert check_adapter(adapter).passed
    # One probe for the capabilities, then one adapter per variant of each scenario that ran.
    assert adapter.opened == adapter.closed == 1 + 2 * 5


class BrokenFactoryAdapter:
    @classmethod
    def for_conformance(cls, base_url: str) -> BrokenFactoryAdapter:
        raise ImportError("install the backend extra")


class OfflineAdapter(OpenAIChatAdapter):
    """Answers without the model endpoint: Kedi's request never reaches the wire."""

    def invoke_sync(self, **_: Any) -> str:
        return "offline"


def test_an_adapter_that_never_reaches_the_endpoint_fails() -> None:
    report = check_adapter(OfflineAdapter)
    text = report.render()
    assert "the program returned 'offline', expected 'Hello, reviewers.'" in text
    assert "no request reached the model endpoint" in text


@pytest.mark.parametrize(("seen", "expected"), [(True, False), (False, True)])
def test_tool_results_follow_the_approval(seen: bool, expected: bool) -> None:
    scenario = next(s for s in SCENARIOS if s.tool_result is expected)
    variant = scenario.variants[0]
    rendered = RenderedRequest(
        system=None,
        instruction_parts=(),
        user="prompt",
        media=(),
        output_json_schema=None,
        output_tool_name=None,
        tools=(),
        prompt_kind="raw",
        call_kind="invoke",
    )
    content = TOOL_MARKER if seen else "error: denied"
    body = {
        "messages": [{"role": "user", "content": "prompt"}, {"role": "tool", "content": content}]
    }
    run = _Run([rendered], [Exchange("openai", body)], value=variant.expected)
    (problem,) = _check_run(scenario, variant, run)
    assert problem == (
        "the tool result reached the model although approval denied it"
        if seen
        else "the tool's result never reached the model"
    )


@pytest.mark.parametrize(
    "factory",
    [
        PydanticAdapter.for_conformance,
        LangChainAdapter.for_conformance,
        WebGPUAdapter.for_conformance,
    ],
    ids=["pydantic-ai", "langchain", "webgpu"],
)
def test_built_in_adapters_conform(factory: Any) -> None:
    report = check_adapter(factory)
    assert report.passed, report.render()
    assert report.harness_text in ([], ["The final response which ends this conversation"])


def test_pydantic_ai_on_anthropic_messages_conforms() -> None:
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    def factory(base_url: str) -> PydanticAdapter:
        # A request budget counts every request: the client makes no hidden retries.
        client = AsyncAnthropic(base_url=base_url, api_key="conformance", max_retries=0)
        provider = AnthropicProvider(anthropic_client=client)
        return PydanticAdapter(AnthropicModel("conformance", provider=provider), request_retries=0)

    report = check_adapter(factory)
    assert report.passed, report.render()
    assert _statuses(report)["media"] == "passed"


def test_the_endpoint_refuses_what_it_cannot_answer() -> None:
    with RecordingEndpoint() as endpoint:
        url = f"{endpoint.base_url}/v1/chat/completions"
        idle = httpx.post(url, json={"messages": []})
        assert idle.status_code == 500 and "no scenario is running" in idle.text
        endpoint.script = Script(final="hi", text=True)
        streamed = httpx.post(url, json={"messages": [], "stream": True})
        assert streamed.status_code == 500 and "streamed requests" in streamed.text
        assert httpx.post(f"{endpoint.base_url}/v2/other", json={}).status_code == 404
        assert [exchange.protocol for exchange in endpoint.take()] == ["openai", "openai"]


def test_kedi_adapter_check_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from kedi.cli import main

    runner = CliRunner()
    module = "tests.test_adapter_conformance"
    monkeypatch.syspath_prepend(".")
    result = runner.invoke(main, ["adapter", "check", f"{module}:OpenAIChatAdapter", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["passed"] is True

    assert "Check that an adapter sends" in runner.invoke(main, ["adapter", "--help"]).output
    failing = runner.invoke(main, ["adapter", "check", f"{module}:OfflineAdapter"])
    assert failing.exit_code == 1
    assert "Conformance check for OfflineAdapter: failed" in failing.output

    broken = runner.invoke(main, ["adapter", "check", f"{module}:BrokenFactoryAdapter"])
    assert broken.exit_code == 1
    assert "BrokenFactoryAdapter.for_conformance failed: install the backend extra" in broken.output

    for arguments, message in (
        (["adapter", "check"], "Usage: kedi adapter check (MODULE:CLASS | --bridge COMMAND)"),
        (["adapter", "check", "--bridge"], "Usage: kedi adapter check"),
        (["adapter", "check", "no_colon"], "MODULE:CLASS"),
        (["adapter", "check", f"{module}:Missing"], "has no attribute 'Missing'"),
        (["adapter", "check", "kedi.agent_adapter:ToolSpec"], "has no for_conformance(base_url)"),
    ):
        refused = runner.invoke(main, arguments)
        assert refused.exit_code == 2, refused.output
        assert message in refused.output


def test_kedi_adapter_check_runs_the_json_rpc_fixtures_on_a_bridge() -> None:
    from kedi.agent_adapter.adapters.jsonrpc import REFERENCE_BRIDGE
    from kedi.cli import main

    runner = CliRunner()
    bridge = f"node {REFERENCE_BRIDGE} --base-url http://127.0.0.1:9/v1"
    result = runner.invoke(main, ["adapter", "check", "--bridge", bridge, "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    statuses = {fixture["name"]: fixture["status"] for fixture in report["fixtures"]}
    assert report["ok"] is True
    assert statuses["kedi-tool"] == statuses["live-input"] == "pass"
    assert statuses["codemode"] == "skip"
    text = runner.invoke(main, ["adapter", "check", "--bridge", bridge])
    assert text.exit_code == 0 and "pass  governed-denied-request" in text.output

    missing = runner.invoke(main, ["adapter", "check", "--bridge", "kedi-no-such-bridge"])
    assert missing.exit_code == 1
    assert "the bridge did not start" in missing.output
