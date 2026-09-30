from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi import KediRuntime
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_profile import AgentProfile
from kedi.artifacts import (
    ArtifactAccessError,
    ArtifactExpiredError,
    ArtifactInputError,
    ArtifactManager,
    ArtifactPolicy,
    ArtifactReleasedError,
    artifact_tool_specs,
)
from kedi.lang import compile_program, parse_program


@pytest.fixture(params=["memory", "file"])
def stored(request, tmp_path):
    manager = ArtifactManager(session_id="lexical-query")
    policy = ArtifactPolicy(store=request.param, path=tmp_path, threshold_bytes=1)
    try:
        yield manager, policy
    finally:
        manager.close()


def test_query_ranks_relevant_tail_above_repeated_noise(stored):
    manager, policy = stored
    text = (
        "The server is operating normally.\n" * 5000
        + "The database connection timeout is 30 seconds. Increase requestTimeout for slow queries.\n"
        + "Routine housekeeping completed.\n" * 30
    )
    ref = manager.artifactize(text, policy=policy).ref.ref_id
    result = manager.query(ref, "What is the database connection timeout?", top_k=3)
    assert "30 seconds" in result["matches"][0]["content"]
    assert result["total_chars"] == len(text)
    assert result["exhaustive"] is False
    assert result["returned_chars"] <= policy.read_max_chars
    for match in result["matches"]:
        assert match["content"] == text[match["offset"] : match["end_offset"]]
    assert manager.history.snapshot()[-1].kind == "artifact_query"


@pytest.mark.parametrize("query", ["request timeout", "requestTimeout", "request_timeout"])
def test_identifiers_are_searchable_without_exact_spelling(stored, query):
    manager, policy = stored
    ref = manager.artifactize(
        "Set requestTimeout to 42.\n" + "noise\n" * 1000, policy=policy
    ).ref.ref_id
    result = manager.query(ref, query)
    assert "requestTimeout" in result["matches"][0]["content"]


def test_stemming_unicode_and_offsets(stored):
    manager, policy = stored
    text = "Örnek 🙂\n" * 1000 + "The runners were running beside the café. 東京駅 is nearby."
    ref = manager.artifactize(text, policy=policy).ref.ref_id
    for query in ("run cafe", "東京駅", "Örnek"):
        result = manager.query(ref, query, max_chars=350)
        assert result["matches"]
        for match in result["matches"]:
            assert text[match["offset"] : match["end_offset"]] == match["content"]


@pytest.mark.parametrize("prefix", [1185, 65525])
def test_matches_cross_passage_and_store_boundaries(stored, prefix):
    manager, policy = stored
    text = "x " * (prefix // 2) + "special boundary evidence ZEBRA42 " + "y " * 1000
    ref = manager.artifactize(text, policy=policy).ref.ref_id
    result = manager.query(ref, "special boundary evidence ZEBRA42")
    assert "special boundary evidence ZEBRA42" in result["matches"][0]["content"]


def test_total_budget_top_k_and_nonoverlap(stored):
    manager, policy = stored
    text = ("database timeout marker\n" + "padding " * 400) * 20
    ref = manager.artifactize(text, policy=policy).ref.ref_id
    for budget in (1, 50, 1700, 100_000, -1):
        result = manager.query(ref, "database timeout", max_chars=budget, top_k=10)
        assert result["returned_chars"] == sum(len(x["content"]) for x in result["matches"])
        assert result["returned_chars"] <= result["applied_max_chars"] <= policy.read_max_chars
        spans = sorted((x["offset"], x["end_offset"]) for x in result["matches"])
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert len(spans) <= 10


def test_empty_no_match_and_query_syntax_are_safe(stored):
    manager, policy = stored
    ref = manager.artifactize("unrelated text", policy=policy).ref.ref_id
    for query in ("unknown", '" OR NEAR(a) NOT *', "'; DROP TABLE passages; --"):
        assert manager.query(ref, query)["matches"] == []
    empty = manager.artifactize("", policy=ArtifactPolicy(threshold_bytes=0)).ref.ref_id
    assert manager.query(empty, "hello")["total_chars"] == 0


def test_small_output_budget_keeps_match_and_does_not_fragment_index(stored):
    manager, policy = stored
    text = "padding " * 80 + "uniqueNeedle is here" + " remainder" * 80
    ref = manager.artifactize(text, policy=policy).ref.ref_id
    for query in ("uniqueNeedle", "unique needle"):
        small = manager.query(ref, query, max_chars=30)
        large = manager.query(ref, query)
        assert "uniqueNeedle" in small["matches"][0]["content"]
        assert small["searched_passages"] == large["searched_passages"]
        assert small["returned_chars"] == 30
        item = small["matches"][0]
        assert text[item["offset"] : item["end_offset"]] == item["content"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": ""},
        {"query": "  "},
        {"query": "***"},
        {"query": "x" * 2049},
        {"query": None},
        {"query": " ".join(f"word{i}" for i in range(65))},
        {"max_chars": 0},
        {"max_chars": -2},
        {"max_chars": True},
        {"max_chars": 2.5},
        {"top_k": 0},
        {"top_k": 11},
        {"top_k": True},
        {"top_k": 1.5},
    ],
)
def test_invalid_requests_do_not_damage_artifact(stored, kwargs):
    manager, policy = stored
    ref = manager.artifactize("useful evidence", policy=policy).ref.ref_id
    with pytest.raises(ArtifactInputError):
        manager.query(ref, **{"query": "evidence", **kwargs})
    assert manager.query(ref, "evidence")["matches"]


def test_json_projection_is_queried_without_materializing_file_payload(stored, monkeypatch):
    manager, policy = stored
    ref = manager.artifactize(
        {"settings": {"retryTimeout": 75}, "log": "ok\n" * 500}, policy=policy
    ).ref.ref_id
    store = manager._ref_stores[ref]
    if policy.store == "file":
        monkeypatch.setattr(store, "get_value", lambda *a: pytest.fail("materialized payload"))
    result = manager.query(ref, "retry timeout")
    assert "retryTimeout" in result["matches"][0]["content"]
    for match in result["matches"]:
        page = manager.read(ref, offset=match["offset"], max_chars=len(match["content"]))
        assert page.content == match["content"]


def test_lifecycle_and_session_isolation(stored):
    manager, policy = stored
    ref = manager.artifactize("secret evidence", policy=policy).ref.ref_id
    other = ArtifactManager(session_id="other")
    try:
        with pytest.raises((ArtifactAccessError, KeyError)):
            other.query(ref, "secret")
    finally:
        other.close()
    manager.query(ref, "secret")
    manager.release(ref)
    with pytest.raises(ArtifactReleasedError):
        manager.query(ref, "secret")
    now = [0.0]
    expiring = ArtifactManager(monotonic=lambda: now[0])
    try:
        ref = expiring.artifactize(
            "evidence", policy=ArtifactPolicy(threshold_bytes=1, ttl_seconds=1)
        ).ref.ref_id
        now[0] = 2.0
        expiring.cleanup_expired()
        with pytest.raises(ArtifactExpiredError):
            expiring.query(ref, "evidence")
    finally:
        expiring.close()


def test_concurrent_queries_are_deterministic(stored):
    manager, policy = stored
    ref = manager.artifactize("noise\n" * 1000 + "unique evidence value", policy=policy).ref.ref_id
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: manager.query(ref, "unique evidence"), range(8)))
    assert all(result == results[0] for result in results)


def test_multiple_passages_relevance_and_phrase_ranking(stored):
    manager, policy = stored
    text = (
        "timeout " * 700
        + "\nDatabase connection timeout recovery uses exponential backoff.\n"
        + "healthy worker idle\n" * 150
        + "Database connection timeout recovery uses bounded retries.\n"
        + "healthy worker idle\n" * 150
    )
    ref = manager.artifactize(text, policy=policy).ref.ref_id
    result = manager.query(ref, "database connection timeout recovery", top_k=2)
    assert len(result["matches"]) == 2
    assert all("Database connection timeout recovery" in m["content"] for m in result["matches"])


def test_query_temporary_index_closes_when_projection_fails(monkeypatch):
    import sqlite3

    from kedi.artifacts import query as query_module

    original_connect = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(query_module.sqlite3, "connect", connect)

    def blocks():
        yield "partially indexed " * 200
        raise OSError("projection read failed")

    with pytest.raises(OSError, match="projection read failed"):
        query_module.query_passages(blocks(), query="indexed", max_chars=100, top_k=1)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain", "claude", "codex"])
@pytest.mark.parametrize("enabled", [False, True])
def test_constructor_controls_tools_and_instructions(adapter_name, enabled):
    from importlib import import_module

    module = import_module(f"kedi.agent_adapter.adapters.{adapter_name}")
    cls = getattr(
        module,
        {
            "pydantic": "PydanticAdapter",
            "langchain": "LangChainAdapter",
            "claude": "ClaudeAdapter",
            "codex": "CodexAdapter",
        }[adapter_name],
    )
    adapter = cls(query_artifact=enabled)
    runtime = KediRuntime(adapter=adapter)
    try:
        with runtime.bind_agent(profile=AgentProfile.empty()) as surface:
            assert ("query_artifact" in {tool.name for tool in surface.tool_specs}) is enabled
            assert ("query_artifact" in (surface.profile.system or "")) is enabled
        with runtime.bind_agent(
            profile=AgentProfile(artifacts=ArtifactPolicy(enabled=False))
        ) as surface:
            assert "query_artifact" not in {tool.name for tool in surface.tool_specs}
        with pytest.raises(TypeError, match="boolean"):
            cls(query_artifact="yes")
    finally:
        runtime.artifacts.close()


def test_query_runs_through_langchain_tool_boundary(stored):
    from kedi.agent_adapter.adapters.langchain import _langchain_tool_from_spec

    manager, policy = stored
    ref = manager.artifactize("needle evidence", policy=policy).ref.ref_id
    spec = next(
        t for t in artifact_tool_specs(manager, query_artifact=True) if t.name == "query_artifact"
    )
    assert spec.artifact_result == "never"
    native = _langchain_tool_from_spec(spec)

    async def run():
        for query, status in [("***", "error"), ("needle", "success")]:
            result = await native.ainvoke(
                {
                    "type": "tool_call",
                    "id": "query-call",
                    "name": spec.name,
                    "args": {"ref_id": ref, "query": query},
                }
            )
            assert result.status == status
            if status == "success":
                assert json.loads(result.content)["matches"][0]["content"] == "needle evidence"

    asyncio.run(run())


@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"ref_id": "ref"},
        {"query": "needle"},
        {"ref_id": "ref", "query": "needle", "extra": True},
    ],
)
def test_query_signature_errors_are_model_correctable(stored, adapter, arguments):
    from pydantic_ai import ModelRetry
    from langchain_core.tools import ToolException
    from kedi.agent_adapter.adapters.pydantic.tools import _validated_tool_function
    from kedi.agent_adapter.adapters.langchain.tools import _langchain_sync_tool_function

    manager, _ = stored
    spec = next(
        t for t in artifact_tool_specs(manager, query_artifact=True) if t.name == "query_artifact"
    )
    function = (
        _validated_tool_function(spec)
        if adapter == "pydantic"
        else _langchain_sync_tool_function(spec)
    )
    error = ModelRetry if adapter == "pydantic" else ToolException
    with pytest.raises(error, match="Artifact request rejected"):
        function(**arguments)
    assert not manager.history.snapshot()


def test_pydantic_model_can_repair_missing_query_arguments():
    from pydantic_ai.messages import RetryPromptPart

    turns = []

    def respond(messages, info):
        turns.append(messages)
        if len(turns) == 1:
            return ModelResponse(parts=[ToolCallPart("query_artifact", {}, tool_call_id="bad")])
        if len(turns) == 2:
            assert any(
                isinstance(p, RetryPromptPart)
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "query_artifact",
                        {"ref_id": ref_id, "query": "invoice approval"},
                        tool_call_id="good",
                    )
                ]
            )
        assert any(
            isinstance(p, ToolReturnPart) and "manager approval" in str(p.content)
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
        )
        return ModelResponse(parts=[TextPart("manager")])

    adapter = PydanticAdapter(FunctionModel(respond), query_artifact=True)
    runtime = KediRuntime(adapter=adapter)
    manager = runtime.artifacts
    ref_id = manager.artifactize(
        "The invoice requires manager approval.", policy=ArtifactPolicy(threshold_bytes=1)
    ).ref.ref_id
    try:
        with runtime.bind_agent(profile=AgentProfile.empty()) as surface:
            with surface.run_scope():
                assert adapter.invoke_sync(prompt="Find the invoice approval rule") == "manager"
        assert len(turns) == 3
    finally:
        manager.close()


def test_query_validation_does_not_hide_tool_implementation_errors(stored, monkeypatch):
    from kedi.agent_adapter.adapters.pydantic.tools import _validated_tool_function

    manager, _ = stored
    spec = next(
        t for t in artifact_tool_specs(manager, query_artifact=True) if t.name == "query_artifact"
    )

    def broken(*args, **kwargs):
        raise TypeError("internal implementation defect")

    monkeypatch.setattr(manager, "query", broken)
    with pytest.raises(TypeError, match="internal implementation defect"):
        _validated_tool_function(spec)(ref_id="ref", query="needle")


def test_real_kedi_program_retrieves_answer_outside_preview():
    calls = []

    def respond(messages, info):
        assert "query_artifact" in {t.name for t in info.function_tools}
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("load_manual", {}, tool_call_id="load")])
        last = returns[-1]
        calls.append(last.tool_name)
        if last.tool_name == "load_manual":
            assert "Lale" not in str(last.content)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "query_artifact",
                        {
                            "ref_id": last.content["ref_id"],
                            "query": "verified release codename",
                        },
                        tool_call_id="query",
                    )
                ]
            )
        assert "VERIFIED_RELEASE_CODENAME=Lale" in str(last.content)
        return ModelResponse(parts=[TextPart("Lale")])

    runtime = compile_program(
        parse_program("""
> approval: allow
> artifacts:
  query_artifacts: enabled
  threshold: 32b
  preview_chars: 16

@load_manual() -> str:
  = `("background " * 10000) + "VERIFIED_RELEASE_CODENAME=Lale"`

> use: load_manual
[answer] << Find the verified release codename in the manual.
= <answer>
"""),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    try:
        assert runtime.run_main() == "Lale"
        assert calls == ["load_manual", "query_artifact"]
        assert sum(e.kind == "artifact_query" for e in runtime.artifacts.history.snapshot()) == 1
    finally:
        runtime.artifacts.close()
