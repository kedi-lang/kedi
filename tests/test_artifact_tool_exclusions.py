from __future__ import annotations

import asyncio

import pytest

from kedi.agent_adapter.tools.execution import admit_artifact_result, admit_artifact_result_async
from kedi.agent_profile.directives import eval_artifacts_directive
from kedi.artifacts import ArtifactHandle, ArtifactManager, ArtifactPolicy, ArtifactStream
from kedi.artifacts.errors import ArtifactPolicyError, ToolOutputTooLargeError
from kedi.artifacts.policy import MAX_INLINE_TOOL_RESULT_BYTES
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ArtifactsDirectiveStmt


@pytest.fixture
def manager():
    value = ArtifactManager()
    try:
        yield value
    finally:
        value.close()


def test_exclusion_names_are_copied_normalized_and_part_of_cache_identity():
    names = ["read_document", "server.lookup", "read_document"]
    policy = ArtifactPolicy.patch({"exclude_tools": names})
    names.append("write_document")
    assert policy.exclude_tools == ("read_document", "server.lookup")
    assert policy.as_cache_key() != ArtifactPolicy().as_cache_key()
    assert (
        policy.overlay(ArtifactPolicy.patch({"threshold": 12})).exclude_tools
        == policy.exclude_tools
    )
    assert policy.overlay(ArtifactPolicy.patch({"exclude_tools": []})).exclude_tools == ()


@pytest.mark.parametrize(
    "value",
    [
        "read_document",
        None,
        True,
        {"read_document"},
        [1],
        [""],
        [" read"],
        ["read*"],
        ["read?"],
        ["read[0]"],
    ],
)
def test_exclusions_reject_nonlists_and_ambiguous_names(value):
    with pytest.raises(ArtifactPolicyError, match="exclude_tools"):
        ArtifactPolicy.patch({"exclude_tools": value})


def test_exclusion_matches_exact_exposed_name_and_keeps_trace(manager):
    policy = ArtifactPolicy(threshold_bytes=8, exclude_tools=("mcp_read_document",))
    result = "source evidence" * 10
    for name, excluded in [
        ("mcp_read_document", True),
        ("read_document", False),
        ("mcp_read_document_more", False),
    ]:
        admitted = admit_artifact_result(
            result, manager=manager, policy=policy, tool_name=name, call_id=name
        )
        if excluded:
            assert admitted == result
            event = manager.history.snapshot()[-1]
            assert event.data["artifact_excluded"] is True
            assert event.data["artifacted"] is False
            assert event.data["call_id"] == name
        else:
            assert admitted["ref_id"]
    assert policy.enabled is True


@pytest.mark.parametrize("transport", ["value", "stream", "async_stream"])
@pytest.mark.parametrize("oversized", [False, True])
def test_excluded_outputs_still_obey_inline_limit(manager, transport, oversized):
    payload = "x" * (MAX_INLINE_TOOL_RESULT_BYTES + 1 if oversized else 300)
    policy = ArtifactPolicy(threshold_bytes=1, exclude_tools=("lookup",))

    async def chunks():
        yield payload[:100]
        yield payload[100:]

    if transport == "async_stream":
        result = ArtifactStream.text(chunks())
    elif transport == "stream":
        result = ArtifactStream.text([payload[:100], payload[100:]])
    else:
        result = payload

    def invoke():
        kwargs = dict(manager=manager, policy=policy, tool_name="lookup", call_id="call_lookup")
        if transport == "async_stream":
            return asyncio.run(admit_artifact_result_async(result, **kwargs))
        return admit_artifact_result(result, **kwargs)

    if oversized:
        with pytest.raises(ToolOutputTooLargeError):
            invoke()
        assert manager.history.snapshot()[-1].data["error"] == "tool_output_too_large"
    else:
        assert invoke() == payload
        assert manager.history.snapshot()[-1].data["artifact_excluded"] is True
    assert not any(event.kind == "artifact_created" for event in manager.history.snapshot())


def test_tool_exclusions_do_not_disable_capture_artifacts(manager):
    policy = ArtifactPolicy(threshold_bytes=1, exclude_tools=("lookup",))
    handle = manager.artifactize("output capture", policy=policy, source="capture:summary")
    assert isinstance(handle, ArtifactHandle)
    assert handle.get_value() == "output capture"


def test_dsl_exclusion_list_is_evaluated_and_validated():
    source = '> artifacts:\n  threshold: 8b\n  exclude_tools: `["lookup"]`\n'
    program = parse_program(source)
    runtime = compile_program(program)
    statement = next(
        item for item in program.ordered_toplevel if isinstance(item, ArtifactsDirectiveStmt)
    )
    try:
        profile = eval_artifacts_directive(runtime, statement, {})
        assert profile.artifacts is not None
        assert profile.artifacts.exclude_tools == ("lookup",)
        assert profile.artifacts.threshold_bytes == 8
    finally:
        runtime.artifacts.close()
