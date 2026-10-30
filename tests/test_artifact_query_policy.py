from importlib import import_module

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

import kedi.lsp.features.diagnostics as lsp_diagnostics_module
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_profile import AgentProfile
from kedi.artifacts import ArtifactManager, ArtifactPolicy, ArtifactPolicyError, artifact_tool_specs
from kedi.core.runtime import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError


def test_query_policy_defaults_overlay_and_cache_identity():
    base = ArtifactPolicy.patch({})
    assert not base.queries_enabled()
    assert base.queries_enabled(default=True)
    assert base.resolved().query_artifacts == "disabled"
    enabled = base.overlay(ArtifactPolicy.patch({"query_artifacts": "enabled"}))
    inherited = enabled.overlay(ArtifactPolicy.patch({"threshold": "8kb"}))
    disabled = inherited.overlay(ArtifactPolicy.patch({"query_artifacts": "disabled"}))
    assert inherited.queries_enabled() and inherited.threshold_bytes == 8000
    assert not disabled.queries_enabled(default=True)
    assert enabled.as_cache_key() != disabled.as_cache_key()
    assert ArtifactPolicy(threshold_bytes=8192).queries_enabled(default=True)


@pytest.mark.parametrize("value", [None, True, False, 1, "on", "yes", "", [], {}])
def test_query_policy_rejects_invalid_patch_values(value):
    with pytest.raises(ArtifactPolicyError, match="query_artifacts must be"):
        ArtifactPolicy.patch({"query_artifacts": value})


@pytest.mark.parametrize("value", [True, 1, "on", [], {}])
def test_query_policy_rejects_invalid_constructor_values(value):
    with pytest.raises(ArtifactPolicyError, match="query_artifacts must be"):
        ArtifactPolicy(query_artifacts=value)


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_artifact_tools_honor_policy_over_constructor_default(mode):
    manager = ArtifactManager()
    try:
        names = {
            tool.name
            for tool in artifact_tool_specs(
                manager, ArtifactPolicy(query_artifacts=mode), query_artifact=mode == "disabled"
            )
        }
        assert ("query_artifact" in names) == (mode == "enabled")
    finally:
        manager.close()


@pytest.mark.parametrize("name", ["pydantic", "langchain", "claude", "codex"])
@pytest.mark.parametrize("default", [False, True])
def test_all_adapters_use_scoped_query_policy_for_tools_and_instructions(name, default):
    cls = getattr(
        import_module(f"kedi.agent_adapter.adapters.{name}"),
        {
            "pydantic": "PydanticAdapter",
            "langchain": "LangChainAdapter",
            "claude": "ClaudeAdapter",
            "codex": "CodexAdapter",
        }[name],
    )
    runtime = KediRuntime(adapter=cls(query_artifact=default))
    try:
        for mode in (None, "enabled", "disabled", None):
            policy = ArtifactPolicy(threshold_bytes=8192, query_artifacts=mode)
            expected = default if mode is None else mode == "enabled"
            with runtime.bind_agent(profile=AgentProfile(artifacts=policy)) as surface:
                assert ("query_artifact" in {t.name for t in surface.tool_specs}) == expected
                assert ("query_artifact" in (surface.profile.system or "")) == expected
        with runtime.bind_agent(
            profile=AgentProfile(artifacts=ArtifactPolicy(enabled=False, query_artifacts="enabled"))
        ) as surface:
            assert "query_artifact" not in {t.name for t in surface.tool_specs}
            assert "query_artifact" not in (surface.profile.system or "")
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("literal", ["true", "false", "none", "on", "1"])
def test_query_directive_parser_and_lsp_reject_invalid_literals(literal):
    source = f"> artifacts:\n  query_artifacts: {literal}\n"
    with pytest.raises(KediParseError):
        parse_program(source)
    diagnostics = lsp_diagnostics_module.compute_diagnostics(source, "query-policy.kedi")
    assert [d.code for d in diagnostics] == ["K-ARTIFACTS-VALUE"]
    assert diagnostics[0].range.start.line == 1
    assert diagnostics[0].range.start.character == len("  query_artifacts: ")


@pytest.mark.parametrize("value", ["enabled", "disabled", "`True`", "`False`"])
def test_query_directive_accepts_literal_and_native_value(value):
    source = f"> artifacts:\n  query_artifacts: {value}\n"
    assert lsp_diagnostics_module.compute_diagnostics(source, "query-policy.kedi") == []
    parse_program(source)


def test_real_program_query_policy_profile_and_procedure_scoping():
    def respond(messages, info):
        enabled = "query_artifact" in {t.name for t in info.function_tools}
        return ModelResponse(parts=[TextPart("enabled" if enabled else "disabled")])

    runtime = compile_program(
        parse_program("""
> artifacts:
  query_artifacts: enabled

> profile: quiet:
  > artifacts:
    query_artifacts: disabled

@inner() -> str:
  > use: quiet
  [answer] << Inspect inner policy.
  = <answer>

@native() -> str:
  > artifacts:
    query_artifacts: `False`
  [answer] << Inspect native policy.
  = <answer>

[before] << Inspect outer policy.
[inside] = `inner()`
[dynamic] = `native()`
[after] << Inspect restored outer policy.
= `before, inside, dynamic, after`
"""),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )
    try:
        assert runtime.run_main() == ("enabled", "disabled", "disabled", "enabled")
    finally:
        runtime.artifacts.close()


def test_native_query_directive_validates_runtime_value():
    runtime = compile_program(parse_program('> artifacts:\n  query_artifacts: `"enabled"`\n'))
    try:
        with pytest.raises(Exception, match="query_artifacts.*boolean"):
            runtime.run_main()
    finally:
        runtime.artifacts.close()
