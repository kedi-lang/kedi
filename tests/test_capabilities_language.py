from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from textwrap import indent
from threading import Event

import pytest

from kedi.agent_profile import AgentProfile, block_agent_profile_scope
from kedi.agent_profile.compile import profile_stmt_to_definition
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.agent_profile.scope import current_block_agent_profile
from kedi.engine import ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError
from tests.mock_adapter import MockAdapter


class CapabilityAdapter(MockAdapter):
    capabilities = replace(MockAdapter.capabilities, supports_capabilities=True)


def instruction_calls(source: str, *, adapter=None, source_path=None) -> list[str]:
    adapter = adapter or CapabilityAdapter()
    runtime = compile_program(parse_program(source, source_path=source_path), adapter=adapter)
    try:
        runtime.run_main()
        return adapter.instruction_calls
    finally:
        runtime.artifacts.close()


def test_eager_capabilities_append_reset_and_restore_without_duplicates():
    calls = instruction_calls(
        "> capability: citations:\n  > instructions: Cite sources.\n"
        "> capability: concise:\n  > instructions: Keep it brief.\n"
        "> instructions: Base rules.\n"
        "> use: citations\n> use: concise\n"
        ">> The summary is [summary: str].\n> show: <summary>\n"
        "> instructions: Base rules.\n"
        ">> The status is [status: str].\n> show: <status>\n"
        "> use: citations\n> use: citations\n"
        ">> The evidence is [evidence: str].\n= <evidence>\n"
    )
    assert len(calls) == 3
    assert all(call.count("Base rules.") == 1 for call in calls)
    assert calls[0].count("Cite sources.") == 1
    assert calls[0].count("Keep it brief.") == 1
    assert "Cite sources." not in calls[1] and "Keep it brief." not in calls[1]
    assert calls[2].count("Cite sources.") == 1 and "Keep it brief." not in calls[2]


@pytest.mark.parametrize(
    ("members", "has_citations"),
    [
        ("  > use: citations\n  > instructions: Review rules.\n", False),
        ("  > instructions: Review rules.\n  > use: citations\n", True),
    ],
)
def test_profile_source_order_controls_capability_reset(members, has_citations):
    calls = instruction_calls(
        "> capability: citations:\n  > instructions: Cite sources.\n"
        "> profile: reviewer:\n" + members + "> instructions: Old base.\n> use: reviewer\n"
        ">> The review is [review: str].\n= <review>\n"
    )
    assert "Old base." not in calls[0]
    assert "Review rules." in calls[0]
    assert ("Cite sources." in calls[0]) is has_citations


def test_capability_local_override_does_not_reset_host():
    calls = instruction_calls(
        "> capability: clear:\n"
        "  > instructions: Old local.\n  > instructions: New local.\n"
        "> instructions: Host rules.\n> use: clear\n"
        ">> The explanation is [explanation: str].\n= <explanation>\n"
    )
    assert "Host rules." in calls[0] and "New local." in calls[0]
    assert "Old local." not in calls[0]


def test_capability_diamond_keeps_each_contribution_once():
    calls = instruction_calls(
        "> capability: shared:\n  > instructions: Shared guidance.\n"
        "> capability: left:\n  > use: shared\n"
        "> capability: right:\n  > use: shared\n"
        "> capability: combined:\n  > use: left\n  > use: right\n"
        "> instructions: Host rules.\n> use: combined\n"
        ">> The result is [result: str].\n= <result>\n"
    )
    assert calls[0].count("Shared guidance.") == 1
    assert "Host rules." in calls[0]


@pytest.mark.parametrize("wrapper", ["", "> profile: wrapper:\n", "> capability: wrapper:\n"])
def test_use_block_composes_capabilities_in_order(wrapper):
    use = "> use:\n  citations\n  concise\n"
    if wrapper:
        use = wrapper + "\n".join("  " + line for line in use.splitlines()) + "\n> use: wrapper\n"
    calls = instruction_calls(
        "> capability: citations:\n  > instructions: Cite sources.\n"
        "> capability: concise:\n  > instructions: Keep it brief.\n"
        "> instructions: Host rules.\n" + use + ">> The summary is [summary: str].\n= <summary>\n"
    )
    assert (
        calls[0].index("Host rules.")
        < calls[0].index("Cite sources.")
        < calls[0].index("Keep it brief.")
    )


def test_local_procedure_override_preserves_outer_capability_instructions():
    calls = instruction_calls(
        "> capability: sources:\n  > instructions: Cite sources.\n"
        "> instructions: Host rules.\n> use: sources\n"
        "@review() -> str:\n"
        "  > instructions: Local review.\n"
        "  >> The review is [review: str].\n  = <review>\n"
        "> show: <review()>\n"
        ">> The status is [status: str].\n= <status>\n"
    )
    assert "Local review." in calls[0]
    assert "Host rules." not in calls[0] and "Cite sources." not in calls[0]
    assert "Host rules." in calls[1] and "Cite sources." in calls[1]
    assert "Local review." not in calls[1]


@pytest.mark.parametrize("scope", ["top-level", "procedure"])
def test_branches_and_loop_iterations_keep_capability_instruction_scopes_local(scope):
    body = (
        "> if: `False`:\n"
        "  > instructions: Unreachable guidance.\n"
        "> if: `True`:\n"
        "  > instructions: Branch guidance.\n  > use: concise\n"
        "  >> The branch result is [branch: str].\n  > show: <branch>\n"
        '> loop [label]: `["First iteration.", "Second iteration."]`:\n'
        "  >> The initial status is [initial: str].\n  > show: <initial>\n"
        "  > instructions: <label>\n  > use: concise\n"
        "  >> The iteration status is [iteration: str].\n  > show: <iteration>\n"
        ">> The final status is [final: str].\n> show: <final>\n"
    )
    if scope == "procedure":
        body = "@exercise() -> str:\n" + indent(body + "= done\n", "  ") + "> show: <exercise()>\n"
    calls = instruction_calls(
        "> capability: sources:\n  > instructions: Cite supplied sources.\n"
        "> capability: concise:\n  > instructions: Keep it brief.\n"
        "> instructions: Host guidance.\n> use: sources\n"
        + body
        + ">> The outer status is [outer: str].\n= <outer>\n"
    )
    expected = [
        ("Branch guidance.", "Keep it brief."),
        ("Host guidance.", "Cite supplied sources."),
        ("First iteration.", "Keep it brief."),
        ("Host guidance.", "Cite supplied sources."),
        ("Second iteration.", "Keep it brief."),
        ("Host guidance.", "Cite supplied sources."),
        ("Host guidance.", "Cite supplied sources."),
    ]
    guidance = {text for pair in expected for text in pair} | {"Unreachable guidance."}
    assert len(calls) == len(expected)
    for instructions, included in zip(calls, expected):
        assert {text for text in guidance if text in instructions} == set(included)
        assert all(instructions.count(text) == 1 for text in included)


def test_return_to_base_profile_resets_guidance_without_rolling_back_other_configuration():
    adapter = CapabilityAdapter()
    calls = instruction_calls(
        "@read_document() -> str:\n  = source evidence\n"
        "> capability: reader:\n  > instructions: Cite supplied sources.\n"
        "  > use: read_document\n"
        "> profile: base:\n  > instructions: Base guidance.\n"
        "> profile: reviewer:\n  > instructions: Reviewer guidance.\n  > effort: low\n"
        "> use: base\n> use: reviewer\n> use: reader\n"
        ">> The review is [review: str].\n> show: <review>\n"
        "> use: base\n"
        ">> The status is [status: str].\n= <status>\n",
        adapter=adapter,
    )
    assert len(calls) == 2
    assert "Reviewer guidance." in calls[0] and "Cite supplied sources." in calls[0]
    assert "Base guidance." not in calls[0]
    assert "Base guidance." in calls[1]
    assert "Reviewer guidance." not in calls[1] and "Cite supplied sources." not in calls[1]
    assert all("read_document" in names for names in adapter.tool_calls)
    assert adapter.effort_calls == ["low", "low"]


@pytest.mark.parametrize("error", [LookupError, TimeoutError, asyncio.CancelledError])
def test_local_instruction_reset_unwinds_on_failure(error):
    observed = []

    def fail():
        observed.append(current_block_agent_profile())
        raise error("scope failure")

    runtime = compile_program(
        parse_program(
            "> capability: concise:\n  > instructions: Keep it brief.\n"
            "> instructions: Definition guidance.\n> use: concise\n"
            "@attempt():\n  > instructions: Local guidance.\n  `fail()`\n"
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"fail": fail},
    )
    try:
        with block_agent_profile_scope(AgentProfile(system="Caller guidance.")):
            caller = current_block_agent_profile()
            with pytest.raises((KediExecutionError, error), match="scope failure"):
                runtime._invoke_procedure(runtime._get_proc("attempt"), [])
            assert current_block_agent_profile() is caller
        assert len(observed) == 1
        assert observed[0].system == "Local guidance."
        assert observed[0].capability_instructions == ()
        assert len(observed[0].capabilities) == 1
    finally:
        runtime.artifacts.close()


def test_instruction_reset_does_not_change_an_in_flight_template_snapshot():
    started, second, release = Event(), Event(), Event()

    class PausedAdapter(CapabilityAdapter):
        def produce_sync(self, *, template, output_schema=None, **kwargs):
            result = super().produce_sync(template=template, output_schema=output_schema, **kwargs)
            if "first" in output_schema:
                started.set()
                assert release.wait(5)
            else:
                second.set()
            return result

    adapter = PausedAdapter()
    engine = ThreadEngine(max_workers=2)
    runtime = compile_program(
        parse_program(
            "> capability: concise:\n  > instructions: Keep it brief.\n"
            "> instructions: Original guidance.\n> use: concise\n"
            ">> The first result is [first: str].\n`assert started.wait(5)`\n"
            "> instructions: Replacement guidance.\n"
            ">> The second result is [second: str].\n= <second>\n"
        ),
        adapter=adapter,
        engine=engine,
        runtime_globals={"started": started},
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            run = pool.submit(runtime.run_main)
            try:
                assert started.wait(5)
                assert second.wait(5)
                assert not run.done()
            finally:
                release.set()
            assert run.result(timeout=5) == "second-value"
        first_instructions, second_instructions = adapter.instruction_calls
        assert "Original guidance." in first_instructions
        assert "Keep it brief." in first_instructions
        assert "Replacement guidance." not in first_instructions
        assert "Replacement guidance." in second_instructions
        assert "Original guidance." not in second_instructions
        assert "Keep it brief." not in second_instructions
    finally:
        release.set()
        engine.shutdown()
        runtime.artifacts.close()


@pytest.mark.parametrize(
    "member",
    [
        "model: fixture",
        "effort: low",
        "agent: codex",
        "adapter: pydantic",
        "history: enabled",
        "skills: enabled",
        "codemode: enabled",
        "output: str",
        "max_agents: 2",
        "workflow: delegate",
        "artifacts:\n    enabled: false",
        "settings:\n    tool_reason: disabled",
        "budget:\n    request_limit: 1",
    ],
)
def test_capability_rejects_agent_owned_settings_at_source(member):
    with pytest.raises(KediParseError, match="not allowed in a capability") as exc:
        parse_program(f"> capability: restricted:\n  > {member}\n")
    assert any(d.code == "K-CAPABILITY-MEMBER" for d in exc.value.diagnostics)


def test_capability_rejects_user_prompt_submit():
    with pytest.raises(KediParseError, match="user_prompt_submit"):
        parse_program(
            "> capability: editor:\n  > hooks:\n    user_prompt_submit: `rewrite_prompt`\n"
        )


def test_capability_cannot_import_profile_overrides_via_use():
    with pytest.raises(KediExecutionError, match="cannot use profile"):
        instruction_calls(
            "> profile: host:\n  > model: fixture\n"
            "> capability: restricted:\n  > use: host\n"
            "> use: restricted\n= done\n"
        )


def test_capability_use_cycles_are_rejected():
    with pytest.raises(KediExecutionError, match="Cyclic"):
        instruction_calls(
            "> capability: left:\n  > use: right\n"
            "> capability: right:\n  > use: left\n"
            "> use: left\n= done\n"
        )


def test_deferred_capability_retains_catalog_without_eager_instructions():
    program = parse_program(
        "> capability: search:\n  > defer: true\n  > instructions: Search guidance.\n"
    )
    definition = profile_stmt_to_definition(program.profile_defs["search"])
    runtime = compile_program(parse_program("= done\n"), adapter=MockAdapter())
    try:
        patch = eval_profile_definition(runtime, definition, {})
        assert patch.system is None and patch.capability_instructions == ()
        assert patch.tools == () and patch.mcp_servers == ()
        assert patch.capabilities[0].deferred is True
        assert patch.capabilities[0].profile.system == "Search guidance."
        assert set(patch.requirements) == {"capabilities", "deferred_tools"}
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("value", ['"false"', "1", "None", "[]"])
def test_defer_expression_requires_a_real_bool(value):
    with pytest.raises(KediExecutionError, match="defer must evaluate to a bool"):
        instruction_calls(
            f"> capability: search:\n  > defer: `{value}`\n"
            "  > instructions: Search guidance.\n> use: search\n= done\n"
        )


def test_unsupported_adapter_rejects_before_model_call():
    adapter = MockAdapter()
    with pytest.raises(KediExecutionError, match="required capability 'capabilities'"):
        instruction_calls(
            "> capability: concise:\n  > instructions: Keep it brief.\n"
            "> use: concise\n>> The result is [result: str].\n= <result>\n",
            adapter=adapter,
        )
    assert adapter.instruction_calls == []


def test_imported_capability_keeps_instruction_identity(tmp_path: Path):
    module = tmp_path / "guidance.kedi"
    module.write_text(
        "> capability: citations:\n  > instructions: Cite imported sources.\n"
        "> export:\n  citations\n"
    )
    calls = instruction_calls(
        "> import: guidance\n> instructions: Host rules.\n> use: citations\n"
        ">> The result is [result: str].\n= <result>\n",
        source_path=str(tmp_path / "main.kedi"),
    )
    assert "Host rules." in calls[0] and "Cite imported sources." in calls[0]


@pytest.mark.parametrize("kind", ["capability", "profile"])
def test_imported_use_block_binds_private_capability_dependencies(tmp_path: Path, kind):
    module = tmp_path / "bundle.kedi"
    module.write_text(
        "> capability: _citations:\n  > instructions: Cite private sources.\n"
        f"> {kind}: review_bundle:\n"
        "  > use:\n    _citations\n"
        "> export:\n  review_bundle\n"
    )
    calls = instruction_calls(
        "> import: bundle\n> instructions: Host rules.\n> use: review_bundle\n"
        ">> The review is [review: str].\n= <review>\n",
        source_path=str(tmp_path / "main.kedi"),
    )
    assert "Host rules." in calls[0] and "Cite private sources." in calls[0]


def test_imported_capabilities_reject_different_private_tools_with_the_same_name(tmp_path):
    for name in ("reader", "writer"):
        (tmp_path / f"{name}.kedi").write_text(
            f"@access() -> str:\n  = {name}\n"
            f"> capability: {name}:\n  > use: access\n"
            f"> export:\n  {name}\n",
            encoding="utf-8",
        )
    adapter = CapabilityAdapter()
    runtime = compile_program(
        parse_program(
            "> import: reader\n> import: writer\n"
            "> use: reader\n> use: writer\n"
            ">> The revision is [revision: str].\n= <revision>\n",
            source_path=str(tmp_path / "main.kedi"),
        ),
        adapter=adapter,
    )
    try:
        with pytest.raises(
            KediExecutionError, match="Conflicting capability tool binding"
        ) as error:
            runtime.run_main()
        assert "reader.kedi::reader" in str(error.value)
        assert "writer.kedi::writer" in str(error.value)
        assert adapter.instruction_calls == []
    finally:
        runtime.artifacts.close()


def test_imported_capabilities_can_share_the_same_private_procedure(tmp_path):
    (tmp_path / "bundle.kedi").write_text(
        "@read_document() -> str:\n  = revision 7\n"
        "> capability: evidence:\n  > use: read_document\n"
        "> capability: citations:\n  > use: read_document\n"
        "> export:\n  evidence\n  citations\n",
        encoding="utf-8",
    )
    adapter = CapabilityAdapter()
    calls = instruction_calls(
        "> import: bundle\n> use: evidence\n> use: citations\n"
        ">> The revision is [revision: str].\n= <revision>\n",
        adapter=adapter,
        source_path=str(tmp_path / "main.kedi"),
    )
    assert len(calls) == 1
