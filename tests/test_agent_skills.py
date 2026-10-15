from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

import kedi
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.skills import (
    MAX_SKILL_LIST_LIMIT,
    SkillStore,
    resolve_skill_store,
    skill_tool_specs,
)
from kedi.agent_adapter.tools.execution import with_approval, with_artifact_result
from kedi.agent_profile import AgentProfile, eval_skills_directive
from kedi.agent_profile.skills import (
    MAX_SKILL_BYTES,
    MAX_SKILLS,
    SkillsSettings,
    SkillsSettingsError,
    normalize_skills_settings,
    skills_settings,
)
from kedi.artifacts import ArtifactManager, ArtifactPolicy, ArtifactStream
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProfileDefStmt, SkillsDirectiveStmt
from kedi.lang.parser.diagnostics import KediParseError
from tests.mock_adapter import MockAdapter


def _write_skill(root: Path, name: str, text: str = "Use concise steps.\n") -> Path:
    path = root / name
    path.mkdir(parents=True)
    skill_file = path / "SKILL.md"
    skill_file.write_text(text, encoding="utf-8")
    return skill_file


@pytest.fixture(autouse=True)
def _isolated_skill_homes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("KEDI_HOME", str(home / ".kedi"))
    assert Path.home() == home


def test_skill_store_lists_and_reads_plain_project_local_skills(tmp_path: Path) -> None:
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "zeta")
    _write_skill(root, "alpha", "Follow the alpha workflow.\n")
    (root / "not-a-skill").write_text("ignored", encoding="utf-8")

    store = SkillStore(root)

    assert store.list_skills() == ["alpha", "zeta"]
    assert store.list_skills(all=True, limit=1) == ["alpha"]
    assert store.read_skill(skill_name="alpha") == "Follow the alpha workflow.\n"


def test_skill_store_for_current_project_uses_the_current_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "local")
    monkeypatch.chdir(tmp_path)

    assert SkillStore.for_current_project().list_skills() == ["local"]


@pytest.mark.parametrize("max_skills", [0, True, MAX_SKILLS + 1])
def test_skill_store_rejects_invalid_maximum(tmp_path: Path, max_skills: object) -> None:
    with pytest.raises(ValueError, match="max_skills"):
        SkillStore(tmp_path, max_skills=max_skills)  # type: ignore[arg-type]


def test_skill_store_handles_missing_root_and_non_utf8_content(tmp_path: Path) -> None:
    assert SkillStore(tmp_path / "missing").list_skills() == []

    root = tmp_path / ".agents" / "skills"
    bad_file = _write_skill(root, "binary")
    bad_file.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="UTF-8"):
        SkillStore(root).read_skill(skill_name="binary")
    with pytest.raises(ValueError, match="UTF-8"):
        list(SkillStore(root).stream_skill(skill_name="binary").iter_chunks())
    assert SkillStore(root).list_skills() == []


def test_skill_store_skips_unreadable_skills_but_direct_reads_still_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / ".agents" / "skills"
    _write_skill(root, "readable")
    unreadable = _write_skill(root, "locked")
    original_open = Path.open

    def deny_locked_skill(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == unreadable:
            raise PermissionError("permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_locked_skill)
    store = SkillStore(root)

    assert store.list_skills() == ["readable"]
    with pytest.raises(PermissionError, match="permission denied"):
        store.read_skill(skill_name="locked")


@pytest.mark.parametrize("limit", [0, -1, True, MAX_SKILL_LIST_LIMIT + 1])
def test_skill_store_rejects_invalid_list_limits(tmp_path: Path, limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        SkillStore(tmp_path).list_skills(limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "../outside", "/tmp/outside", "nested/name", ".hidden"])
def test_skill_store_rejects_invalid_or_missing_names(tmp_path: Path, name: str) -> None:
    store = SkillStore(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        store.read_skill(skill_name=name)


def test_skill_store_rejects_missing_skill_file_and_directory_at_file_path(tmp_path: Path) -> None:
    root = tmp_path / ".agents" / "skills"
    root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="not found"):
        SkillStore(root).read_skill(skill_name="missing")

    (root / "directory-file" / "SKILL.md").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="not found"):
        SkillStore(root).read_skill(skill_name="directory-file")


def test_skill_store_rejects_symlink_escape_and_oversized_content(tmp_path: Path) -> None:
    root = tmp_path / ".agents" / "skills"
    outside = tmp_path / "outside"
    _write_skill(outside, "external")
    root.mkdir(parents=True)
    (root / "escape").symlink_to(outside / "external", target_is_directory=True)
    oversized = _write_skill(root, "oversized", "x" * (MAX_SKILL_BYTES + 1))
    assert oversized.stat().st_size > MAX_SKILL_BYTES

    store = SkillStore(root)
    assert store.list_skills() == []
    with pytest.raises(ValueError, match="outside"):
        store.read_skill(skill_name="escape")
    with pytest.raises(ValueError, match="content limit"):
        store.read_skill(skill_name="oversized")


def test_skill_tools_are_read_only_and_bypass_dynamic_approval(tmp_path: Path) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    list_spec, read_spec = skill_tool_specs(tmp_path / ".agents" / "skills")
    approval_calls: list[Any] = []

    def deny_everything(request: Any) -> ApprovalDecision:
        approval_calls.append(request)
        return ApprovalDecision.deny(reason="unreachable")

    protected_list = with_approval(
        list_spec,
        policy=ApprovalPolicy.dynamic(deny_everything),
        adapter_shortname="mock",
    )
    protected_read = with_approval(
        read_spec,
        policy=ApprovalPolicy.dynamic(deny_everything),
        adapter_shortname="mock",
    )

    assert protected_list.fn() == ["research"]
    streamed = protected_read.fn(skill_name="research")
    assert isinstance(streamed, ArtifactStream)
    assert list(streamed.iter_chunks()) == ["Use concise steps.\n"]
    assert approval_calls == []


def test_read_skill_tool_streams_into_artifact_store_without_changing_logical_type(
    tmp_path: Path,
) -> None:
    content = "Follow the research workflow.\n" * 3_000
    _write_skill(tmp_path / ".agents" / "skills", "research", content)
    _, read_spec = skill_tool_specs(tmp_path / ".agents" / "skills")
    manager = ArtifactManager(session_id="skill-stream")
    try:
        wrapped = with_artifact_result(
            read_spec,
            manager=manager,
            policy=ArtifactPolicy.patch({"threshold": "1kb"}).resolved(),
        )

        result = wrapped.fn(skill_name="research")

        assert result["ref_id"] == "tool_call_result_1"
        assert manager.get_value(result["ref_id"]) == content
        assert read_spec.return_schema == {"type": "string"}
    finally:
        manager.close()


def test_native_approval_adapters_receive_skill_tools_without_approval_wrappers(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "app.kedi"
    source_path.write_text("> skills: enabled\n>> Return [answer].\n", encoding="utf-8")
    runtime = compile_program(
        parse_program(source_path.read_text(encoding="utf-8"), source_path=str(source_path)),
        adapter=MockAdapter(),
    )

    tools = runtime._resolve_agent_tool_specs(
        AgentProfile(skills=True, approval=ApprovalPolicy.deny()),
        object.__new__(PydanticAdapter),
    )

    assert [tool.name for tool in tools] == ["list_skills", "read_skill"]
    assert tools[0].fn() == []
    assert all((tool.metadata or {}).get("kedi_telemetry_wrapped") is True for tool in tools)


def test_skills_directive_registers_skill_tools_and_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    monkeypatch.chdir(tmp_path)
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(
            "> instructions: Be concise.\n> skills: enabled\n>> Return [answer].\n= <answer>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "answer-value"
    assert adapter.tool_calls == [
        (
            "list_skills",
            "read_skill",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]
    assert adapter.instruction_calls[0] is not None
    assert "Be concise." in adapter.instruction_calls[0]
    assert "<SkillInstructions>" in adapter.instruction_calls[0]
    assert "research" in adapter.instruction_calls[0]
    assert "Use concise steps." not in adapter.instruction_calls[0]
    assert (
        "Use supplied task-specific facts rather than assumptions" in adapter.instruction_calls[0]
    )


def test_skill_inventory_is_sorted_bounded_and_does_not_read_hidden_roots(tmp_path: Path):
    from kedi.agent_adapter.skills import skill_instructions

    _write_skill(tmp_path, "zeta", "SECRET_CONTENT_SENTINEL")
    _write_skill(tmp_path, "alpha", "SECRET_CONTENT_SENTINEL")
    text = skill_instructions(store=SkillStore(tmp_path, max_skills=1))
    assert "alpha" in text
    assert "zeta" not in text
    assert "SECRET_CONTENT_SENTINEL" not in text
    assert text == skill_instructions(store=SkillStore(tmp_path, max_skills=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("query_artifact", [False, True])
async def test_runtime_instruction_xml_uses_sibling_pascal_case_sections(
    tmp_path: Path, query_artifact
):
    from kedi.agent_adapter.request import KediTemplatePromptBuilder

    _write_skill(tmp_path / ".agents" / "skills", "research")
    profile = AgentProfile(
        skills=SkillsSettings(enabled=True, cwd=tmp_path, include_registry=False),
        artifacts=ArtifactPolicy(query_artifacts="enabled" if query_artifact else "disabled"),
    )
    runtime = KediRuntime(adapter=MockAdapter())
    try:
        parts = (
            *KediTemplatePromptBuilder(
                template="The city is [city].", output_fields=("city",)
            ).instruction_parts(),
            *runtime._agent_instruction_parts(profile, runtime.adapter),
        )
        root = ElementTree.fromstring(
            "<Instructions>" + "\n".join(part.content for part in parts) + "</Instructions>"
        )
        assert [child.tag for child in root] == [
            "TemplateInstructions",
            "ToolCallInstructions",
            "SkillInstructions",
            "ArtifactInstructions",
        ]
        assert all(re.fullmatch(r"[A-Z][A-Za-z0-9]*", node.tag) for node in root.iter())
        assert [node.text for node in root.findall("SkillInstructions/AvailableSkills/Skill")] == [
            "research"
        ]
        contract = root.findtext("ArtifactInstructions/Contract")
        assert contract is not None
        assert ("Use query_artifact" in contract) is query_artifact
        assert "Do not repeat the source tool" in contract
    finally:
        await runtime.aclose()


def test_grounding_instruction_is_only_attached_to_evidence_enabled_profiles():
    from kedi.agent_adapter.tools.specs import ToolSpec

    runtime = KediRuntime(adapter=MockAdapter())
    assert "grounding.instructions" not in {
        p.identity for p in runtime._agent_instruction_parts(AgentProfile(), runtime.adapter)
    }
    tool = ToolSpec(fn=lambda: "answer", name="lookup", risk="read_only")
    profile = AgentProfile(tools=("lookup",), tool_bindings={"lookup": tool})
    parts = runtime._agent_instruction_parts(profile, runtime.adapter)
    assert sum(p.identity == "grounding.instructions" for p in parts) == 1
    grounding = next(p.content for p in parts if p.identity == "grounding.instructions")
    assert grounding.startswith("<ToolCallInstructions>\n")
    assert grounding.endswith("\n</ToolCallInstructions>")
    assert "Respect explicit tool requirements and permissions" in grounding
    assert "<TemplateInstructions>" not in grounding


def test_skill_tool_without_artifact_boundary_returns_serializable_text(tmp_path: Path):
    from pydantic import TypeAdapter

    _write_skill(tmp_path / ".agents" / "skills", "example", "Use verified evidence.")
    profile = AgentProfile(
        skills=SkillsSettings(enabled=True, cwd=tmp_path, include_registry=False)
    )
    runtime = KediRuntime(adapter=MockAdapter())
    tools = runtime._resolve_agent_tool_specs(profile, object.__new__(PydanticAdapter))
    read = next(tool for tool in tools if tool.name == "read_skill")
    value = read.fn(skill_name="example")
    assert value == "Use verified evidence."
    assert TypeAdapter(str).dump_json(value) == b'"Use verified evidence."'


def test_skills_disabled_removes_an_inherited_skill_policy(tmp_path: Path) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(
            "> skills: enabled\n"
            "@without_skills():\n"
            "    > skills: disabled\n"
            "    >> Return [answer].\n"
            "    = <answer>\n\n"
            "= `without_skills()`\n",
            source_path=str(tmp_path / "main.kedi"),
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "answer-value"
    assert adapter.tool_calls == [
        ("read_artifact", "search_artifacts", "release_artifact", "run_artifact_code")
    ]


def test_file_local_skills_do_not_depend_on_process_current_directory(tmp_path: Path) -> None:
    class SkillReadingAdapter(MockAdapter):
        captured_tools: dict[str, Any]

        def produce_sync(self, **kwargs: Any) -> Any:
            self.captured_tools = {tool.name: tool for tool in self.scoped_tools}
            return super().produce_sync(**kwargs)

    project = tmp_path / "project"
    source_path = project / "app.kedi"
    source_path.parent.mkdir()
    source_path.write_text("> skills: enabled\n>> Return [answer].\n= <answer>\n", encoding="utf-8")
    _write_skill(project / ".agents" / "skills", "local")

    adapter = SkillReadingAdapter()
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = compile_program(
            parse_program(source_path.read_text(encoding="utf-8"), source_path=str(source_path)),
            adapter=adapter,
        )
        assert runtime.run_main() == "answer-value"
    finally:
        os.chdir(previous_cwd)

    assert adapter.captured_tools["list_skills"].fn() == ["local"]


def test_python_bind_uses_the_bound_file_skill_root(tmp_path: Path) -> None:
    class SkillReadingAdapter(MockAdapter):
        captured_tools: dict[str, Any]

        def produce_sync(self, **kwargs: Any) -> Any:
            self.captured_tools = {tool.name: tool for tool in self.scoped_tools}
            return super().produce_sync(**kwargs)

    source_path = tmp_path / "bound.kedi"
    source_path.write_text("> skills: enabled\n>> Return [answer].\n= <answer>\n", encoding="utf-8")
    _write_skill(tmp_path / ".agents" / "skills", "bound-local")
    adapter = SkillReadingAdapter()
    kedi.configure(adapter=adapter)
    try:

        @kedi.bind(file=source_path, skills=True)
        def answer() -> str:
            raise AssertionError("the Kedi file is the implementation")

        assert answer() == "answer-value"
    finally:
        kedi.reset_config()

    assert adapter.captured_tools["list_skills"].fn() == ["bound-local"]


def test_profile_skills_directive_enables_skill_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    monkeypatch.chdir(tmp_path)
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n"
            "    > skills: enabled\n\n"
            "> use: researcher\n"
            ">> Return [answer].\n"
            "= <answer>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "answer-value"
    assert adapter.tool_calls == [
        (
            "list_skills",
            "read_skill",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]


def test_profile_can_combine_skills_with_named_procedure_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    monkeypatch.chdir(tmp_path)
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(
            "@lookup() -> str:\n"
            "    = ok\n\n"
            "> profile: researcher:\n"
            "    > use: lookup\n"
            "    > skills: enabled\n\n"
            "> use: researcher\n"
            ">> Return [answer].\n"
            "= <answer>\n"
        ),
        adapter=adapter,
    )

    assert runtime.run_main() == "answer-value"
    assert adapter.tool_calls == [
        (
            "lookup",
            "list_skills",
            "read_skill",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]


def test_procedure_named_skills_keeps_procedure_precedence() -> None:
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program("@skills():\n    = done\n\n> use: skills\n>> Return [answer].\n"),
        adapter=adapter,
    )

    runtime.run_main()
    assert adapter.tool_calls == [
        (
            "skills",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]


def test_skills_directive_requires_adapter_tool_registration() -> None:
    class NoToolAdapter:
        async def produce(self, **_kwargs: Any) -> object:
            raise AssertionError("tool validation must run before adapter production")

        def produce_sync(self, **_kwargs: Any) -> object:
            raise AssertionError("tool validation must run before adapter production")

    runtime = compile_program(
        parse_program("> skills: enabled\n>> Return [answer].\n"),
        adapter=NoToolAdapter(),
    )

    with pytest.raises(KediExecutionError, match="skill tool registration"):
        runtime.run_main()


def test_python_api_skills_option_registers_skill_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    monkeypatch.chdir(tmp_path)
    adapter = MockAdapter()
    kedi.configure(adapter=adapter, skills=True)
    try:

        @kedi.query
        def answer() -> str:
            """kedi
            >> Return [answer].
            = <answer>
            """

        assert answer() == "answer-value"
        assert adapter.tool_calls == [
            (
                "list_skills",
                "read_skill",
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            )
        ]
        assert "<SkillInstructions>" in (adapter.instruction_calls[0] or "")
    finally:
        kedi.reset_config()


def test_python_api_accepts_expanded_skills_settings(tmp_path: Path) -> None:
    custom_root = tmp_path / "custom" / ".agents" / "skills"
    _write_skill(custom_root, "research")
    adapter = MockAdapter()
    kedi.configure(
        adapter=adapter,
        skills=SkillsSettings(
            enabled=True,
            cwd=tmp_path / "custom",
            include_registry=False,
        ),
    )
    try:

        @kedi.query
        def answer() -> str:
            """kedi
            >> Return [answer].
            = <answer>
            """

        assert answer() == "answer-value"
    finally:
        kedi.reset_config()

    assert adapter.tool_calls[0][0:2] == ("list_skills", "read_skill")


def test_skill_sources_follow_registry_local_global_priority_and_include_all(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "home" / ".kedi" / "registry" / "skills"
    local = tmp_path / "project" / ".agents" / "skills"
    global_root = tmp_path / "home" / ".agents" / "skills"
    _write_skill(registry, "registry")
    _write_skill(registry, "shared", "registry copy\n")
    _write_skill(local, "local")
    _write_skill(local, "shared", "local copy\n")
    _write_skill(global_root, "global")

    first = resolve_skill_store(SkillsSettings(enabled=True), base_dir=tmp_path / "project")
    merged = resolve_skill_store(
        SkillsSettings(enabled=True, include_all=True, max_skills=100),
        base_dir=tmp_path / "project",
    )

    assert first.list_skills() == ["registry", "shared"]
    assert merged.list_skills(limit=100) == ["global", "local", "registry", "shared"]
    assert merged.read_skill(skill_name="shared") == "registry copy\n"


def test_skills_settings_support_cwd_exclusions_and_maximum(tmp_path: Path) -> None:
    project = tmp_path / "project"
    selected = project / "nested" / ".agents" / "skills"
    _write_skill(selected, "alpha")
    _write_skill(selected, "beta")
    store = resolve_skill_store(
        SkillsSettings(
            enabled=True,
            cwd=Path("nested"),
            include_registry=False,
            include_all=True,
            exclude_paths=(Path("nested/.agents/skills/beta"),),
            max_skills=1,
        ),
        base_dir=project,
    )

    assert store.list_skills(limit=100) == ["alpha"]
    with pytest.raises(FileNotFoundError, match="not found"):
        store.read_skill(skill_name="beta")


def test_skills_source_fallback_skips_an_empty_registry(tmp_path: Path) -> None:
    registry = tmp_path / "home" / ".kedi" / "registry" / "skills"
    registry.mkdir(parents=True)
    local = tmp_path / "project" / ".agents" / "skills"
    _write_skill(local, "local")

    store = resolve_skill_store(SkillsSettings(enabled=True), base_dir=tmp_path / "project")

    assert store.list_skills() == ["local"]


def test_skill_store_exclusion_applies_to_direct_reads(tmp_path: Path) -> None:
    root = tmp_path / ".agents" / "skills"
    skill = _write_skill(root, "private")
    store = SkillStore(root, exclude_paths=(skill.parent,))

    with pytest.raises(FileNotFoundError, match="not found"):
        store.read_skill(skill_name="private")


def test_skills_directive_parses_in_all_agent_scopes() -> None:
    program = parse_program(
        "> skills: enabled\n\n"
        "> profile: researcher:\n"
        "    > skills: disabled\n\n"
        "@ask():\n"
        "    > skills: enabled\n"
        "    = done\n"
    )

    assert isinstance(program.ordered_toplevel[0], SkillsDirectiveStmt)
    assert program.ordered_toplevel[0].enabled is True
    profile = program.ordered_toplevel[1]
    assert isinstance(profile, ProfileDefStmt)
    assert profile.skills_members[0].enabled is False


def test_expanded_skills_directive_evaluates_settings() -> None:
    program = parse_program(
        "> skills:\n"
        "    enabled: true\n"
        "    cwd: workspace\n"
        "    max_skills: 7\n"
        "    include_registry: false\n"
        "    include_all: true\n"
        '    exclude_paths: `["ignored", "~/.agents/skills"]`\n'
    )
    directive = program.ordered_toplevel[0]
    assert isinstance(directive, SkillsDirectiveStmt)
    settings = eval_skills_directive(KediRuntime(), directive, {}).skills

    assert settings == SkillsSettings(
        enabled=True,
        cwd=Path("workspace"),
        max_skills=7,
        include_registry=False,
        include_all=True,
        exclude_paths=(Path("ignored"), Path.home() / ".agents" / "skills"),
    )


@pytest.mark.parametrize(
    "value",
    [
        "enabled",
        1,
    ],
)
def test_skills_settings_rejects_non_public_values(value: object) -> None:
    with pytest.raises(SkillsSettingsError, match="SkillsSettings"):
        skills_settings(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_skills", [0, True])
def test_skills_settings_rejects_invalid_maximum(max_skills: object) -> None:
    with pytest.raises(SkillsSettingsError, match="positive integer"):
        normalize_skills_settings({"max_skills": max_skills}, enabled=True)


def test_skills_settings_rejects_excessive_maximum() -> None:
    with pytest.raises(SkillsSettingsError, match=f"exceed {MAX_SKILLS}"):
        normalize_skills_settings({"max_skills": MAX_SKILLS + 1}, enabled=True)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"include_registry": "yes"}, "include_registry must be true or false"),
        ({"cwd": 42}, "cwd must be a path string"),
        ({"exclude_paths": "private"}, "exclude_paths must be a list"),
        ({"exclude_paths": [None]}, "exclude_paths entries must be path strings"),
    ],
)
def test_skills_settings_rejects_invalid_field_types(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(SkillsSettingsError, match=message):
        normalize_skills_settings(values, enabled=True)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("active", "must be 'enabled' or 'disabled'"),
        ("enabled: maybe", "must be 'true' or 'false'"),
        ("include_all: maybe", "must be 'true' or 'false'"),
        ("unknown: true", "skills setting must be one of"),
        ("max_skills: 1\n    max_skills: 2", "Duplicate skills setting"),
        ("enabled: `True`", "must be the literal"),
        ("max_skills: 1", "requires an 'enabled"),
    ],
)
def test_expanded_skills_directive_rejects_invalid_fields(body: str, message: str) -> None:
    if "\n" not in body and ":" not in body:
        source = f"> skills: {body}\n"
    else:
        indented = "\n".join(f"    {line}" for line in body.splitlines())
        source = f"> skills:\n{indented}\n"
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


def test_expanded_skills_directive_ignores_comments() -> None:
    program = parse_program("> skills:\n    # source policy\n    enabled: true\n")

    directive = program.ordered_toplevel[0]
    assert isinstance(directive, SkillsDirectiveStmt)
    assert directive.enabled is True


def test_legacy_use_skills_is_not_a_compatibility_alias() -> None:
    runtime = compile_program(parse_program("> use: skills\n= done\n"), adapter=MockAdapter())

    with pytest.raises(KediExecutionError, match="Unknown use target: 'skills'"):
        runtime.run_main()


def test_skill_instructions_without_store_do_not_claim_an_inventory() -> None:
    from kedi.agent_adapter.skills import skill_instructions

    instructions = skill_instructions()
    assert "Use list_skills when a task may benefit" in instructions
    assert "available_skills" not in instructions
