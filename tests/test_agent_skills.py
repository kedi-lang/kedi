from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import kedi
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.skills import (
    MAX_SKILL_BYTES,
    MAX_SKILL_LIST_LIMIT,
    SkillStore,
    skill_tool_specs,
)
from kedi.agent_adapter.tool_override import with_approval, with_artifact_result
from kedi.artifacts import ArtifactManager, ArtifactPolicy, ArtifactStream
from kedi.agent_profile import AgentProfile
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


def _write_skill(root: Path, name: str, text: str = "Use concise steps.\n") -> Path:
    path = root / name
    path.mkdir(parents=True)
    skill_file = path / "SKILL.md"
    skill_file.write_text(text, encoding="utf-8")
    return skill_file


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


def test_native_approval_adapters_receive_unwrapped_skill_tools(tmp_path: Path) -> None:
    source_path = tmp_path / "app.kedi"
    source_path.write_text("> use: skills\n>> Return [answer].\n", encoding="utf-8")
    runtime = compile_program(
        parse_program(source_path.read_text(encoding="utf-8"), source_path=str(source_path)),
        adapter=MockAdapter(),
    )

    tools = runtime._resolve_agent_tool_specs(
        AgentProfile(skills=True, approval=ApprovalPolicy.deny()),
        object.__new__(PydanticAdapter),
    )

    assert [tool.name for tool in tools] == ["list_skills", "read_skill"]
    assert all(not hasattr(tool.fn, "__wrapped__") for tool in tools)


def test_single_line_use_skills_registers_skill_tools_and_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    monkeypatch.chdir(tmp_path)
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program("> system: Be concise.\n> use: skills\n>> Return [answer].\n= <answer>\n"),
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
    assert "<SKILL_INSTRUCTIONS>" in adapter.instruction_calls[0]


def test_file_local_skills_do_not_depend_on_process_current_directory(tmp_path: Path) -> None:
    class SkillReadingAdapter(MockAdapter):
        captured_tools: dict[str, Any]

        def produce_sync(self, **kwargs: Any) -> Any:
            self.captured_tools = {tool.name: tool for tool in self.scoped_tools}
            return super().produce_sync(**kwargs)

    project = tmp_path / "project"
    source_path = project / "app.kedi"
    source_path.parent.mkdir()
    source_path.write_text("> use: skills\n>> Return [answer].\n= <answer>\n", encoding="utf-8")
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
    source_path.write_text("> use: skills\n>> Return [answer].\n= <answer>\n", encoding="utf-8")
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


def test_profile_use_skills_enables_skill_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_skill(tmp_path / ".agents" / "skills", "research")
    monkeypatch.chdir(tmp_path)
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n"
            "    > use: skills\n\n"
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
            "    > use: skills\n\n"
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


def test_use_skills_requires_adapter_tool_registration() -> None:
    class NoToolAdapter:
        async def produce(self, **_kwargs: Any) -> object:
            raise AssertionError("tool validation must run before adapter production")

        def produce_sync(self, **_kwargs: Any) -> object:
            raise AssertionError("tool validation must run before adapter production")

    runtime = compile_program(
        parse_program("> use: skills\n>> Return [answer].\n"),
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
        assert "<SKILL_INSTRUCTIONS>" in (adapter.instruction_calls[0] or "")
    finally:
        kedi.reset_config()
