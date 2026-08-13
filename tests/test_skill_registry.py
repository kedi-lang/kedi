from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import kedi.cli as cli_module
import kedi.skill_registry as skill_registry
from kedi.agent_profile.skills import MAX_SKILL_BYTES


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / ".kedi"
    monkeypatch.setenv("KEDI_HOME", str(home))
    return home


def _skill_source(tmp_path: Path, name: str = "review") -> Path:
    source = tmp_path / name
    source.mkdir()
    (source / "SKILL.md").write_text("Review carefully.\n", encoding="utf-8")
    return source


def test_install_skill_path_copies_only_root_skill_file(tmp_path: Path) -> None:
    source = _skill_source(tmp_path)
    (source / "ignored.txt").write_text("not installed", encoding="utf-8")

    installed = skill_registry.install_skill_path(source)

    assert installed.name == "review"
    assert (installed.root / "SKILL.md").read_text(encoding="utf-8") == "Review carefully.\n"
    assert not (installed.root / "ignored.txt").exists()


def test_install_skill_path_rejects_missing_symlink_and_oversized_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="root SKILL.md"):
        skill_registry.install_skill_path(tmp_path / "missing")

    external = _skill_source(tmp_path, "external") / "SKILL.md"
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "SKILL.md").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        skill_registry.install_skill_path(linked)

    oversized = _skill_source(tmp_path, "oversized")
    (oversized / "SKILL.md").write_text("x" * (MAX_SKILL_BYTES + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="content limit"):
        skill_registry.install_skill_path(oversized)


def test_install_skill_path_rejects_symlinked_root_directory(tmp_path: Path) -> None:
    source = _skill_source(tmp_path, "source")
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        skill_registry.install_skill_path(linked)


def test_install_skill_path_rejects_directory_non_utf8_and_invalid_name(tmp_path: Path) -> None:
    directory_file = tmp_path / "directory-file"
    (directory_file / "SKILL.md").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="root SKILL.md"):
        skill_registry.install_skill_path(directory_file)

    binary = _skill_source(tmp_path, "binary")
    (binary / "SKILL.md").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        skill_registry.install_skill_path(binary)

    invalid = _skill_source(tmp_path, ".hidden")
    with pytest.raises(ValueError, match="Invalid skill name"):
        skill_registry.install_skill_path(invalid)


def test_install_skill_repo_uses_credential_free_github_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_git(args: list[str], *, error_context: str) -> str:
        del error_context
        calls.append(args)
        if args[0] == "clone":
            checkout = Path(args[-1])
            checkout.mkdir(parents=True)
            (checkout / "SKILL.md").write_text("From GitHub.\n", encoding="utf-8")
            return ""
        return "abc123"

    monkeypatch.setattr(skill_registry, "_run_git", fake_git)

    installed = skill_registry.install_skill_repo("kedi-lang/reviewer")

    assert installed.name == "reviewer"
    assert installed.source_revision == "abc123"
    assert calls[0][0:5] == ["clone", "--depth", "1", "--filter=blob:none", "--no-tags"]
    assert calls[0][-2] == "https://github.com/kedi-lang/reviewer.git"


@pytest.mark.parametrize("source_kind", ["missing", "directory", "symlink"])
def test_install_skill_repo_validates_root_skill_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_kind: str,
) -> None:
    external = tmp_path / "external.md"
    external.write_text("external", encoding="utf-8")

    def fake_git(args: list[str], *, error_context: str) -> str:
        del error_context
        if args[0] != "clone":
            return "abc123"
        checkout = Path(args[-1])
        checkout.mkdir(parents=True)
        skill_file = checkout / "SKILL.md"
        if source_kind == "directory":
            skill_file.mkdir()
        elif source_kind == "symlink":
            skill_file.symlink_to(external)
        return ""

    monkeypatch.setattr(skill_registry, "_run_git", fake_git)
    error = ValueError if source_kind == "symlink" else FileNotFoundError
    with pytest.raises(error):
        skill_registry.install_skill_repo("kedi-lang/reviewer")


@pytest.mark.parametrize("repository", ["repo", "https://github.com/a/b", "a/b?token=x", "../repo"])
def test_install_skill_repo_rejects_non_owner_repository_forms(repository: str) -> None:
    with pytest.raises(ValueError, match="OWNER/REPOSITORY"):
        skill_registry.install_skill_repo(repository)


def test_cli_skills_add_path_and_help(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _skill_source(tmp_path)
    result = CliRunner().invoke(cli_module.main, ["skills", "add", "--path", str(source)])
    help_result = CliRunner().invoke(cli_module.main, ["skills", "--help"])

    assert result.exit_code == 0
    assert "Installed skill review" in result.output
    assert help_result.exit_code == 0
    assert "kedi skills add" in help_result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["skills", "-h"],
        ["skills", "add", "--help"],
        ["skills", "add", "-h"],
    ],
)
def test_cli_skills_help_aliases(arguments: list[str]) -> None:
    result = CliRunner().invoke(cli_module.main, arguments)

    assert result.exit_code == 0
    assert "kedi skills add" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["skills"],
        ["skills", "remove"],
        ["skills", "add"],
        ["skills", "add", "--repo"],
        ["skills", "add", "--unknown", "value"],
        ["skills", "add", "--repo", "a/b", "--path", "somewhere"],
        ["skills", "add", "--repo", "a/b", "--repo", "c/d"],
        ["skills", "add", "--path", "one", "--path", "two"],
    ],
)
def test_cli_skills_add_rejects_ambiguous_sources(arguments: list[str]) -> None:
    result = CliRunner().invoke(cli_module.main, arguments)

    assert result.exit_code == 2


def test_cli_skills_add_reports_install_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli_module.main,
        ["skills", "add", "--path", str(tmp_path / "missing")],
    )

    assert result.exit_code == 1
    assert "root SKILL.md" in result.output


def test_cli_skills_add_repo_reports_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = skill_registry.InstalledSkill(
        name="review",
        root=Path("/tmp/review"),
        source_revision="abc123",
    )
    monkeypatch.setattr(cli_module, "install_skill_repo", lambda _repository: installed)

    result = CliRunner().invoke(
        cli_module.main,
        ["skills", "add", "--repo", "kedi-lang/reviewer"],
    )

    assert result.exit_code == 0
    assert "abc123" in result.output


@pytest.mark.parametrize(
    "blocked_path",
    ["registry-parent", "registry", "skill", "file"],
)
def test_install_skill_rejects_symlinked_registry_destinations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    blocked_path: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("KEDI_HOME", str(home))
    source = _skill_source(tmp_path, "review")
    external = tmp_path / "external"
    external.mkdir()
    registry_parent = home / "registry"
    registry = registry_parent / "skills"
    skill = registry / "review"

    if blocked_path == "registry-parent":
        home.mkdir()
        registry_parent.symlink_to(external, target_is_directory=True)
    elif blocked_path == "registry":
        registry_parent.mkdir(parents=True)
        registry.symlink_to(external, target_is_directory=True)
    elif blocked_path == "skill":
        registry.mkdir(parents=True)
        skill.symlink_to(external, target_is_directory=True)
    else:
        skill.mkdir(parents=True)
        (skill / "SKILL.md").symlink_to(external / "SKILL.md")

    with pytest.raises(ValueError, match="symlink"):
        skill_registry.install_skill_path(source)


def test_install_skill_rejects_invalid_internal_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid skill name"):
        skill_registry._install_skill(".hidden", "content")
