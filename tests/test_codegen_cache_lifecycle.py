from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kedi.codegen.interfaces import CodegenConfig
from kedi.codegen.manager import CodegenManager
from kedi.codegen.mock_agents import MockPassingAgent


@pytest.mark.parametrize(
    "implementation, validated",
    [
        ("", False),
        ("", True),
        ("@other():\n  = ok\n", True),
        ("@f():\n  > auto:\n    Return ok.\n", True),
        ("@f(:\n", True),
        ("@f():\n  = ok\n", False),
    ],
)
def test_incomplete_or_legacy_block_is_not_cached(
    tmp_path: Path, implementation: str, validated: bool
) -> None:
    source = tmp_path / "program.kedi"
    source.write_text("= ok\n")
    manager = CodegenManager(MockPassingAgent())
    manager.write_cache(source, "f", "", implementation, validated=validated)
    assert not manager.is_cached(source, "f")


def test_cache_completion_marker_is_local_to_its_procedure(tmp_path: Path) -> None:
    source = tmp_path / "program.kedi"
    source.write_text("= ok\n")
    manager = CodegenManager(MockPassingAgent())
    manager.write_cache(source, "keep", "", "@keep():\n  = ok\n", validated=True)
    manager.write_cache(source, "f", "", "@f():\n  = ok\n")
    assert manager.is_cached(source, "keep")
    assert not manager.is_cached(source, "f")
    before = manager.cache_path_for(source).read_bytes()
    manager.cleanup_failed_cache(source, "missing")
    assert manager.cache_path_for(source).read_bytes() == before


def test_reversed_cache_markers_are_not_cached(tmp_path: Path) -> None:
    source = tmp_path / "program.kedi"
    source.write_text("= ok\n")
    manager = CodegenManager(MockPassingAgent())
    manager.cache_path_for(source).write_text("# <<< kedi-codegen: f\n# >>> kedi-codegen: f\n")
    assert not manager.is_cached(source, "f")


def test_generated_code_cannot_forge_completion_marker(tmp_path: Path) -> None:
    source = tmp_path / "program.kedi"
    manager = CodegenManager(MockPassingAgent())
    with pytest.raises(ValueError, match="reserved kedi-codegen"):
        manager.write_cache(source, "f", "# kedi-codegen-validated: f", "")
    assert not manager.cache_path_for(source).exists()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError])
@pytest.mark.parametrize("phase", ["tests", "implementation", "validation"])
def test_codegen_failure_removes_only_its_own_partial_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
    phase: str,
) -> None:
    source = tmp_path / "program.kedi"
    source.write_text("@f() -> str:\n  > auto:\n    Return ok.\n")
    manager = CodegenManager(MockPassingAgent(), CodegenConfig(max_retries=1))
    manager.write_cache(source, "keep", "", "@keep():\n  = ok\n", validated=True)
    manager.write_cache(source, "f", "@test: f:\n  > case: pending:\n    `assert True`\n", "")
    keep = manager.cache_path_for(source).read_text().split("# >>> kedi-codegen: f")[0]

    def fail(*args: object, **kwargs: object) -> str:
        raise failure("generation interrupted")

    target, attribute = {
        "tests": (manager.agent, "generate_tests"),
        "implementation": (manager.agent, "generate_impl"),
        "validation": (manager, "run_tests_and_get_errors"),
    }[phase]
    monkeypatch.setattr(target, attribute, fail)
    with pytest.raises(failure, match="generation interrupted"):
        manager.orchestrate_generation(source, "@f() -> str:", "Return ok.", "")
    assert manager.cache_path_for(source).read_text().strip() == keep.strip()
    assert manager.is_cached(source, "keep")
    assert not manager.is_cached(source, "f")


def test_partial_cache_is_regenerated_and_published_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "program.kedi"
    source.write_text("@f() -> str:\n  > auto:\n    Return ok.\n")
    manager = CodegenManager(MockPassingAgent(), CodegenConfig(max_retries=1))
    manager.write_cache(source, "f", "@test: f:\n  > case: pending:\n    `assert True`\n", "")
    validate = manager.run_tests_and_get_errors
    validations = []

    def validate_pending(path: Path) -> str | None:
        assert not manager.is_cached(path, "f")
        validations.append(path)
        return validate(path)

    monkeypatch.setattr(manager, "run_tests_and_get_errors", validate_pending)
    assert manager.orchestrate_generation(source, "@f() -> str:", "Return ok.", "")
    assert validations == [source]
    assert manager.is_cached(source, "f")
    assert manager.orchestrate_generation(source, "@f() -> str:", "Return ok.", "")
    assert validations == [source]


def test_completed_cache_needs_no_provider_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.codegen.runner import run_codegen

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "program.kedi"
    source.write_text("@f() -> str:\n  > auto:\n    Return ok.\n")
    manager = CodegenManager(MockPassingAgent())
    manager.write_cache(source, "f", "", "@f() -> str:\n  = ok\n", validated=True)
    before = manager.cache_path_for(source).read_bytes()
    assert run_codegen(source, model="openai/gpt-6-luna", quiet=True)
    assert manager.cache_path_for(source).read_bytes() == before
