from __future__ import annotations

import builtins
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

import kedi.decisions as surface
from kedi.agent_profile import AgentProfile
from kedi.lang import compile_program, parse_program

PRIMITIVES = {"BooleanCriteria", "ChoiceCriteria", "Probability", "Rubric"}


def test_decision_exports_are_shared_package_primitives() -> None:
    import kedi_decisions as package

    assert PRIMITIVES <= set(surface.__all__)
    for name in PRIMITIVES:
        assert getattr(surface, name) is getattr(package, name)
    assert surface.decision_info is not None
    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(surface, "does_not_exist")


@pytest.mark.parametrize("names", [None, ["Probability", "Rubric"]])
def test_decisions_module_import_does_not_select_model(names) -> None:
    directive = "> import: decisions\n"
    if names is not None:
        directive = "> import: decisions:\n" + "".join(f"  {name}\n" for name in names)
    profile = AgentProfile(model="openai:unchanged")
    runtime = compile_program(
        parse_program(directive + "= imported\n"), default_agent_profile=profile
    )
    assert runtime.run_main() == "imported"
    assert runtime.default_agent_profile.model == profile.model
    for name in PRIMITIVES:
        if names is None or name in names:
            assert runtime._import_env[name] is getattr(surface, name)
        else:
            assert name not in runtime._import_env
    assert "decision_info" not in runtime._import_env


def test_decision_types_validate_native_annotations() -> None:
    runtime = compile_program(
        parse_program(
            "> import: decisions\n"
            "~Review(probability: Probability, quality: Annotated[float, "
            '"Answer quality", `Rubric(["Incorrect", "Correct"])`])\n'
            "= `Review`\n"
        )
    )
    review = runtime.run_main()
    assert review(probability=0.9, quality=1).probability == 0.9
    with pytest.raises(ValidationError):
        review(probability=1.1, quality=1)
    with pytest.raises(ValidationError):
        review(probability=0.9, quality=2)


def test_decision_exports_are_visible_to_editor_without_execution() -> None:
    from kedi.lsp.module_surface import load_imported_surface
    from kedi.module_registry import resolve_module_path

    imported = load_imported_surface(
        resolve_module_path("decisions", source_path=None), set(), "decisions"
    )
    assert imported.exported_names() == PRIMITIVES


@pytest.mark.parametrize("missing_name", ["kedi_decisions", "jsonschema"])
def test_criteria_import_errors_preserve_transitive_failures(
    monkeypatch: pytest.MonkeyPatch, missing_name: str
) -> None:
    original_import = builtins.__import__
    missing = ModuleNotFoundError("Dependency missing", name=missing_name)

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "kedi_decisions":
            raise missing
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ModuleNotFoundError) as error:
        surface.Probability
    assert error.value.name == missing_name
    if missing_name == "kedi_decisions":
        assert "uv add kedi-decisions" in str(error.value)
        assert error.value.__cause__ is missing
    else:
        assert error.value is missing


def test_dsl_import_reports_missing_shared_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
from kedi.lang import compile_program, parse_program

original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "kedi_decisions":
        raise ModuleNotFoundError("Missing optional dependency", name=name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
compile_program(parse_program("> import: decisions\\n= imported\\n"))
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "uv add kedi-decisions" in result.stderr


@pytest.mark.parametrize("with_criteria", [False, True])
def test_core_evidence_and_shared_criteria_have_no_provider_dependency(with_criteria) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import sys

original_import = builtins.__import__
blocked = {"kedi_typesafe", "typesafe_sdk", "kedi_laya", "laya", "laya_mlx", "torch", "mlx"}
if sys.argv[1] == "False":
    blocked.add("kedi_decisions")

def blocked_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("Unexpected optional import: " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import kedi
from kedi.decisions import DecisionInfo, capture_decisions, decision_info
assert not blocked.intersection(sys.modules)
if sys.argv[1] == "True":
    from kedi.decisions import BooleanCriteria, ChoiceCriteria, Probability, Rubric
    from kedi.lang import compile_program, parse_program
    runtime = compile_program(parse_program("> import: decisions\\n= imported\\n"))
    assert runtime.run_main() == "imported"
assert not blocked.intersection(sys.modules)
""",
            str(with_criteria),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
