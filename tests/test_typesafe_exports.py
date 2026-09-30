from __future__ import annotations

import builtins
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_profile import AgentProfile
from kedi.lang import compile_program, parse_program


def test_typesafe_exports_are_package_primitives() -> None:
    import kedi_typesafe as package

    import kedi.typesafe as surface

    assert surface.__all__ == ["BooleanCriteria", "ChoiceCriteria", "Probability", "Rubric"]
    for name in surface.__all__:
        assert getattr(surface, name) is getattr(package, name)


@pytest.mark.parametrize("names", [None, ["Probability", "Rubric"]])
def test_jev_module_exports_existing_primitives_without_selecting_model(names) -> None:
    import kedi.typesafe as surface

    directive = "> import: typesafe\n"
    if names is not None:
        directive = "> import: typesafe:\n" + "".join(f"  {name}\n" for name in names)
    profile = AgentProfile(model="openai:unchanged")
    runtime = compile_program(
        parse_program(directive + "= imported\n"), default_agent_profile=profile
    )
    assert runtime.run_main() == "imported"
    assert runtime.default_agent_profile.model == profile.model
    for name in surface.__all__:
        if names is None or name in names:
            assert runtime._import_env[name] is getattr(surface, name)
        else:
            assert name not in runtime._import_env


def test_jev_module_types_work_in_native_annotations() -> None:
    from pydantic import ValidationError

    runtime = compile_program(
        parse_program(
            "> import: typesafe\n"
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


def test_jev_exports_are_visible_to_editor_without_running_module() -> None:
    from kedi.lsp.module_surface import load_imported_surface
    from kedi.module_registry import resolve_module_path

    surface = load_imported_surface(
        resolve_module_path("typesafe", source_path=None), set(), "typesafe"
    )
    assert surface.exported_names() == {
        "BooleanCriteria",
        "ChoiceCriteria",
        "Probability",
        "Rubric",
    }


def test_jev_module_reports_missing_optional_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
from kedi.lang import compile_program, parse_program

original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "kedi_typesafe":
        raise ModuleNotFoundError("Missing optional dependency", name=name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
compile_program(parse_program("> import: typesafe\\n= imported\\n"))
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "uv add 'kedi[typesafe]'" in result.stderr


@pytest.mark.parametrize("missing_name", ["kedi_typesafe", "typesafe_sdk"])
def test_typesafe_import_errors(monkeypatch: pytest.MonkeyPatch, missing_name: str) -> None:
    original_import = builtins.__import__
    missing = ModuleNotFoundError("Dependency missing", name=missing_name)

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "kedi_typesafe":
            raise missing
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    module = Path(__file__).parents[1] / "src" / "kedi" / "typesafe.py"
    with pytest.raises(ModuleNotFoundError) as error:
        runpy.run_path(str(module))
    assert error.value.name == missing_name
    if missing_name == "kedi_typesafe":
        assert "uv add 'kedi[typesafe]'" in str(error.value)
        assert error.value.__cause__ is missing
    else:
        assert error.value is missing


def test_import_kedi_does_not_require_typesafe() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import sys

original_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "kedi_typesafe" or name.startswith("kedi_typesafe."):
        raise AssertionError("Kedi must not eagerly import TypeSafe")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import kedi
assert "kedi.typesafe" not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
