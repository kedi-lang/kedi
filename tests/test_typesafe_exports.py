from __future__ import annotations

import builtins
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def test_typesafe_exports_are_package_primitives() -> None:
    import kedi_typesafe as package

    import kedi.typesafe as surface

    assert surface.__all__ == ["BooleanCriteria", "ChoiceCriteria", "Probability", "Rubric"]
    for name in surface.__all__:
        assert getattr(surface, name) is getattr(package, name)


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
