from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from kedi.cli import main


def test_debug_help_does_not_start_server():
    result = CliRunner().invoke(main, ["debug", "--help"])
    assert result.exit_code == 0
    assert "kedi debug --stdio" in result.output


@pytest.mark.parametrize("arguments", [[], ["--port", "9000"]])
def test_debug_requires_stdio(arguments):
    result = CliRunner().invoke(main, ["debug", *arguments])
    assert result.exit_code == 2
    assert "kedi debug --stdio" in result.output


def test_debug_dispatches_to_optional_package(monkeypatch):
    calls = []
    original = importlib.import_module

    def load(name, package=None):
        if name == "kedi_debugger.__main__":
            return SimpleNamespace(main=lambda args: calls.append(args))
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)
    result = CliRunner().invoke(main, ["debug", "--stdio"])
    assert result.exit_code == 0, result.output
    assert calls == [["--stdio"]]


@pytest.mark.parametrize("missing", ["kedi_debugger", "kedi_debugger.__main__", "dependency"])
def test_debug_reports_missing_package_without_hiding_dependency_errors(monkeypatch, missing):
    original = importlib.import_module
    error = ModuleNotFoundError("Missing test dependency", name=missing)

    def load(name, package=None):
        if name == "kedi_debugger.__main__":
            raise error
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)
    result = CliRunner().invoke(main, ["debug", "--stdio"])
    assert result.exit_code == 1
    if missing == "dependency":
        assert result.exception is error
    else:
        assert "selected Python needs kedi-debugger" in result.output
