from __future__ import annotations

import importlib
import subprocess
import sys
from textwrap import dedent


def test_dspy_metadata_does_not_import_optional_runtime(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "dspy", None)
    sys.modules.pop("kedi.agent_adapter.dspy_metadata", None)

    metadata = importlib.import_module("kedi.agent_adapter.dspy_metadata")

    assert metadata.DSPY_ADAPTER.shortname == "dspy"
    assert metadata.DSPY_ADAPTER.kind == "agent-framework"
    assert metadata.DSPY_ADAPTER.capabilities.supports_structured_output is True


def test_langchain_configuration_defers_optional_runtime_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""
                import importlib.abc
                import sys
                import warnings

                attempted = []

                class BlockLangChain(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.split(".")[0] in {"langchain", "langchain_core", "langgraph"}:
                            attempted.append(fullname)
                            raise ModuleNotFoundError(fullname, name=fullname)

                sys.meta_path.insert(0, BlockLangChain())
                from kedi.agent_adapter import LangChainAdapter

                adapter = LangChainAdapter(model="openai:test")
                with warnings.catch_warnings(record=True):
                    LangChainAdapter(
                        history_processor=lambda ctx: ctx.messages,
                        history_processor_condition=lambda state: False,
                    )
                assert not attempted, attempted

                try:
                    adapter._chat_model_for_run()
                except ModuleNotFoundError as exc:
                    assert "Install with: uv add 'kedi[langchain]'" in str(exc), str(exc)
                else:
                    raise AssertionError("model execution must require optional dependencies")
                assert attempted
            """),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
