from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["kedi.lang", "kedi.lsp"])
def test_language_services_do_not_load_model_frameworks(module: str) -> None:
    code = f"""
import importlib
import sys
from importlib.abc import MetaPathFinder

class BlockFrameworks(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'pydantic_ai', 'pydantic_monty', 'langchain', 'codex_cli', 'claude_agent_sdk'}}:
            raise RuntimeError('Unexpected framework import: ' + fullname)

sys.meta_path.insert(0, BlockFrameworks())
importlib.import_module({module!r})
assert not any(name.startswith('kedi.agent_adapter.adapters.pydantic') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_adapter_exports_still_resolve_and_cache() -> None:
    from kedi.agent_adapter import adapters, compaction

    for name in adapters.__all__:
        first = adapters.__getattr__(name)
        assert getattr(adapters, name) is first
    with pytest.raises(AttributeError, match="no attribute"):
        adapters.__getattr__("unknown")
    name = "PydanticArtifactHistoryProcessor"
    assert compaction.__getattr__(name) is getattr(compaction, name)
    with pytest.raises(AttributeError, match="no attribute"):
        compaction.__getattr__("unknown")
