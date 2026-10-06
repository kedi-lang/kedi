from __future__ import annotations

import asyncio

import pytest

from kedi.agent_adapter.tools.conversion import kedi_procedure_tool_spec
from kedi.agent_adapter.tools.retry import ToolRetryPolicy
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.procedure_introspection import get_procedure_def
from tests.mock_adapter import MockAdapter


@pytest.mark.parametrize("reexport", [False, True])
def test_imported_tool_preserves_definition_site_retry_policy(tmp_path, reexport) -> None:
    (tmp_path / "store.kedi").write_text("""
```
Transient = TimeoutError
```
@read_record(reference: str) -> str:
    > tool:
        name: private_lookup
        risk: read_only
        retries: 1
        retry_on:
            Transient
    = `read_backend(reference)`
> export:
    read_record
""")
    (tmp_path / "forward.kedi").write_text("> import: store\n> export:\n    read_record\n")
    module = "forward" if reexport else "store"
    calls = []

    def read_backend(reference):
        calls.append(reference)
        if reference == "invalid-record":
            raise ValueError("Nonretryable record failure")
        if len(calls) % 2:
            raise TimeoutError("Temporary read failure")
        return "receipt"

    runtime = compile_program(
        parse_program(
            f"> import: {module}\n```\nTransient = ValueError\n```\n> use: read_record\n= ready\n",
            source_path=str(tmp_path / "main.kedi"),
        ),
        adapter=MockAdapter(),
        runtime_globals={"read_backend": read_backend},
    )
    try:
        runtime.run_main()
        fn = runtime._get_proc("read_record")
        spec = kedi_procedure_tool_spec(runtime, get_procedure_def(fn))
        assert spec.name == "private_lookup"
        assert spec.risk == "read_only"
        assert spec.retry_policy == ToolRetryPolicy(1, (TimeoutError,))
        assert spec.fn(reference="R-7") == "receipt"
        assert calls == ["R-7", "R-7"]
        with pytest.raises(KediExecutionError):
            fn(reference="direct")
        assert calls == ["R-7", "R-7", "direct"]
        with pytest.raises(KediExecutionError):
            spec.fn(reference="invalid-record")
        assert calls == ["R-7", "R-7", "direct", "invalid-record"]
        with pytest.raises(TypeError, match="does not match annotated type"):
            spec.fn(reference=17)
        assert calls == ["R-7", "R-7", "direct", "invalid-record"]
    finally:
        asyncio.run(runtime.aclose())
