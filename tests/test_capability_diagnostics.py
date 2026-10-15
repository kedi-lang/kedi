from dataclasses import replace

import pytest
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


class CapabilityAdapter(PydanticAdapter):
    capabilities = replace(PydanticAdapter.capabilities, supports_capabilities=True)


def test_conflicting_approval_reports_both_capability_definition_locations(tmp_path):
    def read_document() -> str:
        return "revision 7"

    path = str(tmp_path / "policies.kedi")
    runtime = compile_program(
        parse_program(
            "> capability: first:\n  > approval: allow\n  > use: read_document\n"
            "> capability: second:\n  > approval: deny\n  > use: read_document\n"
            "> use: first\n> use: second\n[reply] << Read the revision.\n= <reply>\n",
            source_path=path,
        ),
        adapter=CapabilityAdapter(TestModel(call_tools=[])),
        runtime_globals={"read_document": read_document},
    )
    try:
        with pytest.raises(KediExecutionError, match="Conflicting capability approval") as caught:
            runtime.run_main()
    finally:
        runtime.artifacts.close()
    origins = {
        (frame.label, frame.span.source_path, frame.span.start_line)
        for frame in caught.value.frames
        if frame.kind == "capability" and frame.span is not None
    }
    assert origins == {("capability first", path, 1), ("capability second", path, 4)}


def test_conflicting_private_tools_report_both_imported_definition_locations(tmp_path):
    for name in ("first", "second"):
        (tmp_path / f"{name}.kedi").write_text(
            f"@access() -> str:\n  = {name}\n"
            f"> capability: {name}:\n  > use: access\n> export:\n  {name}\n"
        )
    runtime = compile_program(
        parse_program(
            "> import: first\n> import: second\n> use: first\n> use: second\n= done\n",
            source_path=str(tmp_path / "main.kedi"),
        ),
        adapter=CapabilityAdapter(TestModel(call_tools=[])),
    )
    try:
        with pytest.raises(
            KediExecutionError, match="Conflicting capability tool binding"
        ) as caught:
            runtime.run_main()
    finally:
        runtime.artifacts.close()
    origins = {
        (frame.span.source_path, frame.span.start_line)
        for frame in caught.value.frames
        if frame.kind == "capability" and frame.span is not None
    }
    assert origins == {(str(tmp_path / f"{name}.kedi"), 3) for name in ("first", "second")}
