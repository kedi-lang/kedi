import asyncio
import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.conversation import ConversationState
from kedi.engine import SequentialEngine
from kedi.interactive_session import InteractiveSession
from kedi.lang.parser.diagnostics import KediParseError


def test_cli_uses_source_relative_files_and_scoped_model(monkeypatch, tmp_path):
    import kedi.agent_adapter.adapters.pydantic.adapter as models
    from kedi.cli import main

    calls = []

    def respond(messages, info):
        calls.append(messages)
        output = {"receipt": "abc123", "usable": 37}
        return ModelResponse([ToolCallPart(info.output_tools[0].name, output)])

    def resolve(model):
        return FunctionModel(respond) if model == "test:scoped" else model

    monkeypatch.setattr(models, "resolve_pydantic_model", resolve)
    monkeypatch.setenv("KEDI_TEST_MODEL", "test:scoped")
    monkeypatch.setenv("KEDI_TEST_MARKER", "environment-marker")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    application = tmp_path / "application"
    application.mkdir()
    (application / "main.kedi").write_text(
        (Path(__file__).parent / "files/cli_environment/stock.kedi").read_text()
    )
    (application / "stock.txt").write_text(
        "Receipt abc123: seven cartons of six units; five units are damaged."
    )
    result = CliRunner().invoke(main, ["--adapter", "pydantic", str(application / "main.kedi")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "receipt": "abc123",
        "usable": 37,
        "environment": "environment-marker",
    }
    assert len(calls) == 1


def test_interactive_fragments_preserve_lexical_instructions_after_parse_failure():
    calls, observations = [], []

    def respond(messages, info):
        code = re.search(r"operation code is (\w+)\.", info.instructions)[1]
        calls.append(code)
        output = {name: code for name in info.output_tools[0].parameters_json_schema["properties"]}
        return ModelResponse([ToolCallPart(info.output_tools[0].name, output)])

    engine, state = SequentialEngine(), ConversationState()
    session = InteractiveSession(
        adapter=PydanticAdapter(FunctionModel(respond)),
        engine=engine,
        conversation_state=state,
        runtime_globals={"observe": observations.append},
    )
    fragments = [
        "> history:\n  enabled: true\n> instructions: The operation code is first.\n"
        "@original() -> str:\n  >> The operation code is [code: str].\n  = `code`",
        "[initial] = <original()>\n`observe(initial)`",
        "> instructions: The operation code is second.\n> effort: low",
        "[captured] = <original()>\n`observe(captured)`",
        ">> The operation code is [current: str].",
        "`observe(current)`\n`observe(initial)`\n`observe(current)`",
    ]
    try:
        for source in fragments:
            session.execute(source)
        with pytest.raises(KediParseError):
            session.execute("> missing_directive:")
        session.execute("`observe(current)`")
        assert observations == ["first", "first", "second", "first", "second", "second"]
        assert calls == ["first", "first", "second"]
    finally:
        asyncio.run(session._compiler.rt.aclose())
        session.close()
        state.close()
        engine.shutdown()
