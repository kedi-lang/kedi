from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

import kedi
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.errors import KediExecutionError


class InteractiveCapabilities(PydanticAdapter):
    capabilities = replace(PydanticAdapter.capabilities, supports_capabilities=True)


@pytest.fixture(autouse=True)
def isolated_configuration():
    kedi.reset_config()
    yield
    kedi.reset_config()


def test_interactive_declarations_reset_and_reuse_do_not_replay_or_cross_sessions():
    instructions = []

    def respond(messages, info):
        instructions.append(info.instructions or "")
        return ModelResponse([TextPart("done")])

    adapter = InteractiveCapabilities(FunctionModel(respond))
    with kedi.interactive(adapter=adapter) as first, kedi.interactive(adapter=adapter) as second:
        first.execute(
            "> capability: concise:\n  > instructions: Keep the reply brief.\n"
            "> instructions: Base guidance.\n> use: concise\n"
        )
        first.execute("[reply] << Report the status.\n> show: <reply>")
        first.execute(
            "> instructions: Base guidance.\n[reply] << Report the status.\n> show: <reply>"
        )
        first.execute("> use: concise\n[reply] << Report the status.\n> show: <reply>")
        with pytest.raises(KediExecutionError, match="concise"):
            second.execute("> use: concise")
        second.execute("[reply] << Report the status.\n> show: <reply>")
        first.execute("[reply] << Report the status.\n> show: <reply>")

    assert len(instructions) == 5
    assert ["Keep the reply brief." in text for text in instructions] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert [text.count("Keep the reply brief.") for text in instructions] == [1, 0, 1, 0, 1]
    assert ["Base guidance." in text for text in instructions] == [True, True, True, False, True]


def test_interactive_procedure_keeps_capability_capture_across_later_cells():
    instructions = []

    def respond(messages, info):
        instructions.append(info.instructions or "")
        return ModelResponse([TextPart("done")])

    with kedi.interactive(adapter=InteractiveCapabilities(FunctionModel(respond))) as session:
        session.execute(
            "> capability: concise:\n  > instructions: Keep the reply brief.\n"
            "> use: concise\n@report() -> str:\n  [reply] << Report the status.\n  = <reply>\n"
        )
        session.execute("> instructions: New cell guidance.\n> show: <report()>")
        session.execute("[reply] << Report the status.\n> show: <reply>")

    assert "Keep the reply brief." in instructions[0]
    assert "New cell guidance." not in instructions[0]
    assert "Keep the reply brief." not in instructions[1]
    assert "New cell guidance." in instructions[1]
