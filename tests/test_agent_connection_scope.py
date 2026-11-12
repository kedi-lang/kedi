from __future__ import annotations

import asyncio

import pytest

from kedi.lang import compile_program, parse_program


@pytest.mark.parametrize("transport", ["a2a", "acp"])
@pytest.mark.parametrize("procedure", [False, True])
def test_connection_expressions_see_prelude_and_lexical_parameters(
    transport: str, procedure: bool
) -> None:
    field = "endpoint" if transport == "a2a" else "command"
    attribute = "a2a_connection.endpoint" if transport == "a2a" else "agent_command"
    value = "https://agent.example.com" if transport == "a2a" else "uv run worker"
    prelude = (
        "```\n"
        "from kedi.agent_profile import current_block_agent_profile\n"
        f"connection_value = {value!r}\n"
        "```\n\n"
    )
    statements = (
        f"> agent: {transport}:\n"
        f"    {field}: `connection_value`\n"
        f"= `current_block_agent_profile().{attribute}`\n"
    )
    if procedure:
        prelude += "@select(connection_value: str) -> str:\n"
        statements = "".join("    " + line + "\n" for line in statements.splitlines())
        statements += f"\n= `select({value + '/local'!r})`\n"
    runtime = compile_program(parse_program(prelude + statements))
    try:
        assert runtime.run_main() == value + ("/local" if procedure else "")
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize("profile", [False, True])
def test_model_and_settings_expressions_see_prelude(profile: bool) -> None:
    source = (
        "```\n"
        "from kedi.agent_profile import current_block_agent_profile\n"
        "model_name = 'test:model'\n"
        "retry_limit = 3\n"
        "```\n\n"
    )
    directives = "> model: `model_name`\n> settings:\n    max_retries: `retry_limit`\n"
    if profile:
        source += "> profile: worker:\n"
        source += "".join("    " + line + "\n" for line in directives.splitlines())
        source += "\n> use: worker\n"
    else:
        source += directives
    source += "= `(current_block_agent_profile().model, current_block_agent_profile().settings['max_retries'])`\n"
    runtime = compile_program(parse_program(source))
    try:
        assert runtime.run_main() == ("test:model", 3)
    finally:
        asyncio.run(runtime.aclose())
