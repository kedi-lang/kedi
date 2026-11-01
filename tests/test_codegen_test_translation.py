from __future__ import annotations

from pathlib import Path

import pytest

from kedi.codegen.interfaces import CodegenConfig, ProgramContext
from kedi.codegen.manager import CodegenManager
from kedi.codegen.mock_agents import MockPassingAgent
from kedi.codegen.translator import python_tests_to_kedi_tests
from kedi.tests_runner import run_tests


def test_generated_test_helpers_and_multiline_docstrings_survive_translation() -> None:
    translated = python_tests_to_kedi_tests(
        "EXPECTED = 3\n"
        "def check(value):\n"
        "    assert value == EXPECTED\n"
        "def test_helper():\n"
        '    """Check the normal case.\n'
        "    This documentation is not executable Python.\n"
        '    """\n'
        "    check(increment(2))\n"
        "def test_nested_helper():\n"
        "    def check_zero():\n"
        "        assert increment(-1) == 0\n"
        "    check_zero()\n",
        "increment",
    )
    results = run_tests("@increment(value: int) -> int:\n  = `value + 1`\n\n" + translated)
    cases = [case for suite in results for case in suite.cases]
    assert len(cases) == 2
    assert all(case.success for case in cases), [case.error for case in cases]
    assert "This documentation" not in translated


@pytest.mark.parametrize(
    "source",
    [
        "def test_fixture(tmp_path):\n    assert True\n",
        "def test_kwargs(**kwargs):\n    assert True\n",
        "def test_args(*args):\n    assert True\n",
        "def test_keyword(*, value=1):\n    assert value == 1\n",
        "def test_positional(value, /):\n    assert True\n",
        "async def test_async():\n    assert True\n",
        "@pytest.mark.parametrize('value', [1])\ndef test_parameterized(value):\n    assert True\n",
    ],
)
def test_unsupported_generated_test_shapes_fail_explicitly(source: str) -> None:
    with pytest.raises(ValueError, match="fixtures and parametrization are unsupported"):
        python_tests_to_kedi_tests(source, "f")


def test_test_translation_error_is_fed_back_to_generator(tmp_path: Path) -> None:
    class RepairingAgent(MockPassingAgent):
        def __init__(self) -> None:
            self.feedback: list[str | None] = []

        def generate_tests(
            self,
            python_signature: str,
            prompt: str,
            context: ProgramContext,
            previous_error: str | None = None,
        ) -> str:
            self.feedback.append(previous_error)
            if previous_error is None:
                return "def test_invalid(fixture):\n    assert True\n"
            return super().generate_tests(python_signature, prompt, context, previous_error)

    source = tmp_path / "program.kedi"
    source.write_text("@f() -> str:\n  > auto:\n    Return ok.\n")
    agent = RepairingAgent()
    manager = CodegenManager(agent, CodegenConfig(max_retries=2))
    assert manager.orchestrate_generation(source, "@f() -> str:", "Return ok.", "")
    assert len(agent.feedback) == 2
    assert "Translation error:" in (agent.feedback[1] or "")
    assert manager.is_cached(source, "f")
