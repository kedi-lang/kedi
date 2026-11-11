from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ConditionalLoopStmt, IfStmt, TextSegment
from tests.mock_adapter import MockAdapter


@pytest.mark.parametrize(
    "source",
    [
        '= `"\x00"`\n',
        '```\nvalue = "\x00"\n```\n= ok\n',
        '[value: str] = ```\nreturn "\x00"\n```\n',
        '= ```\nreturn "\x00\x00"\n```\n',
    ],
)
@pytest.mark.parametrize("surface", ["parser", "lsp", "cli"])
def test_nul_in_python_source_fails_without_hanging(
    tmp_path: Path, source: str, surface: str
) -> None:
    script = tmp_path / "parse_nul.py"
    script.write_text(
        "from tree_sitter import Language, Parser\n"
        "from tree_sitter_kedi import language\n"
        f"source = {source!r}\n"
        "parser = Parser(Language(language()))\n"
        "assert parser.parse(source.encode()).root_node.has_error\n"
        "assert not parser.parse(b'= ok\\n').root_node.has_error\n"
        + (
            "from kedi.lang import parse_program, KediParseError\n"
            "try:\n"
            "    parse_program(source)\n"
            "except KediParseError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('NUL source was accepted')\n"
            if surface == "parser"
            else (
                "from kedi.lsp.features.diagnostics import compute_diagnostics\n"
                "assert any(d.severity == 1 for d in compute_diagnostics(source, 'file:///nul.kedi'))\n"
                if surface == "lsp"
                else "from click.testing import CliRunner\n"
                "from kedi.cli import main\n"
                "from pathlib import Path\n"
                "path = Path(__file__).with_suffix('.kedi')\n"
                "path.write_bytes(source.encode())\n"
                "result = CliRunner().invoke(main, [str(path)])\n"
                "assert result.exit_code == 1, result.output\n"
                "assert 'parse error' in result.output.lower(), result.output\n"
            )
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=5, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_escaped_nul_in_python_source_is_still_valid() -> None:
    parse_program('```\nvalue = "\\0"\n```\n= ok\n')


@pytest.mark.parametrize("directive", ["if", "loop"])
@pytest.mark.parametrize(
    "claim, punctuation",
    [
        ("The time <hour>:<minute> is in the morning", ":"),
        ("The ratio <hour> :1 is above 2", " :1 is above 2"),
        ("<hour>:<minute>", ":"),
        ("<hour>: still too early", ": still too early"),
    ],
)
def test_internal_colon_after_substitution_is_claim_text(
    directive: str, claim: str, punctuation: str
) -> None:
    program = parse_program(f"> {directive}: {claim}\n  > show: matched\n")
    statement = program.toplevel_body[0]
    assert isinstance(statement, (IfStmt, ConditionalLoopStmt))
    assert statement.condition_segments is not None
    assert punctuation in "".join(
        segment.text for segment in statement.condition_segments if isinstance(segment, TextSegment)
    )


@pytest.mark.parametrize("directive", ["if", "loop"])
@pytest.mark.parametrize(
    "claim, rendered",
    [
        ("The time <hour>:<minute> is in the morning", "The time 8:30 is in the morning"),
        ("The ratio <hour> :1 is above 2", "The ratio 8 :1 is above 2"),
        ("<hour>:<minute>", "8:30"),
        ("<hour>: still too early", "8: still too early"),
    ],
)
def test_internal_claim_colons_reach_the_model(directive, claim, rendered) -> None:
    class ClaimAdapter(MockAdapter):
        def produce_sync(self, *, template, **kwargs):
            self.request_calls.append(template)
            return False

    adapter = ClaimAdapter()
    source = f"[hour] = 8\n[minute] = 30\n> {directive}: {claim}\n  > show: matched\n"
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        runtime.run_main()
    finally:
        runtime.artifacts.close()
    assert len(adapter.request_calls) == 1
    assert rendered in adapter.request_calls[0]
    assert f"{rendered}:" not in adapter.request_calls[0]
