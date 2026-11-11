from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import kedi.cli as cli_module


def test_cli_programs_without_a_prelude_see_file_and_args(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("notes for the demo", encoding="utf-8")
    source = tmp_path / "repro.kedi"
    source.write_text(
        "[notes] = ```\n"
        "from pathlib import Path\n"
        'return Path(__file__).with_name("notes.md").read_text(encoding="utf-8")\n'
        "```\n"
        "= <notes> (<`args.mode`>)\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli_module.main, [str(source), "--no-cache", "--mode", "demo"])

    assert result.exit_code == 0, result.output
    assert "notes for the demo (demo)" in result.output
