from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from click.testing import CliRunner

from kedi.cli import main as cli_main
from kedi.lang import compile_program, parse_program


@pytest.mark.parametrize("source_path", [None, "<string>", "<notebook:session:1>"])
def test_synthetic_programs_do_not_invent_a_file_binding(source_path):
    runtime = compile_program(
        parse_program('= `"__file__" in globals()`\n', source_path=source_path)
    )
    assert runtime.run_main() is False


def test_file_binding_is_absolute_and_available_in_prelude_and_procedure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = compile_program(
        parse_program(
            "```\nsource = __file__\n```\n"
            "@location() -> str:\n  = `__file__`\n"
            "= `(source, location(), __file__)`\n",
            source_path="main.kedi",
        ),
        runtime_globals={"__file__": "caller.py"},
    )
    expected = str(tmp_path / "main.kedi")
    assert runtime.run_main() == (expected, expected, expected)
    assert Path.cwd() == tmp_path


def test_imported_modules_keep_their_own_file_binding(tmp_path, monkeypatch):
    (tmp_path / "library").mkdir()
    (tmp_path / "library" / "note.txt").write_text("library data", encoding="utf-8")
    (tmp_path / "library" / "reader.kedi").write_text(
        "```\nfrom pathlib import Path\nsource = __file__\n```\n"
        "@read_note() -> str:\n"
        '  = `Path(__file__).with_name("note.txt").read_text()`\n'
        "> export: *\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    source = "> import: library/reader\n= `(__file__, source, read_note())`\n"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    runtime = compile_program(parse_program(source, source_path=str(main)))
    assert runtime.run_main() == (
        str(main),
        str(tmp_path / "library" / "reader.kedi"),
        "library data",
    )
    assert "__file__" not in runtime._import_env
    assert Path.cwd() == elsewhere


def test_concurrent_programs_do_not_change_each_others_file_or_cwd(tmp_path):
    cwd = Path.cwd()
    runtimes = [
        compile_program(parse_program("= `__file__`\n", source_path=str(tmp_path / name)))
        for name in ("first.kedi", "second.kedi")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda runtime: runtime.run_main(), runtimes))
    assert results == [str(tmp_path / name) for name in ("first.kedi", "second.kedi")]
    assert Path.cwd() == cwd


@pytest.mark.parametrize("from_source_directory", [False, True])
def test_cli_reads_sibling_data_from_either_directory(tmp_path, monkeypatch, from_source_directory):
    folder = tmp_path / "example"
    folder.mkdir()
    source = folder / "main.kedi"
    source.write_text(
        "```\nfrom pathlib import Path\n```\n"
        '= `Path(__file__).with_name("notes.md").read_text()`\n',
        encoding="utf-8",
    )
    (folder / "notes.md").write_text("correct sibling", encoding="utf-8")
    (tmp_path / "notes.md").write_text("wrong working directory", encoding="utf-8")
    cwd = folder if from_source_directory else tmp_path
    monkeypatch.chdir(cwd)
    result = CliRunner().invoke(cli_main, [str(source.relative_to(cwd))])
    assert result.exit_code == 0, result.output
    assert "correct sibling" in result.output
    assert "wrong working directory" not in result.output
    assert Path.cwd() == cwd
