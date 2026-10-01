from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def sources_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "debugger" / "src"))
    return importlib.import_module("kedi_debugger.sources")


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "```\nseed = 1\n```\n> import: this\n~Record(value: int)\n"
            "@outer():\n  @inner():\n    = `1`\n  = <inner()>\n= <outer()>\n",
            {1, 4, 5, 6, 7, 8, 9, 10},
        ),
        (
            "> profile: standard:\n  > model: example\n> model: example\n>> A [a]\nB [b]\n= done\n",
            {1, 3, 4, 6},
        ),
        (
            "# comment\n> if: `True`:\n  [x] = yes\n> else:\n  [x] = no\n"
            "> loop [item]: `range(2)`:\n  [value: int] = `item`\n"
            "> loop: `False`:\n  > show: body\n= done\n",
            {2, 3, 5, 6, 7, 8, 9, 10},
        ),
        (
            "> profile: worker:\n  > adapter: pydantic\n> task [job]: worker:\n"
            "  >> Explain the result.\n> await: job\n= done\n",
            {1, 3, 5, 6},
        ),
        (
            "[value: int] = ```\nreturn 7\n```\n= ```\nreturn value\n```\n",
            {1, 4},
        ),
    ],
    ids=["declarations", "configuration-and-templates", "control-flow", "task-template", "python"],
)
def test_locations_follow_executable_blocks(sources_module, tmp_path, text, expected):
    sources = sources_module.Sources()
    path = sources.register(str(tmp_path / "source.kedi"), text)
    assert sources.lines[path] == expected


def test_registration_is_immutable_and_does_not_read_disk(sources_module, tmp_path):
    sources = sources_module.Sources()
    path = tmp_path / "source.kedi"
    text = "[value: int] = `1`\n= <value>\n"
    normalized = sources.register(str(path), text)
    original_lines = sources.lines[normalized].copy()
    assert sources.register(path.as_uri(), text) == normalized
    with pytest.raises(ValueError, match="restart required"):
        sources.register(str(path), "= changed\n")
    assert sources.text[normalized] == text
    assert sources.lines[normalized] == original_lines


def test_repeated_load_and_verify_detect_changes_without_replacing_snapshot(
    sources_module, tmp_path
):
    path = tmp_path / "source.kedi"
    path.write_text("= before\n", encoding="utf-8")
    sources = sources_module.Sources()
    normalized = sources.load(str(path))
    assert sources.load(path.as_uri()) == normalized
    sources.verify()
    path.write_text("= after\n", encoding="utf-8")
    for check in (sources.verify, lambda: sources.load(str(path))):
        with pytest.raises(ValueError, match="restart required"):
            check()
    assert sources.content(normalized) == "= before\n"


def test_verify_checks_registered_documents_and_missing_files(sources_module, tmp_path):
    sources = sources_module.Sources()
    for name in ("main", "imported"):
        path = tmp_path / f"{name}.kedi"
        path.write_text("= done\n", encoding="utf-8")
        sources.register(str(path), "= done\n")
    sources.verify()
    (tmp_path / "imported.kedi").unlink()
    with pytest.raises(ValueError, match="restart required"):
        sources.verify()


def test_crlf_disk_sources_match_compiler_newline_normalization(sources_module, tmp_path):
    path = tmp_path / "source.kedi"
    path.write_bytes(b"[value: int] = `1`\r\n= <value>\r\n")
    sources = sources_module.Sources()
    normalized = sources.load(str(path))
    assert sources.register(normalized, path.read_text(encoding="utf-8")) == normalized
    sources.verify()
    assert sources.lines[normalized] == {1, 2}


def test_content_masks_full_source_without_mutating_cached_text(
    sources_module, tmp_path, monkeypatch
):
    secret = "fixture-source-credential"
    monkeypatch.setenv("EXAMPLE_TOKEN", secret)
    text = "# " + secret + "\r\n" + "# padding\r\n" * 2000 + "= " + secret + "\r\n"
    sources = sources_module.Sources()
    path = sources.register(str(tmp_path / "source.kedi"), text)
    masked = sources.content(Path(path).as_uri())
    assert masked == text.replace(secret, "*" * len(secret))
    assert len(masked) == len(text)
    assert sources.text[path] == text


def test_paths_normalize_relative_unicode_uri_and_symlinks(sources_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "source \u0131.kedi"
    path.write_text("= done\n", encoding="utf-8")
    alias = tmp_path / "alias.kedi"
    alias.symlink_to(path)
    sources = sources_module.Sources()
    assert sources.load(path.name) == str(path)
    assert sources.load(path.as_uri()) == str(path)
    assert sources.load(str(alias)) == str(path)
    assert len(sources.text) == 1
    with pytest.raises(ValueError, match="local file"):
        sources_module.source_path("file://remote/source.kedi")


def test_source_count_limit_allows_existing_registrations(sources_module, tmp_path):
    sources = sources_module.Sources()
    for index in range(64):
        sources.register(str(tmp_path / f"{index}.kedi"), "= done\n")
    sources.register(str(tmp_path / "0.kedi"), "= done\n")
    with pytest.raises(ValueError, match="64 files"):
        sources.register(str(tmp_path / "65.kedi"), "= done\n")
    with pytest.raises(ValueError, match="64 files"):
        sources.load(str(tmp_path / "65.kedi"))
    assert len(sources.text) == 64


def test_source_byte_limit_applies_to_registration_and_disk(sources_module, tmp_path):
    maximum = sources_module.MAX_SOURCE_BYTES
    path = tmp_path / "large.kedi"
    sources = sources_module.Sources()
    text = "#" + "x" * (maximum - 2) + "\n"
    sources.register(str(path), text)
    path.write_bytes(text.encode("utf-8"))
    sources.verify()
    path.write_bytes((text + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="restart required"):
        sources.verify()
    with pytest.raises(ValueError, match="2 MiB"):
        sources_module.Sources().load(str(path))
    for oversized in (text + "\n", "#" + "\u0131" * (maximum // 2)):
        with pytest.raises(ValueError, match="2 MiB"):
            sources_module.Sources().register(str(path), oversized)


def test_invalid_source_does_not_poison_registry(sources_module, tmp_path):
    sources = sources_module.Sources()
    path = tmp_path / "source.kedi"
    with pytest.raises(Exception):
        sources.register(str(path), "[broken\n")
    assert sources.text == sources.lines == {}
    sources.register(str(path), "= fixed\n")
    with pytest.raises(ValueError, match="not loaded"):
        sources.content(str(tmp_path / "unknown.kedi"))
    with pytest.raises(TypeError, match="exact string"):
        sources.register(str(path), b"= done\n")
    with pytest.raises(ValueError, match=".kedi"):
        sources.register(str(tmp_path / "source.py"), "= done\n")
