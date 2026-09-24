import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_distribution


@pytest.mark.parametrize(
    ("platform", "python_path"),
    [("posix", "bin/python"), ("nt", "Scripts/python.exe")],
)
def test_clean_distribution_uses_platform_paths_and_local_wheels(
    monkeypatch: pytest.MonkeyPatch, platform: str, python_path: str
) -> None:
    calls = []

    def run(command, *, cwd, check, timeout):
        assert check and timeout > 0
        calls.append((command, Path(cwd)))
        if command[1] == "build":
            dist = Path(command[command.index("--out-dir") + 1])
            dist.mkdir(exist_ok=True)
            name = "parser" if command[2].endswith("tree-sitter-kedi") else "kedi"
            (dist / f"{name}.whl").touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(check_distribution, "os", SimpleNamespace(name=platform, environ={}))
    monkeypatch.setattr(
        check_distribution.sys, "argv", ["check_distribution.py", "--python", "3.10"]
    )
    monkeypatch.setattr(check_distribution.subprocess, "run", run)
    check_distribution.main()

    assert len(calls) == 6
    for command, _ in calls[:3]:
        assert command[command.index("--python") + 1] == "3.10"
    install, directory = calls[3]
    python = directory / "venv" / python_path
    assert install[:5] == ["uv", "pip", "install", "--python", str(python)]
    assert len(install[5:]) == 2
    assert all(Path(wheel).parent == directory / "dist" for wheel in install[5:])
    assert calls[4][0] == ["uv", "pip", "check", "--python", str(python)]
    assert calls[5][0] == [
        str(python),
        "-I",
        str(check_distribution.ROOT / "scripts/distribution_smoke.py"),
    ]
    assert all(cwd == directory for _, cwd in calls[2:])
    assert " " in str(directory)
    assert not directory.exists()


def test_distribution_build_failure_propagates_and_cleans_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories = []

    def fail(command, **kwargs):
        directory = Path(command[command.index("--out-dir") + 1]).parent
        directories.append(directory)
        assert directory.exists()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(
        check_distribution.sys, "argv", ["check_distribution.py", "--python", "3.10"]
    )
    monkeypatch.setattr(check_distribution.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        check_distribution.main()
    assert len(directories) == 1
    assert not directories[0].exists()
