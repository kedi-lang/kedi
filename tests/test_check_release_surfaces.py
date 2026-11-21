"""6.25: release-surface checks clean up staging and replace the archive.

The script itself needs node/cargo/submodules to run end to end, so the
regression test has two parts: it asserts the script contains the trap and
the pre-zip removal, and it runs the exact fixed shell pattern in a scratch
directory to prove the pattern behaves as claimed.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release_surfaces.sh"


def test_release_script_traps_staging_cleanup_and_replaces_archive() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "trap 'rm -rf \"$staging\"' EXIT" in text
    rm_index = text.index('rm -f "$ROOT/$zed_archive"')
    zip_index = text.index('zip -qr "$ROOT/$zed_archive"')
    assert rm_index < zip_index


def test_staging_is_removed_when_the_packaging_step_fails(tmp_path: Path) -> None:
    marker = tmp_path / "staging-path.txt"
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            'staging="$(mktemp -d)"\n'
            "trap 'rm -rf \"$staging\"' EXIT\n"
            f"printf '%s' \"$staging\" > {marker}\n"
            "false\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    staging = Path(marker.read_text(encoding="utf-8"))
    assert not staging.exists()


def test_pre_existing_archive_is_replaced_not_appended_to(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "old.txt").write_text("old", encoding="utf-8")
    archive = tmp_path / "extension.zip"
    subprocess.run(
        ["bash", "-c", f"cd {staging} && zip -qr {archive} ."],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "old.txt" in zipfile.ZipFile(archive).namelist()

    (staging / "old.txt").unlink()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    subprocess.run(
        ["bash", "-c", f"rm -f {archive} && (cd {staging} && zip -qr {archive} .)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert zipfile.ZipFile(archive).namelist() == ["new.txt"]
