from __future__ import annotations

import os
import subprocess
import sys


def test_importing_kedi_disables_the_pydantic_ai_banner() -> None:
    environment = os.environ.copy()
    environment["PYDANTIC_AI_NO_BANNER"] = "0"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import kedi; print(os.environ['PYDANTIC_AI_NO_BANNER'])",
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )

    assert completed.stdout == "1\n"
    assert completed.stderr == ""
