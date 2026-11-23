from __future__ import annotations

from typing import Any

# Re-exported: `kedi.__main__.dotenv` is the hook tests (and embedders)
# patch to observe the .env load; the load itself goes through
# load_project_dotenv() below.
import dotenv  # noqa: F401

from .utils import load_project_dotenv


def main(*args: Any, **kwargs: Any) -> Any:
    load_project_dotenv()

    from .cli import main as cli_main

    return cli_main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
