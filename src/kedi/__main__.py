from __future__ import annotations

from typing import Any

import dotenv


def main(*args: Any, **kwargs: Any) -> Any:
    dotenv.load_dotenv()

    from .cli import main as cli_main

    return cli_main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
