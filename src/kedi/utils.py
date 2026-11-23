from __future__ import annotations

from typing import Any, TypeVar, cast

import dotenv

K = TypeVar("K")  # key type
V = TypeVar("V")  # value type
ArgType = str | bool | None
_MISSING = object()


def load_project_dotenv() -> bool:
    """Load ``.env`` from the current working directory tree.

    Bare ``dotenv.load_dotenv()`` anchors discovery at the *caller's* file:
    python-dotenv's ``find_dotenv(usecwd=False)`` walks up from the file
    that called it, i.e. from the installed library - so it never sees the
    ``.env`` of the project the user launched from (and can instead pick
    up a ``.env`` sitting above the library). Search from the cwd instead,
    so each project loads its own ``.env``. Existing environment variables
    still win (``override=False``).
    """
    path = dotenv.find_dotenv(usecwd=True)
    if not path:
        return False
    return dotenv.load_dotenv(path)


class AttrDict(dict[K, V]):
    """A dictionary that supports attribute-style access."""

    def __init__(self, *args: Any, default: Any = _MISSING, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._default = default

    def __getattr__(self, key: str) -> V:
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        if key in self:
            return super().__getitem__(cast(K, key))
        default = self.__dict__.get("_default", _MISSING)
        if default is not _MISSING:
            return default
        raise AttributeError(key)


def parse_args(args: list[str]) -> AttrDict[str, ArgType]:
    """
    Parse command-line arguments in the format --key value or --flag.

    Args:
        args: List of command-line arguments.

    Returns:
        A dictionary mapping option names to their values.

    Example:
        >>> parse_args(["--foo", "bar", "--debug", "--timeout", "30"])
        {"foo": "bar", "debug": True, "timeout": "30"}
    """
    data: dict[str, ArgType] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        if not arg.startswith("--"):
            index += 1
            continue

        key = arg.lstrip("--").replace("-", "_")
        next_arg = args[index + 1] if index + 1 < len(args) else None
        if next_arg and not next_arg.startswith("--"):
            data.setdefault(key, next_arg)
            index += 2
        else:
            data.setdefault(key, True)
            index += 1

    return AttrDict(data, default=None)
