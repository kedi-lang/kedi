from __future__ import annotations

from typing import Any, TypeVar, cast

K = TypeVar("K")  # key type
V = TypeVar("V")  # value type
ArgType = str | bool | None
_MISSING = object()


class AttrDict(dict[K, V]):
    """A dictionary that supports attribute-style access."""

    def __init__(self, *args: Any, default: Any = _MISSING, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._default = default

    def __getattr__(self, key: str) -> V:
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
