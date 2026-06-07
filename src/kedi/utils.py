from __future__ import annotations

from typing import Any, TypeVar

K = TypeVar("K")  # key type
V = TypeVar("V")  # value type
ArgType = str | bool | None
_MISSING = object()


class AttrDict(dict[K, V]):
    """A dictionary that supports attribute-style access."""

    def __init__(self, *args: Any, default: Any = _MISSING, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._default = default

    def __getattr__(self, key: K) -> V:
        if key in self:
            return self[key]
        if self._default is not _MISSING:
            return self._default
        return self[key]


def cast_value(value: str) -> int | float | bool | str:
    """
    Cast a string value to its corresponding Python type.

    Args:
        value: The string value to cast.

    Returns:
        An int, float, bool, or str depending on the value.
    """
    lowered = value.lower().strip()
    if lowered == "true":
        return True
    elif lowered == "false":
        return False

    try:
        if "." not in value:
            return int(value)
        return float(value)
    except ValueError:
        return value


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
