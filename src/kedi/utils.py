from __future__ import annotations

from typing import TypeVar

K = TypeVar("K")  # key type
V = TypeVar("V")  # value type


class AttrDict(dict[K, V]):
    """A dictionary that supports attribute-style access."""

    def __getattr__(self, key: K) -> V:
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


def parse_args(args: list[str]) -> AttrDict[str, str | bool | int | float]:
    """
    Parse command-line arguments in the format --key value or --flag.

    Args:
        args: List of command-line arguments.

    Returns:
        A dictionary mapping option names to their values.

    Example:
        >>> parse_args(["--foo", "bar", "--debug", "--timeout", "30"])
        {"foo": "bar", "debug": True, "timeout": 30}
    """
    data = {}
    for i, arg in enumerate(args):
        if not arg.startswith("--"):
            continue

        key = arg.lstrip("--").replace("-", "_")
        next_arg = args[i + 1] if i + 1 < len(args) else None

        # Check if next argument is a value or another flag
        if next_arg and not next_arg.startswith("--"):
            value = cast_value(next_arg)
            if key not in data:
                data[key] = value
        else:
            # It's a flag
            data[key] = True

    return AttrDict(data)
