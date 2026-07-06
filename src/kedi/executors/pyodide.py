from __future__ import annotations

from typing import TypeAlias

from .playground import (
    PlaygroundBridge,
    PlaygroundExecutor,
    PlaygroundReference,
    _decode_type_descriptor,
    _json_value,
    _prelude_names,
    _referenced_names,
    _response_env,
    _serialize_env,
)

PyodideBridge: TypeAlias = PlaygroundBridge
PyodideReference: TypeAlias = PlaygroundReference


class PyodideExecutor(PlaygroundExecutor):
    """Execute user-authored Python through a browser Pyodide bridge worker."""


__all__ = [
    "PyodideBridge",
    "PyodideExecutor",
    "PyodideReference",
    "_decode_type_descriptor",
    "_json_value",
    "_prelude_names",
    "_referenced_names",
    "_response_env",
    "_serialize_env",
]
