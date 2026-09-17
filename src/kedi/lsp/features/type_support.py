from __future__ import annotations

import builtins as _builtins
import typing as _typing
from typing import Optional

from kedi.lang.compiler.environment import KEDI_RUNTIME_TYPE_ORIGINS, KEDI_TYPING_TYPE_NAMES
from kedi.lang.compiler.type_system import UNSUPPORTED_KEDI_TYPE_NAMES
from kedi.lsp.features.parser_state import _typing_extensions


def _build_lsp_type_registry() -> dict[str, str]:
    registry: dict[str, str] = {}
    for name in dir(_builtins):
        if name.startswith("_") or name in UNSUPPORTED_KEDI_TYPE_NAMES:
            continue
        value = vars(_builtins)[name]
        if isinstance(value, type) and not name.startswith("_"):
            registry[name] = "builtins"
    for module, origin in (
        (_typing, "typing"),
        (_typing_extensions, "typing_extensions"),
    ):
        if module is None:
            continue
        for name in KEDI_TYPING_TYPE_NAMES:
            if hasattr(module, name):
                registry.setdefault(name, origin)
    for name, origin in KEDI_RUNTIME_TYPE_ORIGINS.items():
        registry[name] = origin
    return registry


_LSP_TYPE_REGISTRY = _build_lsp_type_registry()


def _known_lsp_type_origin(name: str) -> Optional[str]:
    return _LSP_TYPE_REGISTRY.get(name)
