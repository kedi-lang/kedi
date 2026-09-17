from __future__ import annotations

import importlib.resources
from types import ModuleType

from tree_sitter import Language as _TSLanguage
from tree_sitter import Parser as _TSParser
from tree_sitter import Query as _TSQuery

from kedi.lang.parser.parser import get_language as _get_kedi_language

try:
    import tree_sitter_python as _ts_python_mod

    _PY_LANGUAGE = _TSLanguage(_ts_python_mod.language())
    _PY_PARSER = _TSParser(_PY_LANGUAGE)
    _PY_HIGHLIGHTS_SRC = (
        importlib.resources.files("tree_sitter_python") / "queries" / "highlights.scm"
    ).read_text()
    _PY_HIGHLIGHTS_QUERY = _TSQuery(_PY_LANGUAGE, _PY_HIGHLIGHTS_SRC)
except Exception:  # pragma: no cover — degraded mode if the wheel is missing
    _PY_PARSER = None
    _PY_LANGUAGE = None
    _PY_HIGHLIGHTS_QUERY = None


try:
    _KEDI_LANGUAGE = _get_kedi_language()
    _KEDI_HIGHLIGHTS_SRC = (
        importlib.resources.files("tree_sitter_kedi") / "queries" / "highlights.scm"
    ).read_text()
    _KEDI_INJECTIONS_SRC = (
        importlib.resources.files("tree_sitter_kedi") / "queries" / "injections.scm"
    ).read_text()
    _KEDI_HIGHLIGHTS_QUERY = _TSQuery(_KEDI_LANGUAGE, _KEDI_HIGHLIGHTS_SRC)
    _KEDI_INJECTIONS_QUERY = _TSQuery(_KEDI_LANGUAGE, _KEDI_INJECTIONS_SRC)
except Exception:  # pragma: no cover
    _KEDI_HIGHLIGHTS_QUERY = None
    _KEDI_INJECTIONS_QUERY = None


try:
    import typing_extensions as _typing_extensions
except Exception:  # pragma: no cover - optional runtime dependency in degraded envs
    _typing_extensions: ModuleType | None = None
