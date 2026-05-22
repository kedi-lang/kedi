"""Helpers for Python-facing Kedi procedure introspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .ast import ProcedureDef
from .type_render import procedure_param_type_to_python, procedure_return_type_to_python

_PROCEDURE_METADATA_ATTR = "__kedi_procedure_metadata__"


@dataclass(frozen=True)
class ProcedurePythonMetadata:
    name: str
    params: tuple[str, ...]
    param_types: tuple[str, ...]
    return_type: str

    def _signature(self) -> str:
        annotated_params = ", ".join(
            f"{name}: {typ}" for name, typ in zip(self.params, self.param_types, strict=False)
        )
        return f"def {self.name}({annotated_params}) -> {self.return_type}:"

    def display_source(self) -> str:
        args = ", ".join(self.params)
        call = f"__kedi_call__({args})" if args else "__kedi_call__()"
        return (
            f"{self._signature()}\n"
            '    """This stub is generated for Python introspection."""\n'
            f"    return {call}\n"
        )

    def executable_source(self) -> str:
        args = ", ".join(self.params)
        call = f"__kedi_call__({args})" if args else "__kedi_call__()"
        return f"{self._signature()}\n    return {call}\n"


def attach_procedure_metadata(fn: Callable[..., Any], proc: ProcedureDef) -> Callable[..., Any]:
    metadata = ProcedurePythonMetadata(
        name=proc.name,
        params=tuple(proc.params),
        param_types=tuple(
            procedure_param_type_to_python(proc, param_name, default="str")
            for param_name in proc.params
        ),
        return_type=procedure_return_type_to_python(proc, default="str"),
    )
    setattr(fn, _PROCEDURE_METADATA_ATTR, metadata)
    return fn


def get_procedure_metadata(fn: Callable[..., Any]) -> ProcedurePythonMetadata | None:
    metadata = getattr(fn, _PROCEDURE_METADATA_ATTR, None)
    return metadata if isinstance(metadata, ProcedurePythonMetadata) else None
