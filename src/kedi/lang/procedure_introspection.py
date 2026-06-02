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
    param_defaults: tuple[str | None, ...]
    default_values: tuple[Any | None, ...]
    return_type: str
    docstring: str | None

    def _signature(self) -> str:
        annotated_params = ", ".join(
            self._param_signature(name, typ, default)
            for name, typ, default in zip(
                self.params, self.param_types, self.param_defaults, strict=False
            )
        )
        return f"def {self.name}({annotated_params}) -> {self.return_type}:"

    def _param_signature(self, name: str, typ: str, default_source: str | None) -> str:
        piece = f"{name}: {typ}"
        if default_source is not None:
            piece += f" = {default_source}"
        return piece

    def _executable_signature(self) -> str:
        annotated_params = ", ".join(
            self._param_signature(
                name,
                typ,
                f"__kedi_default_{index}" if default is not None else None,
            )
            for index, (name, typ, default) in enumerate(
                zip(self.params, self.param_types, self.param_defaults, strict=False)
            )
        )
        return f"def {self.name}({annotated_params}) -> {self.return_type}:"

    def default_namespace(self) -> dict[str, Any]:
        return {
            f"__kedi_default_{index}": value
            for index, (source, value) in enumerate(
                zip(self.param_defaults, self.default_values, strict=False)
            )
            if source is not None
        }

    def display_source(self) -> str:
        args = ", ".join(self.params)
        call = f"__kedi_call__({args})" if args else "__kedi_call__()"
        lines = [self._signature()]
        if self.docstring:
            lines.extend(_render_docstring_lines(self.docstring, "    "))
        lines.append(f"    return {call}")
        return "\n".join(lines) + "\n"

    def executable_source(self) -> str:
        args = ", ".join(self.params)
        call = f"__kedi_call__({args})" if args else "__kedi_call__()"
        return f"{self._executable_signature()}\n    return {call}\n"


def attach_procedure_metadata(
    fn: Callable[..., Any],
    proc: ProcedureDef,
    defaults: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    param_defaults = proc.param_defaults or {}
    default_values = defaults or {}
    metadata = ProcedurePythonMetadata(
        name=proc.name,
        params=tuple(proc.params),
        param_types=tuple(
            procedure_param_type_to_python(proc, param_name, default="str")
            for param_name in proc.params
        ),
        param_defaults=tuple(param_defaults.get(param_name) for param_name in proc.params),
        default_values=tuple(default_values.get(param_name) for param_name in proc.params),
        return_type=procedure_return_type_to_python(proc, default="str"),
        docstring=proc.docstring,
    )
    setattr(fn, _PROCEDURE_METADATA_ATTR, metadata)
    setattr(fn, "__kedi_procedure_def__", proc)
    fn.__doc__ = proc.docstring
    return fn


def get_procedure_metadata(fn: Callable[..., Any]) -> ProcedurePythonMetadata | None:
    metadata = getattr(fn, _PROCEDURE_METADATA_ATTR, None)
    return metadata if isinstance(metadata, ProcedurePythonMetadata) else None


def _render_docstring_lines(docstring: str, indent: str) -> list[str]:
    escaped = docstring.replace('"""', '\\"""')
    lines = escaped.splitlines() or [""]
    if len(lines) == 1:
        return [f'{indent}"""{lines[0]}"""']
    out = [f'{indent}"""{lines[0]}']
    out.extend(f"{indent}{line}" for line in lines[1:-1])
    out.append(f'{indent}{lines[-1]}"""')
    return out
