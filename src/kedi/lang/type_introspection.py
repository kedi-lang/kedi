"""Helpers for Python-facing Kedi type introspection."""

from __future__ import annotations

import copy
import linecache
import sys
import types
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .ast import TypeDef
from .type_render import type_expr_to_python


@dataclass(frozen=True)
class TypeInitMetadata:
    name: str
    fields: tuple[str, ...]
    field_types: tuple[str, ...]
    default_sources: tuple[str | None, ...]
    default_values: tuple[Any | None, ...]

    def display_source(self) -> str:
        return (
            f"{self._signature()}\n"
            '    """This stub is generated for Python introspection."""\n'
            f"    {self._init_call()}\n"
        )

    def display_class_source(self) -> str:
        lines = [
            f"class {self.name}:",
            '    """This stub is generated for Python introspection."""',
        ]
        field_lines = self._class_field_lines()
        if field_lines:
            lines.extend(field_lines)
            lines.append("")
        lines.extend(
            [
                f"    {self._new_signature()}",
                '        """This stub is generated for Python introspection."""',
                f"        {self._new_call()}",
                "",
                f"    {self._signature()}",
                '        """This stub is generated for Python introspection."""',
                f"        {self._init_call()}",
            ]
        )
        return "\n".join(lines) + "\n"

    def executable_source(self) -> str:
        lines = [self._executable_signature()]
        for index, (field, default_source) in enumerate(
            zip(self.fields, self.default_sources, strict=False)
        ):
            if default_source is not None:
                lines.append(f"    if {field} is __kedi_default_{index}:")
                lines.append(f"        {field} = __kedi_default_{index}.value()")
        lines.append(f"    {self._init_call()}")
        return "\n".join(lines) + "\n"

    def display_new_source(self) -> str:
        return (
            f"{self._new_signature()}\n"
            '    """This stub is generated for Python introspection."""\n'
            f"    {self._new_call()}\n"
        )

    def executable_new_source(self) -> str:
        return f"{self._new_signature(executable=True)}\n    {self._new_call()}\n"

    def executable_class_source(self) -> str:
        lines = [
            f"class {self.name}:",
            '    """This stub is generated for Python introspection."""',
            *self._class_field_lines(),
            f"    {self._new_signature()}",
            f"        {self._new_call()}",
            f"    {self._signature()}",
            '        """This stub is generated for Python introspection."""',
            f"        {self._init_call()}",
        ]
        return "\n".join(lines) + "\n"

    def default_namespace(self) -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        for index, (source, value) in enumerate(
            zip(self.default_sources, self.default_values, strict=False)
        ):
            if source is None:
                continue
            default_value = _DefaultValue(source, value)
            namespace[f"__kedi_default_{index}"] = default_value
            namespace[self._class_runtime_default_name(index)] = default_value
        return namespace

    def _signature(self) -> str:
        params = ", ".join(
            self._param_signature(field, typ, default)
            for field, typ, default in zip(
                self.fields, self.field_types, self.default_sources, strict=False
            )
        )
        if not params:
            return "def __init__(self) -> None:"
        return f"def __init__(self, {params}) -> None:"

    def _executable_signature(self) -> str:
        params = ", ".join(
            self._param_signature(
                field,
                typ,
                f"__kedi_default_{index}" if default is not None else None,
            )
            for index, (field, typ, default) in enumerate(
                zip(self.fields, self.field_types, self.default_sources, strict=False)
            )
        )
        if not params:
            return "def __init__(self) -> None:"
        return f"def __init__(self, {params}) -> None:"

    def _new_signature(self, *, executable: bool = False) -> str:
        params = ", ".join(
            self._param_signature(
                field,
                typ,
                (
                    self._class_runtime_default_name(index)
                    if executable and default is not None
                    else default
                ),
            )
            for index, (field, typ, default) in enumerate(
                zip(self.fields, self.field_types, self.default_sources, strict=False)
            )
        )
        if not params:
            return f'def __new__(cls) -> "{self.name}":'
        return f'def __new__(cls, {params}) -> "{self.name}":'

    def _param_signature(self, field: str, typ: str, default_source: str | None) -> str:
        piece = f"{field}: {typ}"
        if default_source is not None:
            piece += f" = {default_source}"
        return piece

    def _keyword_args(self) -> str:
        return ", ".join(f"{field}={field}" for field in self.fields)

    def _init_call(self) -> str:
        kwargs = self._keyword_args()
        if not kwargs:
            return "__kedi_model_init__(self)"
        return f"__kedi_model_init__(self, {kwargs})"

    def _new_call(self) -> str:
        kwargs = self._keyword_args()
        if not kwargs:
            return "return __kedi_model_new__(cls)"
        return f"return __kedi_model_new__(cls, {kwargs})"

    def _class_runtime_default_name(self, index: int) -> str:
        return f"kedi_default_{index}"

    def _class_field_lines(self) -> list[str]:
        lines: list[str] = []
        for field, typ, default in zip(
            self.fields, self.field_types, self.default_sources, strict=False
        ):
            line = f"    {field}: {typ}"
            if default is not None:
                line += f" = {default}"
            lines.append(line)
        return lines


class _DefaultValue:
    def __init__(self, source: str, value: Any) -> None:
        self._source = source
        self._value = copy.deepcopy(value)

    def value(self) -> Any:
        return copy.deepcopy(self._value)

    def __repr__(self) -> str:
        return self._source


def make_type_init(
    typedef: TypeDef,
    model_class: type[BaseModel],
    *,
    default_values: dict[str, Any],
    namespace: dict[str, Any],
):
    metadata = TypeInitMetadata(
        name=typedef.name,
        fields=tuple(field.name for field in typedef.fields),
        field_types=tuple(
            type_expr_to_python(field.type_expr, field.type_annotation, default="str")
            for field in typedef.fields
        ),
        default_sources=tuple(field.default_expr for field in typedef.fields),
        default_values=tuple(default_values.get(field.name) for field in typedef.fields),
    )
    class_source = metadata.display_class_source()
    init_source = metadata.display_source()
    new_source = metadata.display_new_source()
    class_filename = f"<kedi-type-class:{typedef.name}:{id(model_class)}>"
    init_filename = f"<kedi-type-init:{typedef.name}:{id(model_class)}>"
    new_filename = f"<kedi-type-new:{typedef.name}:{id(model_class)}>"
    _cache_source(class_filename, class_source)
    _cache_source(init_filename, init_source)
    _cache_source(new_filename, new_source)

    def init_model(instance: BaseModel, **kwargs: Any) -> None:
        super(model_class, instance).__init__(**kwargs)

    def new_model(cls: type[BaseModel], **_: Any) -> BaseModel:
        return object.__new__(cls)

    default_namespace = metadata.default_namespace()
    init_namespace = dict(namespace)
    init_namespace.update(default_namespace)
    init_namespace["__kedi_model_init__"] = init_model
    exec(
        compile(metadata.executable_source(), init_filename, "exec", dont_inherit=True),
        init_namespace,
        init_namespace,
    )
    new_namespace = dict(namespace)
    new_namespace.update(default_namespace)
    new_namespace["__kedi_model_new__"] = new_model
    exec(
        compile(
            metadata.executable_new_source(),
            new_filename,
            "exec",
            dont_inherit=True,
        ),
        new_namespace,
        new_namespace,
    )

    module_name = f"_kedi_type_introspection_{typedef.name}_{id(model_class)}"
    module = types.ModuleType(module_name)
    module.__file__ = class_filename
    module.__dict__[metadata.name] = model_class
    sys.modules[module_name] = module

    init = init_namespace["__init__"]
    new = new_namespace["__new__"]
    init.__qualname__ = f"{typedef.name}.__init__"
    init.__module__ = module_name
    init.__doc__ = "This stub is generated for Python introspection."
    new.__qualname__ = f"{typedef.name}.__new__"
    new.__module__ = module_name
    new.__doc__ = "This stub is generated for Python introspection."
    model_class.__module__ = module_name
    model_class.__qualname__ = typedef.name
    model_class.__doc__ = "This stub is generated for Python introspection."
    setattr(model_class, "__new__", staticmethod(new))
    return init


def _cache_source(filename: str, source: str) -> None:
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )
