from __future__ import annotations

import builtins
import importlib
import re
import typing

import typing_extensions
from pydantic import BaseModel

BUILTIN_NAMESPACES = ("builtins", "typing", "typing_extensions")


class ValidationResult(BaseModel):
    """
    Result of validating a type annotation string and resolving its Python type.
    """

    validated_type: str
    errors: list[Exception] = []
    pytype: typing.Any | None = None
    invalid_names: list[str] = []
    type_map: dict[str, str] = {}

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __bool__(self) -> bool:  # pragma: no cover - simple convenience
        return not self.invalid_names and self.pytype is not None

    @property
    def is_valid(self) -> bool:
        return bool(self)

    def get_origin(self, type_name: str) -> str | None:  # pragma: no cover
        return self.type_map.get(type_name)


class TypeValidator:
    """
    Validate and resolve names in a type annotation string to real Python types.
    """

    def __init__(self, annotation: str):
        assert annotation, "annotation can not be empty"
        self.annotation = annotation
        self.validated_type = annotation
        self.pattern = re.compile(
            r'''[bfr]*(?:""".*?"""|\'\'\'.*?\'\'\'|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')|(?<!\w)(?<!\.)([a-zA-Z_][a-zA-Z0-9_]*)(?!\.)(?!\w)'''
        )
        self.namespaces: dict[str, typing.Any] = {}

    def only(self, iter):  # pragma: no cover - trivial
        return list(set([x for x in iter if x]))

    def find_names(self, annotation: str | None = None) -> list[str]:
        return self.pattern.findall(annotation or self.annotation)

    def seperate_names(self) -> list[str]:  # keep spelling from provided code
        names = self.find_names()
        seperated: list[str] = []
        for i, name in enumerate(list(names)):
            sub_names = self.find_names(name)
            if len(sub_names) > 1:
                seperated.extend(sub_names)
                try:
                    names.pop(i)
                except Exception:
                    pass
        seperated.extend(names)
        return self.only(seperated)

    def validate_names(self) -> ValidationResult:
        errors: list[Exception] = []
        type_map: dict[str, str] = {}
        invalid_names: list[str] = []
        all_types = self.seperate_names()

        for type_ in all_types:
            if "." in type_ and type_ in BUILTIN_NAMESPACES:
                continue
            entity_info: tuple[str, typing.Any] | None = None
            NotFoundSentinel = object()
            for ns in BUILTIN_NAMESPACES:
                try:
                    module = self.namespaces.get(ns) or importlib.import_module(ns)
                    self.namespaces[ns] = module
                except Exception as e:  # pragma: no cover - defensive
                    continue
                entity = getattr(module, type_, NotFoundSentinel)
                if entity is not NotFoundSentinel:
                    entity_info = (ns, entity)
                    break

            if entity_info is None:
                invalid_names.append(type_)
                continue

            origin, _ = entity_info
            if origin:
                entity_repr = f"{origin}.{type_}" if origin else type_
                type_map[type_] = origin
            else:
                type_map[type_] = ""
                entity_repr = type_

            self.validated_type = re.sub(
                rf"(?<!\.)(?<!\w){re.escape(type_)}\b",
                entity_repr,
                self.validated_type,
            )

        if not invalid_names:
            try:
                pytype = eval(
                    self.validated_type,
                    {
                        "typing": typing,
                        "typing_extensions": typing_extensions,
                        "builtins": builtins,
                    },
                )
            except Exception as e:
                pytype = None
                errors.append(e)
            return ValidationResult(
                validated_type=self.validated_type,
                pytype=pytype,
                errors=errors,
                type_map=type_map,
                invalid_names=invalid_names,
            )
        else:
            return ValidationResult(
                validated_type=self.validated_type,
                pytype=None,
                type_map=type_map,
                errors=errors,
                invalid_names=invalid_names,
            )
