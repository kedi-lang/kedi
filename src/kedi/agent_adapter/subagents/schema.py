"""Bounded, document-local JSON Schema normalization for child results."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote

from jsonschema import Draft3Validator, Draft4Validator, Draft6Validator, Draft7Validator
from jsonschema.validators import validator_for

MAX_FINAL_SCHEMA_BYTES = 32 * 1024
MAX_FINAL_SCHEMA_DEPTH = 16

_SCHEMA_MAPS = frozenset(
    {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas", "dependencies"}
)
_SCHEMA_VALUES = frozenset(
    {
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
        "contains",
        "items",
        "additionalItems",
        "unevaluatedItems",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
        "type",
        "disallow",
        "extends",
    }
)


def validate_final_schema(schema: dict[str, Any]) -> None:
    normalize_final_schema(schema)


def normalize_final_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local pointers without mutating input or fetching external resources."""
    if not isinstance(schema, dict):
        raise ValueError(
            "final_schema must be a JSON object, not a JSON-encoded string or another value. "
            "Omit final_schema to use the child profile's configured output type."
        )
    try:
        encoded = json.dumps(schema, ensure_ascii=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("final_schema must be JSON serializable") from exc
    if len(encoded) > MAX_FINAL_SCHEMA_BYTES:
        raise ValueError(f"final_schema exceeds the {MAX_FINAL_SCHEMA_BYTES}-byte size limit")
    _check_source(schema, schema_node=True)
    validator_for(schema).check_schema(schema)
    result = _SchemaNormalizer(schema).copy(schema, schema_node=True)
    # A root reference may target a boolean schema; keep the public object contract.
    if isinstance(result, bool):
        result = {} if result else {"not": {}}
    if "$schema" in schema:
        result["$schema"] = schema["$schema"]
    if len(json.dumps(result, ensure_ascii=True, separators=(",", ":"))) > MAX_FINAL_SCHEMA_BYTES:
        raise ValueError(
            f"expanded final_schema exceeds the {MAX_FINAL_SCHEMA_BYTES}-byte size limit"
        )
    validator_for(result).check_schema(result)
    return result


def _check_source(
    value: Any,
    *,
    depth: int = 0,
    schema_node: bool = False,
    schema_map: bool = False,
) -> None:
    _check_depth(depth)
    if isinstance(value, dict):
        if schema_node:
            for keyword in ("$id", "$dynamicRef", "$recursiveRef"):
                if keyword in value:
                    raise ValueError(f"final_schema does not support {keyword}")
            if "$ref" in value:
                reference = value["$ref"]
                local = isinstance(reference, str) and reference.startswith("#")
                pointer = unquote(reference[1:], errors="strict") if local else None
                if pointer is None or (pointer and not pointer.startswith("/")):
                    raise ValueError(
                        f"final_schema does not support $ref outside local JSON pointers: {reference!r}"
                    )
        for key, child in value.items():
            _check_source(
                child,
                depth=depth + 1,
                schema_node=schema_map or (schema_node and key in _SCHEMA_VALUES),
                schema_map=schema_node and key in _SCHEMA_MAPS,
            )
    elif isinstance(value, list):
        for child in value:
            _check_source(child, depth=depth + 1, schema_node=schema_node)


def _check_depth(depth: int) -> None:
    if depth > MAX_FINAL_SCHEMA_DEPTH:
        raise ValueError(
            f"final_schema exceeds the maximum nesting depth of {MAX_FINAL_SCHEMA_DEPTH}"
        )


class _SchemaNormalizer:
    def __init__(self, root: dict[str, Any]) -> None:
        self.root = root
        self.remaining = MAX_FINAL_SCHEMA_BYTES
        self.ignore_ref_siblings = validator_for(root) in (
            Draft3Validator,
            Draft4Validator,
            Draft6Validator,
            Draft7Validator,
        )

    def _spend(self, size: int) -> None:
        self.remaining -= size
        if self.remaining < 0:
            raise ValueError(
                f"expanded final_schema exceeds the {MAX_FINAL_SCHEMA_BYTES}-byte size limit"
            )

    def _target(self, reference: str) -> Any:
        target: Any = self.root
        pointer = unquote(reference[1:], errors="strict")
        try:
            for raw_token in pointer.split("/")[1:]:
                if "~" in raw_token.replace("~1", "").replace("~0", ""):
                    raise ValueError(f"Invalid final_schema JSON pointer: {reference!r}")
                token = raw_token.replace("~1", "/").replace("~0", "~")
                if isinstance(target, dict):
                    target = target[token]
                elif (
                    isinstance(target, list)
                    and token.isascii()
                    and token.isdecimal()
                    and (token == "0" or not token.startswith("0"))
                ):
                    target = target[int(token)]
                else:
                    raise KeyError(token)
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unknown final_schema reference: {reference!r}") from exc
        if not isinstance(target, (dict, bool)):
            raise ValueError(f"final_schema reference must target a schema: {reference!r}")
        return target

    def copy(
        self,
        value: Any,
        *,
        depth: int = 0,
        active: frozenset[str] = frozenset(),
        schema_node: bool = False,
        schema_map: bool = False,
    ) -> Any:
        _check_depth(depth)
        if isinstance(value, dict):
            if schema_node and "$ref" in value:
                reference = value["$ref"]
                # Targets reached through pointers need the same checks as inline schemas.
                target = self._target(reference)
                _check_source(target, schema_node=True)
                if reference in active:
                    raise ValueError(
                        f"Recursive final_schema reference is not supported: {reference!r}"
                    )
                siblings = (
                    {
                        key: child
                        for key, child in value.items()
                        if key not in {"$ref", "$defs", "definitions"}
                    }
                    if not self.ignore_ref_siblings
                    else {}
                )
                nested_depth = depth + 2 if siblings else depth
                resolved = self.copy(
                    target, depth=nested_depth, active=active | {reference}, schema_node=True
                )
                if not siblings:
                    return resolved
                # $ref siblings constrain the target conjunctively, not by overriding it.
                self._spend(len('{"allOf":[,]}'))
                return {
                    "allOf": [
                        resolved,
                        self.copy(siblings, depth=nested_depth, active=active, schema_node=True),
                    ]
                }
            entries = [
                (key, child)
                for key, child in value.items()
                if not (schema_node and key in {"$defs", "definitions"})
            ]
            self._spend(2 + max(0, len(entries) - 1))
            result = {}
            for key, child in entries:
                self._spend(len(json.dumps(key, ensure_ascii=True)) + 1)
                result[key] = self.copy(
                    child,
                    depth=depth + 1,
                    active=active,
                    schema_node=schema_map or (schema_node and key in _SCHEMA_VALUES),
                    schema_map=schema_node and key in _SCHEMA_MAPS,
                )
            return result
        if isinstance(value, list):
            self._spend(2 + max(0, len(value) - 1))
            return [
                self.copy(child, depth=depth + 1, active=active, schema_node=schema_node)
                for child in value
            ]
        self._spend(len(json.dumps(value, ensure_ascii=True)))
        return value
