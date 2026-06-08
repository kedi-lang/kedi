"""Service for resolving type annotations from AST to Python types."""

from __future__ import annotations

from ...lang.ast import ProcedureDef
from ...lang.compiler.environment import KediRuntime
from ...lang.compiler.type_system import KediType
from ...resolved_types import ResolvedKediType, as_resolved_kedi_type


class TypeResolver:
    """Service for resolving type annotations from AST to Python types.

    Handles conversion of Kedi type annotation strings to actual Python types,
    including custom types defined in the program (Pydantic models, etc.).
    """

    def __init__(self, runtime: KediRuntime):
        """Initialize type resolver.

        Args:
            runtime: Compiled Kedi runtime with type environment
        """
        self.runtime = runtime

    def resolve_param_types(
        self,
        procedure: ProcedureDef,
    ) -> dict[str, ResolvedKediType]:
        """Resolve parameter type annotations to Python types.

        Uses the runtime's type environment to evaluate type annotation strings,
        supporting custom types, Pydantic models, and standard Python types.

        Args:
            procedure: Procedure definition with param_type_annotations

        Returns:
            Dictionary mapping parameter names to Python types.
            Parameters without annotations or unresolvable types are omitted.
        """
        param_types: dict[str, ResolvedKediType] = {}

        if not procedure.param_type_annotations:
            return param_types

        # Resolve each parameter's type annotation
        for param_name in procedure.params:
            type_anno = procedure.param_type_annotations.get(param_name)
            if not type_anno:
                continue

            try:
                resolved_type = KediType(type_anno).resolve(self.runtime, {})

                stored_type = as_resolved_kedi_type(resolved_type)
                if stored_type is not None:
                    param_types[param_name] = stored_type
            except Exception:
                # Type resolution failed - skip this parameter
                # Optimizer will skip type validation for unresolvable types
                pass

        return param_types

    def resolve_output_types(
        self,
        output_fields: list[tuple[str, str | None]],
    ) -> dict[str, ResolvedKediType]:
        """Resolve output field type annotations to Python types.

        Args:
            output_fields: List of (field_name, type_annotation) tuples

        Returns:
            Dictionary mapping field names to Python types
        """
        output_types: dict[str, ResolvedKediType] = {}

        if not output_fields:
            return output_types

        # Resolve each output field's type annotation
        for field_name, type_anno in output_fields:
            if not type_anno:
                # Default to str if no annotation
                output_types[field_name] = str
                continue

            try:
                resolved_type = KediType(type_anno).resolve(self.runtime, {})

                stored_type = as_resolved_kedi_type(resolved_type)
                if stored_type is not None:
                    output_types[field_name] = stored_type
                else:
                    # Fallback to str if type resolution fails
                    output_types[field_name] = str
            except Exception:
                # Type resolution failed - default to str
                output_types[field_name] = str

        return output_types
