"""Service for resolving type annotations from AST to Python types."""

from __future__ import annotations

from ...lang.ast import ProcedureDef
from ...lang.compiler.environment import KediRuntime
from ...lang.compiler.type_system import KediType


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
    ) -> dict[str, type]:
        """Resolve parameter type annotations to Python types.

        Uses the runtime's type environment to evaluate type annotation strings,
        supporting custom types, Pydantic models, and standard Python types.

        Args:
            procedure: Procedure definition with param_type_annotations

        Returns:
            Dictionary mapping parameter names to Python types.
            Parameters without annotations or unresolvable types are omitted.
        """
        param_types: dict[str, type] = {}

        if not procedure.param_type_annotations:
            return param_types

        # Resolve each parameter's type annotation
        for param_name in procedure.params:
            type_anno = procedure.param_type_annotations.get(param_name)
            if not type_anno:
                continue

            try:
                resolved_type = KediType(type_anno).resolve(self.runtime, {})

                # Validate it's actually a type
                if isinstance(resolved_type, type):
                    param_types[param_name] = resolved_type
                # Handle typing constructs (List, Dict, etc.)
                elif hasattr(resolved_type, "__origin__") or str(type(resolved_type)).startswith(
                    "<class 'typing."
                ):
                    # Store typing constructs as-is (they're still valid)
                    param_types[param_name] = resolved_type  # type: ignore
            except Exception:
                # Type resolution failed - skip this parameter
                # Optimizer will skip type validation for unresolvable types
                pass

        return param_types

    def resolve_output_types(
        self,
        output_fields: list[tuple[str, str | None]],
    ) -> dict[str, type]:
        """Resolve output field type annotations to Python types.

        Args:
            output_fields: List of (field_name, type_annotation) tuples

        Returns:
            Dictionary mapping field names to Python types
        """
        output_types: dict[str, type] = {}

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

                # Validate it's actually a type
                if isinstance(resolved_type, type):
                    output_types[field_name] = resolved_type
                # Handle typing constructs
                elif hasattr(resolved_type, "__origin__") or str(type(resolved_type)).startswith(
                    "<class 'typing."
                ):
                    output_types[field_name] = resolved_type  # type: ignore
                else:
                    # Fallback to str if type resolution fails
                    output_types[field_name] = str
            except Exception:
                # Type resolution failed - default to str
                output_types[field_name] = str

        return output_types
