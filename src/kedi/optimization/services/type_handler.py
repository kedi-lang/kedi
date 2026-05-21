"""Service for handling structured types using adapters."""

from __future__ import annotations

import asyncio

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore

from ...agent_adapter.adapters import DSPyAdapter
from ...lang.compiler.environment import KediRuntime
from ...lang.compiler.type_system import KediType


class StructuredTypeHandler:
    """Service that delegates to adapters for creating DSPy signatures from structured types.

    Uses the adapter pattern to handle complex types (Pydantic models, custom types)
    by delegating to DSPyAdapter's type_builder method.
    """

    def __init__(self, runtime: KediRuntime):
        """Initialize structured type handler.

        Args:
            runtime: Compiled Kedi runtime with type environment
        """
        self.runtime = runtime

    def build_output_schema(
        self,
        output_fields: list[tuple[str, str | None]],
    ) -> dict[str, type]:
        """Convert Kedi output fields to output schema dict.

        Resolves type annotations from AST to Python types using the runtime's
        type environment.

        Args:
            output_fields: List of (field_name, type_annotation) tuples

        Returns:
            Dictionary mapping field names to Python types
        """
        output_schema: dict[str, type] = {}

        for field_name, type_anno in output_fields:
            if not type_anno:
                # Default to str if no annotation
                output_schema[field_name] = str
                continue

            try:
                resolved_type = KediType(type_anno).resolve(self.runtime, {})

                # Validate it's a type or typing construct
                if isinstance(resolved_type, type):
                    output_schema[field_name] = resolved_type
                elif hasattr(resolved_type, "__origin__") or str(
                    type(resolved_type)
                ).startswith("<class 'typing."):
                    # Handle typing constructs (List, Dict, etc.)
                    output_schema[field_name] = resolved_type  # type: ignore
                else:
                    # Fallback to str
                    output_schema[field_name] = str
            except Exception:
                # Type resolution failed - default to str
                output_schema[field_name] = str

        return output_schema

    async def create_dspy_signature(
        self,
        inputs: list[str],
        output_fields: list[tuple[str, str | None]],
        use_string_signature: bool = True,
    ) -> dspy.Signature | str:
        """Create DSPy Signature using adapter pattern.

        Delegates to DSPyAdapter.type_builder() to handle structured types
        (Pydantic models, custom types) correctly.

        Args:
            inputs: List of input parameter names
            output_fields: List of (field_name, type_annotation) tuples
            use_string_signature: If True, use string-based signature (simpler for GEPA)

        Returns:
            DSPy Signature object or string signature

        Raises:
            ImportError: If DSPy is not available
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for signature creation")

        # Build output schema from fields
        output_schema = self.build_output_schema(output_fields)

        if not output_schema:
            raise ValueError("Output schema cannot be empty")

        # Use string signatures - DSPy handles type parsing automatically
        # Delegate to adapter for type information, then adapt to our inputs
        signature = await DSPyAdapter.type_builder(
            output_schema,
            string=True,
        )

        # Adapt string signature to use our inputs instead of "user_prompt"
        # CRITICAL: Preserve type annotations from adapter so DSPy can parse custom types correctly
        if isinstance(signature, str):
            input_str = ", ".join(inputs)

            # Extract type annotations from adapter signature
            # Format: "user_prompt: str -> coord: Coordinates"
            if " -> " in signature:
                adapter_output_part = signature.split(" -> ", 1)[1]
                # Parse individual output annotations: ["coord: Coordinates"]
                adapter_outputs = [s.strip() for s in adapter_output_part.split(",")]

                # Map adapter outputs to our field names, preserving type annotations
                output_parts = []
                for idx, ((out_name, _), adapter_output) in enumerate(
                    zip(output_fields, adapter_outputs)
                ):
                    if ":" in adapter_output:
                        # Extract type annotation: "coord: Coordinates" -> "Coordinates"
                        type_part = adapter_output.split(":", 1)[1].strip()
                        output_parts.append(f"{out_name}: {type_part}")
                    else:
                        # No type in adapter output - try to get from our schema
                        # Get field name from adapter output (before ":")
                        adapter_field_name = (
                            adapter_output.split(":")[0].strip()
                            if ":" in adapter_output
                            else adapter_output.strip()
                        )
                        # Find matching field in output_schema
                        matching_field = None
                        for schema_name in output_schema.keys():
                            if (
                                schema_name == adapter_field_name
                                or schema_name == out_name
                            ):
                                matching_field = schema_name
                                break

                        if matching_field and matching_field in output_schema:
                            type_name = (
                                output_schema[matching_field].__name__
                                if hasattr(output_schema[matching_field], "__name__")
                                else str(output_schema[matching_field])
                            )
                            output_parts.append(f"{out_name}: {type_name}")
                        else:
                            output_parts.append(out_name)

                output_str = ", ".join(output_parts)
            else:
                # No " -> " found - build from output_schema directly
                output_parts = []
                for out_name, _ in output_fields:
                    if out_name in output_schema:
                        type_name = (
                            output_schema[out_name].__name__
                            if hasattr(output_schema[out_name], "__name__")
                            else str(output_schema[out_name])
                        )
                        output_parts.append(f"{out_name}: {type_name}")
                    else:
                        output_parts.append(out_name)
                output_str = ", ".join(output_parts)

            return f"{input_str} -> {output_str}"

        # Fallback
        input_str = ", ".join(inputs)
        output_names = [name for name, _ in output_fields]
        output_str = ", ".join(output_names)
        return f"{input_str} -> {output_str}"

    def create_dspy_signature_sync(
        self,
        inputs: list[str],
        output_fields: list[tuple[str, str | None]],
        use_string_signature: bool = True,
    ) -> str:
        """Synchronous wrapper for create_dspy_signature.

        Args:
            inputs: List of input parameter names
            output_fields: List of (field_name, type_annotation) tuples
            use_string_signature: If True, use string-based signature

        Returns:
            DSPy Signature object or string signature
        """
        return asyncio.run(
            self.create_dspy_signature(inputs, output_fields, use_string_signature)
        )
