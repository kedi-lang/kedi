from __future__ import annotations

from typing import Any

from kedi.agent_adapter.tool_override import ToolSpec

from .manager import ArtifactManager


def artifact_tool_specs(manager: ArtifactManager) -> tuple[ToolSpec, ToolSpec, ToolSpec]:
    """Build artifact management tools bound to one runtime-owned session."""

    def read_artifact(
        *,
        ref_id: str,
        max_chars: int = -1,
        offset: int = 0,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Read a bounded chunk from an artifact in the active session."""

        return manager.read(
            ref_id,
            max_chars=max_chars,
            offset=offset,
            path=path,
        ).model_dump(mode="json")

    def search_artifacts(
        *,
        query: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search active artifact metadata without reading payloads."""

        return [item.model_dump(mode="json") for item in manager.search(query, limit=limit)]

    def invalidate_artifact(*, ref_id: str) -> dict[str, Any]:
        """Invalidate an artifact and compact its linked portable history."""

        return manager.release(ref_id).model_dump(mode="json")

    metadata = {
        "kind": "artifact_management",
        "kedi_tool_kind": "artifact_management",
    }
    ref_id_schema = {"type": "string", "description": "Artifact reference ID."}
    return (
        ToolSpec(
            fn=read_artifact,
            name="read_artifact",
            description=(
                "Read a bounded artifact chunk. Use offset for pagination and an RFC 6901 "
                "JSON Pointer path to select structured content."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "ref_id": ref_id_schema,
                    "max_chars": {
                        "type": "integer",
                        "default": -1,
                        "description": "-1 uses the configured bounded read limit.",
                    },
                    "offset": {"type": "integer", "default": 0, "minimum": 0},
                    "path": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                },
                "required": ["ref_id"],
                "additionalProperties": False,
            },
            metadata=metadata,
            risk="read_only",
            artifact_result="never",
        ),
        ToolSpec(
            fn=search_artifacts,
            name="search_artifacts",
            description="Search active artifact summaries and metadata without reading payloads.",
            json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "additionalProperties": False,
            },
            metadata=metadata,
            risk="read_only",
            artifact_result="never",
        ),
        ToolSpec(
            fn=invalidate_artifact,
            name="invalidate_artifact",
            description=(
                "Invalidate an artifact that is no longer needed and compact its linked history."
            ),
            json_schema={
                "type": "object",
                "properties": {"ref_id": ref_id_schema},
                "required": ["ref_id"],
                "additionalProperties": False,
            },
            metadata=metadata,
            risk="mutating",
            artifact_result="never",
        ),
    )
