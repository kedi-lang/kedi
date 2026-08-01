from __future__ import annotations

from typing import Any, Literal

from kedi.agent_adapter.tool_override import ToolSpec

from .codemode import artifact_code_provenance, run_artifact_code as execute_artifact_code
from .manager import ArtifactManager
from .models import ArtifactHandle
from .policy import ArtifactPolicy


def artifact_tool_specs(
    manager: ArtifactManager,
    policy: ArtifactPolicy | None = None,
) -> tuple[ToolSpec, ...]:
    """Build artifact management tools bound to one runtime-owned session."""

    def read_artifact(
        *,
        ref_id: str,
        max_chars: int = -1,
        offset: int = 0,
        offset_from: Literal["start", "end"] = "start",
        path: str | None = None,
        pattern: str | None = None,
        max_matches: int = 20,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Read a bounded chunk from an artifact in the active session."""

        if pattern is not None:
            if offset != 0 or offset_from != "start":
                raise ValueError("artifact pattern search cannot be combined with an offset")
            if path is not None:
                raise ValueError("artifact pattern search does not support JSON Pointer path")
            return manager.find(ref_id, pattern, max_matches=max_matches)
        return manager.read(
            ref_id,
            max_chars=max_chars,
            offset=offset,
            offset_from=offset_from,
            path=path,
        ).model_dump(mode="json")

    def search_artifacts(
        *,
        query: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search active artifact metadata without reading payloads."""

        return [item.model_dump(mode="json") for item in manager.search(query, limit=limit)]

    def release_artifact(*, ref_id: str) -> dict[str, Any]:
        """Release an artifact payload without rewriting conversation history."""

        return manager.release(ref_id).model_dump(mode="json")

    def run_artifact_code(
        *,
        code: str,
        artifact_refs: list[str],
    ) -> dict[str, Any]:
        """Run bounded Python over explicitly allowed artifact references."""

        execution = execute_artifact_code(
            manager=manager,
            code=code,
            artifact_refs=artifact_refs,
        )
        effective_policy = (policy or ArtifactPolicy()).resolved()
        stored = manager.artifactize(
            execution.result,
            policy=effective_policy,
            source="artifact_code",
            provenance=artifact_code_provenance(code, execution.source_refs),
        )
        result = (
            stored.as_ref().model_dump(mode="json")
            if isinstance(stored, ArtifactHandle)
            else stored
        )
        return execution.model_copy(update={"result": result}).model_dump(mode="json")

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
                    "offset_from": {
                        "type": "string",
                        "enum": ["start", "end"],
                        "default": "start",
                    },
                    "path": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "pattern": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Optional literal substring to find in the artifact.",
                    },
                    "max_matches": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
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
            fn=release_artifact,
            name="release_artifact",
            description=(
                "Release an artifact after its needed content has been consumed and will not be "
                "needed again. This frees payload quota while preserving its history reference."
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
        ToolSpec(
            fn=run_artifact_code,
            name="run_artifact_code",
            description=(
                "Run bounded Python to inspect, filter, join, or aggregate explicitly listed "
                "artifacts. Sandbox helpers: artifact_metadata(ref_id) -> dict; "
                "read_artifact(ref_id, offset=0, offset_from='start', max_chars=-1) -> page "
                "dict (text is page['content']); find_artifact(ref_id, pattern, "
                "max_matches=20, context_chars=120) -> match dicts; "
                "iter_artifact(ref_id, chunk_chars=4000) -> text chunks; and "
                "get_artifact(ref_id) -> the small native value. Prefer find_artifact for a "
                "known literal. The sandbox has no imports, filesystem, network, model, or "
                "subagent access."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code whose final expression is the result. Artifact helper "
                            "functions are already available; do not pass read_artifact the "
                            "outer tool's path, pattern, or max_matches arguments."
                        ),
                    },
                    "artifact_refs": {
                        "type": "array",
                        "items": ref_id_schema,
                        "maxItems": 32,
                        "description": "Exact artifact refs that this execution may access.",
                    },
                },
                "required": ["code", "artifact_refs"],
                "additionalProperties": False,
            },
            metadata=metadata,
            risk="read_only",
            artifact_result="never",
        ),
    )
