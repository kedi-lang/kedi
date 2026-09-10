from __future__ import annotations


class ArtifactError(RuntimeError):
    """Base error for artifact lifecycle and storage operations."""


class ArtifactInputError(ArtifactError, ValueError):
    """Raised when artifact read/search arguments need correction."""


class ArtifactPolicyError(ArtifactError, ValueError):
    """Raised when artifact policy configuration is invalid."""


class ArtifactNotFoundError(ArtifactError, KeyError):
    """Raised when an artifact reference is unknown to the active session."""


class ArtifactExpiredError(ArtifactError):
    """Raised when an artifact has passed its configured lifetime."""


class ArtifactReleasedError(ArtifactError):
    """Raised when an artifact payload was explicitly released."""


class ArtifactQuotaExceededError(ArtifactError):
    """Raised when storing an artifact would exceed session limits."""


class ToolOutputTooLargeError(ArtifactError):
    """Raised before an oversized tool result can enter model history."""

    def __init__(self, *, size_bytes: int, limit_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"tool output is {size_bytes} bytes, exceeding the inline safety limit of "
            f"{limit_bytes} bytes; enable artifacts or request a narrower result"
        )


class ArtifactSerializationError(ArtifactError, TypeError):
    """Raised when a value cannot be safely represented by a store."""


class ArtifactStreamError(ArtifactSerializationError):
    """Raised when an explicit artifact stream is malformed or misused."""


class ArtifactStreamConsumedError(ArtifactStreamError):
    """Raised when a single-consumption artifact stream is reused."""


class ArtifactAccessError(ArtifactError, PermissionError):
    """Raised when a reference is used outside its owning artifact session."""


class ArtifactCapabilityError(ArtifactError):
    """Raised when the active adapter cannot expose artifact operations."""


class ArtifactCodeError(ArtifactError):
    """Raised when sandboxed artifact code cannot complete safely."""


class ArtifactCodeAccessError(ArtifactCodeError, PermissionError):
    """Raised when sandboxed code accesses a ref outside its allowlist."""


class ArtifactCodeBudgetError(ArtifactCodeError):
    """Raised when sandboxed artifact processing exceeds a host budget."""


def artifact_tool_error_message(exc: BaseException | None) -> str | None:
    """Describe correctable tool requests without masking storage/runtime failures."""

    if isinstance(exc, ArtifactInputError):
        return f"Artifact request rejected: {exc}. Correct the arguments and retry."
    if isinstance(exc, (ArtifactNotFoundError, ArtifactExpiredError, ArtifactReleasedError)):
        return f"Artifact reference unavailable: {exc}. Use an active artifact ref_id."
    if isinstance(exc, ArtifactCodeError):
        return (
            f"{exc}. Correct the artifact Python using the documented helper signatures and retry."
        )
    return None
