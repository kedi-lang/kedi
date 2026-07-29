from __future__ import annotations


class ArtifactError(RuntimeError):
    """Base error for artifact lifecycle and storage operations."""


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


class ArtifactSerializationError(ArtifactError, TypeError):
    """Raised when a value cannot be safely represented by a store."""


class ArtifactAccessError(ArtifactError, PermissionError):
    """Raised when a reference is used outside its owning artifact session."""


class ArtifactCapabilityError(ArtifactError):
    """Raised when the active adapter cannot expose artifact operations."""
