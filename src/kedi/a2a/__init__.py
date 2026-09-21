"""A2A protocol constants shared by Kedi clients and servers."""

from .protocol import (
    KEDI_STRUCTURED_OUTPUT_EXTENSION_URI,
    KEDI_STRUCTURED_OUTPUT_METADATA_KEY,
    A2AAmbiguousSendError,
    A2AError,
    A2ARemoteResult,
    A2ARemoteTaskError,
    A2ARemoteTaskSnapshot,
    A2ATaskTimeoutError,
    A2ATransportError,
)

__all__ = [
    "A2AAmbiguousSendError",
    "A2AError",
    "A2ARemoteResult",
    "A2ARemoteTaskSnapshot",
    "A2ARemoteTaskError",
    "A2ATaskTimeoutError",
    "A2ATransportError",
    "KEDI_STRUCTURED_OUTPUT_EXTENSION_URI",
    "KEDI_STRUCTURED_OUTPUT_METADATA_KEY",
]
