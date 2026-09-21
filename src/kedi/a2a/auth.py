from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Literal

A2AServerAuthScheme = Literal["basic", "bearer", "api_key"]


@dataclass(frozen=True, slots=True, kw_only=True)
class A2AServerAuth:
    """Authentication policy for a Kedi-hosted A2A endpoint."""

    scheme: A2AServerAuthScheme
    secret: str = field(repr=False)
    username: str | None = None
    header: str = "X-API-Key"

    def __post_init__(self) -> None:
        if self.scheme not in {"basic", "bearer", "api_key"}:
            raise ValueError("A2A server auth must be basic, bearer, or api_key")
        if not self.secret or "\r" in self.secret or "\n" in self.secret:
            raise ValueError("A2A server credential must be non-empty and single-line")
        if self.scheme == "basic":
            if not self.username or ":" in self.username:
                raise ValueError("A2A basic auth requires a username without ':'")
        elif self.username is not None:
            raise ValueError(f"A2A {self.scheme} auth does not accept a username")
        if self.scheme == "api_key":
            if not _valid_header_name(self.header):
                raise ValueError("A2A API-key header is invalid")
        elif self.header != "X-API-Key":
            raise ValueError(f"A2A {self.scheme} auth does not accept an API-key header")

    @property
    def principal(self) -> str:
        if self.scheme == "basic":
            assert self.username is not None
            return self.username
        fingerprint = hashlib.sha256(self.secret.encode()).hexdigest()[:16]
        return f"{self.scheme}:{fingerprint}"

    def authenticate(self, headers: object) -> str | None:
        get = getattr(headers, "get")
        if self.scheme == "basic":
            value = get("authorization")
            credentials = _http_credentials(value, expected_scheme="basic")
            if credentials is None:
                return None
            try:
                decoded = base64.b64decode(credentials, validate=True).decode("utf-8")
                username, secret = decoded.split(":", 1)
            except (ValueError, UnicodeError):
                return None
            assert self.username is not None
            if _credentials_match(username, self.username) and _credentials_match(
                secret,
                self.secret,
            ):
                return self.principal
            return None
        if self.scheme == "bearer":
            value = get("authorization")
            credentials = _http_credentials(value, expected_scheme="bearer")
            if credentials is None:
                return None
            return self.principal if _credentials_match(credentials, self.secret) else None
        value = get(self.header)
        return (
            self.principal
            if isinstance(value, str) and _credentials_match(value, self.secret)
            else None
        )


def _credentials_match(value: str, expected: str) -> bool:
    return hmac.compare_digest(value.encode("utf-8"), expected.encode("utf-8"))


def _http_credentials(value: object, *, expected_scheme: str) -> str | None:
    if not isinstance(value, str):
        return None
    parts = value.split(None, 1)
    if len(parts) != 2 or parts[0].casefold() != expected_scheme:
        return None
    return parts[1]


def _valid_header_name(value: str) -> bool:
    token = "!#$%&'*+-.^_`|~"
    return (
        bool(value)
        and value.isascii()
        and all(character.isalnum() or character in token for character in value)
    )


__all__ = ["A2AServerAuth", "A2AServerAuthScheme"]
