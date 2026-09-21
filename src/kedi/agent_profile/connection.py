from __future__ import annotations

import base64
import inspect
import ipaddress
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlsplit, urlunsplit


A2AAuthScheme: TypeAlias = Literal["basic", "bearer", "api_key"]
CredentialProvider: TypeAlias = Callable[[], str] | Callable[["A2AAuthConfig"], str]

DEFAULT_A2A_REQUEST_TIMEOUT = 30.0
DEFAULT_A2A_TASK_TIMEOUT = 600.0
DEFAULT_A2A_POLL_INTERVAL = 0.5
MAX_A2A_SECRET_FILE_BYTES = 64 * 1024


class A2AConnectionError(ValueError):
    """Raised when an A2A connection or credential reference is invalid."""


class A2ACredentialError(RuntimeError):
    """Raised when an A2A credential cannot be resolved safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class A2AAuthConfig:
    """Secret-safe client authentication configuration for one A2A endpoint."""

    scheme: A2AAuthScheme
    username: str | None = None
    secret_env: str | None = None
    secret_file: str | None = None
    provider: CredentialProvider | None = None
    header: str | None = None

    def __post_init__(self) -> None:
        if self.scheme not in {"basic", "bearer", "api_key"}:
            raise A2AConnectionError("A2A auth scheme must be one of: basic, bearer, api_key")
        sources = tuple(
            source
            for source in (self.secret_env, self.secret_file, self.provider)
            if source is not None
        )
        if len(sources) != 1:
            raise A2AConnectionError(
                "A2A auth must define exactly one credential source: env, file, or provider"
            )
        if self.scheme == "basic":
            if not isinstance(self.username, str) or not self.username:
                raise A2AConnectionError("A2A basic auth requires a non-empty username")
            if ":" in self.username:
                raise A2AConnectionError("A2A basic auth username cannot contain ':'")
        elif self.username is not None:
            raise A2AConnectionError(f"A2A {self.scheme} auth does not accept a username")
        if self.scheme == "api_key":
            header = self.header or "X-API-Key"
            if not _valid_header_name(header):
                raise A2AConnectionError("A2A API-key header is not a valid HTTP header name")
            object.__setattr__(self, "header", header)
        elif self.header is not None:
            raise A2AConnectionError(f"A2A {self.scheme} auth does not accept a custom header")
        if self.secret_env is not None and not _valid_env_name(self.secret_env):
            raise A2AConnectionError("A2A credential environment variable name is invalid")
        if self.secret_file is not None and not self.secret_file.strip():
            raise A2AConnectionError("A2A credential file path cannot be empty")
        if self.provider is not None and not callable(self.provider):
            raise A2AConnectionError("A2A credential provider must be callable")

    def resolve_secret(self, *, env: Mapping[str, str] | None = None) -> str:
        if self.secret_env is not None:
            source = os.environ if env is None else env
            value = source.get(self.secret_env)
            if value is None:
                raise A2ACredentialError(
                    f"A2A credential environment variable {self.secret_env!r} is not set"
                )
            return _require_secret(value)
        if self.secret_file is not None:
            path = Path(self.secret_file).expanduser()
            try:
                if not path.is_file():
                    raise A2ACredentialError(
                        f"A2A credential file {str(path)!r} is not a regular file"
                    )
                if path.stat().st_size > MAX_A2A_SECRET_FILE_BYTES:
                    raise A2ACredentialError(
                        f"A2A credential file exceeds {MAX_A2A_SECRET_FILE_BYTES} bytes"
                    )
                value = path.read_text(encoding="utf-8")
            except A2ACredentialError:
                raise
            except (OSError, UnicodeError) as exc:
                raise A2ACredentialError(
                    f"Could not read A2A credential file {str(path)!r}"
                ) from exc
            return _require_secret(value.rstrip("\r\n"))
        assert self.provider is not None
        provider = self.provider
        try:
            signature = inspect.signature(provider)
            if len(signature.parameters) == 0:
                value = cast(Callable[[], str], provider)()
            else:
                value = cast(Callable[[A2AAuthConfig], str], provider)(self)
        except A2ACredentialError:
            raise
        except Exception as exc:
            raise A2ACredentialError("A2A credential provider failed") from exc
        if not isinstance(value, str):
            raise A2ACredentialError("A2A credential provider must return str")
        return _require_secret(value)

    def headers(self, *, env: Mapping[str, str] | None = None) -> dict[str, str]:
        secret = self.resolve_secret(env=env)
        if self.scheme == "basic":
            raw = f"{self.username}:{secret}".encode()
            return {"Authorization": f"Basic {base64.b64encode(raw).decode('ascii')}"}
        if self.scheme == "bearer":
            return {"Authorization": f"Bearer {secret}"}
        assert self.header is not None
        return {self.header: secret}

    def identity(self) -> tuple[str, str | None, str | None, str | None, str | None, str | None]:
        provider_name = None
        if self.provider is not None:
            provider_name = (
                f"{getattr(self.provider, '__module__', '')}."
                f"{getattr(self.provider, '__qualname__', type(self.provider).__qualname__)}"
            )
        return (
            self.scheme,
            self.username,
            self.secret_env,
            self.secret_file,
            provider_name,
            self.header,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class A2AConnection:
    """Validated endpoint and transport policy for a remote A2A agent."""

    endpoint: str
    auth: A2AAuthConfig | None = None
    request_timeout: float = DEFAULT_A2A_REQUEST_TIMEOUT
    task_timeout: float = DEFAULT_A2A_TASK_TIMEOUT
    poll_interval: float = DEFAULT_A2A_POLL_INTERVAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", normalize_a2a_endpoint(self.endpoint))
        for name in ("request_timeout", "task_timeout", "poll_interval"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise A2AConnectionError(f"A2A {name} must be a finite positive number")
            object.__setattr__(self, name, float(value))

    def identity(self) -> tuple[Any, ...]:
        return (
            self.endpoint,
            None if self.auth is None else self.auth.identity(),
            self.request_timeout,
            self.task_timeout,
            self.poll_interval,
        )


def normalize_a2a_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A2AConnectionError("A2A endpoint must be a non-empty string")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"}:
        raise A2AConnectionError("A2A endpoint must use http or https")
    if not parsed.hostname:
        raise A2AConnectionError("A2A endpoint must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise A2AConnectionError("A2A endpoint cannot embed credentials")
    if parsed.query or parsed.fragment:
        raise A2AConnectionError("A2A endpoint cannot include a query or fragment")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise A2AConnectionError("Remote A2A endpoints must use https")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_secret(value: str) -> str:
    if not value:
        raise A2ACredentialError("A2A credential is empty")
    if "\r" in value or "\n" in value:
        raise A2ACredentialError("A2A credential cannot contain newline characters")
    return value


def _valid_env_name(value: str) -> bool:
    return (
        bool(value)
        and (value[0].isalpha() or value[0] == "_")
        and all(character.isalnum() or character == "_" for character in value)
    )


def _valid_header_name(value: str) -> bool:
    token = "!#$%&'*+-.^_`|~"
    return (
        bool(value)
        and value.isascii()
        and all(character.isalnum() or character in token for character in value)
    )
