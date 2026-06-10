from __future__ import annotations

import json
from typing import Final

from cassetter import RawRequest, RawResponse

_FILTERED: Final[str] = "[FILTERED]"

# Headers stripped on write via cassetter SecurityConfig.filter_headers.
CASSETTE_FILTER_HEADERS: list[str] = [
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-auth-token",
    "proxy-authorization",
    "www-authenticate",
    "x-request-id",
    "cf-ray",
    "via",
    "date",
    "x-groq-region",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "x-stainless-os",
    "x-stainless-arch",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "x-stainless-lang",
    "x-stainless-package-version",
    "x-stainless-async",
    "x-stainless-read-timeout",
    "x-stainless-retry-count",
]

_REDACT_HEADER_NAMES: Final[frozenset[str]] = frozenset(h.lower() for h in CASSETTE_FILTER_HEADERS)

_REDACT_JSON_KEYS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "seed",
        "created",
        "system_fingerprint",
        "x_groq",
    }
)

_CANONICAL_RECORDED_AT: Final[str] = "2000-01-01T00:00:00+00:00"


def _drop_headers(headers: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        name: values for name, values in headers.items() if name.lower() not in _REDACT_HEADER_NAMES
    }


def _scrub_response_body(headers: dict[str, list[str]], body: bytes | None) -> bytes | None:
    if body is None:
        return None
    content_type = next(
        (
            value
            for name, values in headers.items()
            if name.lower() == "content-type"
            for value in values
        ),
        "",
    )
    if "application/json" not in content_type.lower():
        return body
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if not isinstance(payload, dict):
        return body
    scrubbed = {key: value for key, value in payload.items() if key not in _REDACT_JSON_KEYS}
    return json.dumps(scrubbed, separators=(",", ":")).encode()


def redact_record_request(request: RawRequest) -> RawRequest:
    return RawRequest(
        method=request.method,
        uri=request.uri,
        headers=_drop_headers(request.headers),
        body=request.body,
    )


def redact_record_response(response: RawResponse) -> RawResponse:
    headers = _drop_headers(response.headers)
    body = _scrub_response_body(headers, response.body)
    return RawResponse(status=response.status, headers=headers, body=body)


def cassette_security_config() -> dict[str, object]:
    return {
        "filter_headers": CASSETTE_FILTER_HEADERS,
        "filter_replacement": _FILTERED,
    }


def normalize_cassette_file(path: str) -> None:
    """Normalize volatile cassette envelope fields after recording."""
    import re
    from pathlib import Path

    cassette_path = Path(path)
    if not cassette_path.is_file():
        return
    text = cassette_path.read_text(encoding="utf-8")
    normalized = re.sub(
        r"^  recorded_at:.*$",
        f"  recorded_at: {_CANONICAL_RECORDED_AT}",
        text,
        flags=re.MULTILINE,
    )
    cassette_path.write_text(normalized, encoding="utf-8")
