from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from cassetter._types import CassetteConfig
from cassetter.cassette import Cassette

from tests.cassette_redaction import (
    cassette_security_config,
    normalize_cassette_file,
    redact_record_request,
    redact_record_response,
)


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()


@pytest.fixture(scope="module")
def vcr_config() -> CassetteConfig:
    """Record missing cassettes; replay when present. CI passes ``--record-mode=none`` via Makefile."""
    return {
        "record_mode": "once",
        "intercept": ["httpx"],
        "before_record_request": redact_record_request,
        "before_record_response": redact_record_response,
        **cassette_security_config(),
    }


@pytest.fixture(autouse=True)
def _normalize_on_cassette_save(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if request.node.get_closest_marker("vcr") is None:
        return

    original_save = Cassette.save

    def save_and_normalize(self: Cassette) -> None:
        original_save(self)
        normalize_cassette_file(self.path)

    monkeypatch.setattr(Cassette, "save", save_and_normalize)


@pytest.fixture(autouse=True)
def _placeholder_llm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Providers validate credentials when an adapter/client is constructed, before
    any HTTP. Tests drive the actual calls with mocks or cassettes, so supply
    placeholder keys when none are set. Real keys (present when recording cassettes)
    are left untouched.
    """
    for var in ("OPENAI_API_KEY", "GROQ_API_KEY"):
        if not os.environ.get(var):
            monkeypatch.setenv(var, "test-placeholder")
