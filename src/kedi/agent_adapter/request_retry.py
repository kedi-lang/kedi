"""Bounded retries of failed provider requests, never whole agent runs or tools."""

from __future__ import annotations

from asyncio import sleep
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Any

DEFAULT_REQUEST_RETRIES = 2
MAX_RETRY_DELAY = 30.0


def request_retry_limit(settings: Mapping[str, Any], default: int = DEFAULT_REQUEST_RETRIES) -> int:
    value = settings.get("request_retries", default)
    if type(value) is not int or value < 0:
        raise ValueError("request_retries must be a non-negative integer")
    return value


def provider_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    return status if type(status) is int else None


def transient_provider_error(error: Exception) -> bool:
    status = provider_status(error)
    return status is not None and (status in (408, 429) or 500 <= status < 600)


def retry_delay(error: Exception, attempt: int) -> float:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        raw = headers.get("retry-after") if headers is not None else None
        if isinstance(raw, str):
            try:
                delay = float(raw)
            except ValueError:
                try:
                    date = parsedate_to_datetime(raw)
                    delay = (date - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    break
            if isfinite(delay):
                return max(0.0, delay)
            break
        current = current.__cause__
    return min(MAX_RETRY_DELAY, 0.5 * 2 ** min(attempt - 1, 6))


async def wait_for_request_retry(error: Exception, *, attempt: int, retries: int) -> bool:
    if attempt > retries or not transient_provider_error(error):
        return False
    delay = retry_delay(error, attempt)
    # Do not violate a long Retry-After by retrying early, or wait indefinitely.
    if delay > MAX_RETRY_DELAY:
        return False
    await sleep(delay)
    return True
