from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter

F = TypeVar("F", bound=Callable[..., Any])


def llm_retry(tries: int = 3, delay_s: float = 0.0) -> Callable[[F], F]:
    """Decorator to retry flaky LLM assertions a few times.

    Retries the wrapped test function up to `tries` times when an AssertionError is raised.
    Optionally sleeps `delay_s` seconds between attempts.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):  # type: ignore[override]
            last_exc: Exception | None = None
            for attempt in range(tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt == tries - 1:
                        raise
                    if delay_s > 0:
                        time.sleep(delay_s)

        return wrapper  # type: ignore[return-value]

    return decorator


def run_without_llm(src: str) -> Any:
    runtime = compile_program(parse_program(src), adapter=MockAdapter())
    return runtime.run_main()
