from __future__ import annotations

import functools
import time
from typing import Any, Callable

from typing_extensions import ParamSpec

from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter

P = ParamSpec("P")


def llm_retry(
    tries: int = 3, delay_s: float = 0.0
) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    """Decorator to retry flaky LLM assertions a few times.

    Retries the wrapped test function up to `tries` times when an AssertionError is raised.
    Optionally sleeps `delay_s` seconds between attempts.
    """

    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            for attempt in range(tries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == tries - 1:
                        raise
                    if delay_s > 0:
                        time.sleep(delay_s)
            raise AssertionError("unreachable")

        return wrapper

    return decorator


def run_without_llm(src: str) -> Any:
    runtime = compile_program(parse_program(src), adapter=MockAdapter())
    return runtime.run_main()
