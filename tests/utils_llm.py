from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar, Any

from marker_pattern.lang import parse_program, compile_program
import marker_pattern.core as core


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
    original_agent = core.Agent

    class _Agent(original_agent):  # type: ignore[misc]
        def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:  # type: ignore[no-untyped-def]
            super().__init__(model=model, output_type=output_type, **_)

        def run_sync(self, prompt: str):  # type: ignore[override]
            return type("R", (), {"output": {}})()

    try:
        core.Agent = _Agent  # type: ignore[assignment]
        runtime = compile_program(parse_program(src))
        return runtime.run_main()
    finally:
        core.Agent = original_agent  # type: ignore[assignment]

