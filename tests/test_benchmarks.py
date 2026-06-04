"""Performance benchmarks for the Kedi parser and compiler pipeline.

These benchmarks measure the core CPU-bound operations in the Kedi language
toolchain: parsing source text into ASTs, compiling programs into runtimes,
and the combined pipeline. They use MockAdapter to avoid LLM calls.
"""

from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter

# ---------------------------------------------------------------------------
# Source fixtures -- varying complexity levels
# ---------------------------------------------------------------------------

SIMPLE_SOURCE = "= Paris\n"

MEDIUM_SOURCE = (
    "@get_country(city):\n"
    "  <city> is located in [country].\n"
    "  = <country>\n\n"
    "@get_language(country):\n"
    "  The primary language spoken in <country> is [language].\n"
    "  = <language>\n\n"
    "= <get_language(<get_country(Paris)>)>\n"
)

COMPLEX_SOURCE = (
    "~City(name, history, economy, cuisine)\n\n"
    "@get_city_info(city) -> str:\n"
    "  Paragraphs [history], [economy], and [cuisine] describe <city> in different aspects.\n"
    "  = <history>\n\n"
    "@get_cities_in(country) -> list[str]:\n"
    "  All cities in <country> are [cities: list[str]].\n"
    "  = `cities`\n\n"
    "@concat(a, b):\n"
    "  = <`str(a) + str(b)`>\n\n"
    "@times2(n: int):\n"
    "  = <`int(n) * 2`>\n\n"
    "@pipeline(x):\n"
    "  from <x> we ask for [y]\n"
    "  [z: str] = ```\n"
    "  return 'prefix-' + y\n"
    "  ```\n"
    "  final [out] uses <`z`>\n"
    "  = <out>\n\n"
    "[country] = Singapore\n"
    "= <pipeline(<concat(hello, world)>)>\n"
)

TYPED_SOURCE = (
    "@add(x: int, y: int) -> int:\n"
    "  = `x + y`\n\n"
    "@process(items: list[int]) -> str:\n"
    "  [total: int] = `sum(items)`\n"
    "  [avg: float] = `sum(items) / len(items) if items else 0.0`\n"
    "  = <`f'{total} (avg {avg:.1f})'`>\n\n"
    "@multi_output():\n"
    "  need [a: int], [b: float], and [c: bool]\n"
    "  = <a>-<`int(b)`>-<`'T' if c else 'F'`>\n\n"
    "= <process(`[1, 2, 3, 4, 5]`)>\n"
)


# ---------------------------------------------------------------------------
# Parse benchmarks
# ---------------------------------------------------------------------------


def test_bench_parse_simple(benchmark):
    """Benchmark parsing a minimal Kedi program."""
    benchmark(parse_program, SIMPLE_SOURCE)


def test_bench_parse_medium(benchmark):
    """Benchmark parsing a program with nested procedure calls."""
    benchmark(parse_program, MEDIUM_SOURCE)


def test_bench_parse_complex(benchmark):
    """Benchmark parsing a program with custom types, python blocks, and multiple procedures."""
    benchmark(parse_program, COMPLEX_SOURCE)


def test_bench_parse_typed(benchmark):
    """Benchmark parsing a program with rich type annotations."""
    benchmark(parse_program, TYPED_SOURCE)


# ---------------------------------------------------------------------------
# Compile benchmarks (parse + compile)
# ---------------------------------------------------------------------------


def test_bench_compile_simple(benchmark):
    """Benchmark compiling a minimal Kedi program."""
    prog = parse_program(SIMPLE_SOURCE)
    adapter = MockAdapter()
    benchmark(compile_program, prog, adapter=adapter)


def test_bench_compile_medium(benchmark):
    """Benchmark compiling a program with nested procedure calls."""
    prog = parse_program(MEDIUM_SOURCE)
    adapter = MockAdapter()
    benchmark(compile_program, prog, adapter=adapter)


def test_bench_compile_complex(benchmark):
    """Benchmark compiling a program with custom types, python blocks, and multiple procedures."""
    prog = parse_program(COMPLEX_SOURCE)
    adapter = MockAdapter()
    benchmark(compile_program, prog, adapter=adapter)


def test_bench_compile_typed(benchmark):
    """Benchmark compiling a program with rich type annotations."""
    prog = parse_program(TYPED_SOURCE)
    adapter = MockAdapter()
    benchmark(compile_program, prog, adapter=adapter)


# ---------------------------------------------------------------------------
# Full pipeline benchmarks (parse + compile + run)
# ---------------------------------------------------------------------------


def test_bench_pipeline_simple(benchmark):
    """Benchmark the full pipeline for a minimal program."""
    adapter = MockAdapter()

    def pipeline():
        prog = parse_program(SIMPLE_SOURCE)
        rt = compile_program(prog, adapter=adapter)
        return rt.run_main()

    benchmark(pipeline)


def test_bench_pipeline_medium(benchmark):
    """Benchmark the full pipeline for a program with nested procedure calls."""
    adapter = MockAdapter()

    def pipeline():
        prog = parse_program(MEDIUM_SOURCE)
        rt = compile_program(prog, adapter=adapter)
        return rt.run_main()

    benchmark(pipeline)


def test_bench_pipeline_typed(benchmark):
    """Benchmark the full pipeline for a program with type annotations."""
    adapter = MockAdapter()

    def pipeline():
        prog = parse_program(TYPED_SOURCE)
        rt = compile_program(prog, adapter=adapter)
        return rt.run_main()

    benchmark(pipeline)
