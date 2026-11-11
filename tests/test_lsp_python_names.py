"""Which names in embedded Python refer to a Kedi binding."""

from __future__ import annotations

import pytest

from kedi.lang.parser.parser import parse as ts_parse
from kedi.lsp.features.python_names import binding_uses, python_regions


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("n > 1", [(0, 0)]),
        ("obj.n + f(n=1)", []),
        ("y = [n for n in range(3)]\nn = n * 2", [(1, 0), (1, 4)]),
        ("[x for x in n if x]", [(0, 12)]),
        ("{n: v for v in range(n)}", [(0, 1), (0, 21)]),
        ("[a + b for a in n for b in n]", [(0, 16), (0, 27)]),
        ("f = lambda n: n + 1\ng = lambda: n", [(1, 12)]),
        ("def f(x=n, *rest, k=n, **opts) -> n:\n    return n", [(0, 8), (0, 20), (0, 34), (1, 11)]),
        ("@n\ndef f(x: n):\n    n = 1\n    return n", [(0, 1), (1, 9)]),
        ("def f():\n    global n\n    n = n + 1", [(1, 11), (2, 4), (2, 8)]),
        ("def f():\n    import n as m\n    return n", [(2, 11)]),
        ("def f():\n    from os import path as n\n    return n", []),
        ("def f():\n    try:\n        pass\n    except Exception as n:\n        return n", []),
        ("def outer():\n    n = 1\n    def inner():\n        nonlocal n\n        n = 2", []),
        ("def f():\n    def n():\n        pass\n    return n", []),
        (
            "class C(n):\n    n = 1\n    m = n\n    def get(self):\n        return n",
            [(0, 8), (4, 15)],
        ),
        ("class C:\n    x = n", [(1, 8)]),
        ("async def f():\n    return n", [(1, 11)]),
        ("def (", []),
        ("def f():\n    nonlocal n\n    return n", []),
    ],
)
def test_binding_uses(code: str, expected: list[tuple[int, int]]) -> None:
    assert binding_uses(code, "n") == expected


def test_python_regions_map_dedented_lines_to_source_bytes() -> None:
    source = "@f():\n  ```\n  x = 1\n\n    y = 2\n  ```\n  = ok\n= `n`\n".encode()

    regions = list(python_regions(ts_parse(source).root_node, source))

    fenced = next(code for code, _ in regions if "x = 1" in code)
    offsets = next(offsets for code, offsets in regions if code is fenced)
    first_line = fenced.split("\n").index("x = 1")
    assert source[offsets[first_line] : offsets[first_line] + 5] == b"x = 1"
    assert ("n", [source.index(b"`n`") + 1]) in regions
