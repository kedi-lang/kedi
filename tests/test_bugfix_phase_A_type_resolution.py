"""Phase A bugfix tests: type resolution & validation.

Covers the validated findings in ``mlruns/bug-findings.md`` whose shared root
cause is that a *written-but-unresolvable* type annotation (e.g. ``MissingType``)
was treated identically to *no annotation* — both silently became ``str`` or
skipped validation. The distinction now lives at the resolution boundary
(``KediType.resolve_required``).

Findings: BUG-001, 002, 003, 004, 005, 016, 022, 023.
"""

from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import TypeRef
from kedi.lang.compiler.environment import KediRuntime
from kedi.lang.compiler.type_system import KediType
from kedi.resolved_types import runtime_isinstance
from tests.utils_llm import run_without_llm


# ---------------------------------------------------------------------------
# BUG-001/002/004 — a written-but-unresolvable annotation raises (not str)
# ---------------------------------------------------------------------------


def test_bug001_unknown_assignment_type_raises() -> None:
    """`[x: MissingType] = `123`` must raise, not silently accept the value."""
    src = "[x: MissingType] = `123`\n= <`x`>\n"
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        run_without_llm(src)


def test_bug002_unknown_output_type_raises() -> None:
    """An unresolvable output annotation must raise, not become a str schema."""
    src = "@p():\n  >> Produce [x: MissingType]\n  = <x>\n\n= <p()>\n"
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        run_without_llm(src)


def test_bug004_unknown_return_type_raises() -> None:
    """An unresolvable return annotation must raise, not accept a native object."""
    src = "@p() -> MissingType:\n  = `object()`\n\n= <p()>\n"
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        run_without_llm(src)


def test_no_annotation_still_defaults_to_str() -> None:
    """No annotation (vs. an unresolvable one) keeps the lenient str default."""
    assert run_without_llm('[x] = `"hi"`\n= <x>\n') == "hi"


# ---------------------------------------------------------------------------
# BUG-003 — procedure-call assignment validates the LHS annotation
# ---------------------------------------------------------------------------


def test_bug003_call_assignment_validates_lhs() -> None:
    """`[x: int] = <returns_text()>` must reject the str procedure result."""
    src = "@returns_text() -> str:\n  = hello\n\n[x: int] = <returns_text()>\n= <`x`>\n"
    with pytest.raises(Exception, match="valid integer|does not match"):
        run_without_llm(src)


def test_bug003_call_assignment_passes_matching_type() -> None:
    """A call result that matches the LHS annotation is accepted."""
    src = "@returns_int() -> int:\n  = `7`\n\n[x: int] = <returns_int()>\n= <`str(x + 1)`>\n"
    assert run_without_llm(src) == "8"


# ---------------------------------------------------------------------------
# BUG-005 — untyped param is `str`: str passes, native non-str raises
# ---------------------------------------------------------------------------


def test_bug005_native_int_into_untyped_param_raises() -> None:
    """A native int passed to an untyped (=> str) param raises — no coercion."""
    src = "@show(x):\n  = <`type(x).__name__`>\n\n= <show(`1`)>\n"
    with pytest.raises(Exception, match="Parameter 'x' does not match annotated type"):
        run_without_llm(src)


def test_bug005_str_into_untyped_param_passes() -> None:
    """A str argument to an untyped param passes through unchanged."""
    src = "@show(x):\n  = <`type(x).__name__`>\n\n= <show(hello)>\n"
    assert run_without_llm(src) == "str"


# ---------------------------------------------------------------------------
# BUG-016 — structured-only typed returns take the enforced native path
# ---------------------------------------------------------------------------


def test_bug016_structured_return_type_enforced() -> None:
    """A typed (structured ``return_type_expr``) return is validated natively."""
    src = "@get_two() -> int:\n  = `2`\n\n= <`str(get_two() + 3)`>\n"
    # The native int is preserved (not stringified) so arithmetic works.
    assert run_without_llm(src) == "5"


# ---------------------------------------------------------------------------
# BUG-022 — runtime_isinstance validates generic arguments, fail-closed
# ---------------------------------------------------------------------------


def test_bug022_runtime_isinstance_generics() -> None:
    assert runtime_isinstance("x", str) is True
    assert runtime_isinstance(1, str) is False
    # Generic args are validated, not just the outer container shape.
    assert runtime_isinstance(["x"], list[str]) is True
    assert runtime_isinstance(["x"], list[int]) is False
    assert runtime_isinstance([1, 2, 3], list[int]) is True
    # A bare str is not a list[str] (the previously buggy `return True`).
    assert runtime_isinstance("x", list[str]) is False
    # Unsupported / unmatched values fail closed.
    assert runtime_isinstance(123, list[int]) is False


# ---------------------------------------------------------------------------
# KediType.resolve_required — the centralized rule
# ---------------------------------------------------------------------------


def test_resolve_required_distinguishes_none_from_unresolvable() -> None:
    rt = KediRuntime()
    # No annotation -> None (lenient downstream).
    assert KediType(None).resolve_required(rt, {}) is None
    # Resolvable annotation -> the resolved type.
    assert KediType(TypeRef(name="int")).resolve_required(rt, {}) is int
    # Written-but-unresolvable -> raise.
    with pytest.raises(TypeError, match="Unknown type annotation MissingType"):
        KediType(TypeRef(name="MissingType")).resolve_required(rt, {})


def test_get_adapter_stays_lenient_for_introspection() -> None:
    """get_adapter (introspection-only) must not raise on a typo."""
    rt = KediRuntime()
    assert KediType(TypeRef(name="MissingType")).get_adapter(rt, {}) is None
    assert KediType(None).get_adapter(rt, {}) is None


# ---------------------------------------------------------------------------
# BUG-023 — optimizer resolvers fail loud on unresolvable annotations
# ---------------------------------------------------------------------------


def test_bug023_optimizer_resolvers_fail_loud() -> None:
    from kedi.optimization.services import TypeResolver

    rt = compile_program(parse_program("= ok\n"), adapter=None)
    resolver = TypeResolver(rt)

    # No annotation -> omit (params) / str (outputs); resolvable -> resolved.
    proc = parse_program("@p(a: int, b) -> str:\n  = ok\n").procedures[0]
    assert resolver.resolve_param_types(proc) == {"a": int}
    assert resolver.resolve_output_types([("x", None), ("y", "int")]) == {"x": str, "y": int}

    # Written-but-unresolvable -> raise.
    bad = parse_program("@p(a: MissingType) -> str:\n  = ok\n").procedures[0]
    with pytest.raises(TypeError, match="Unknown type annotation MissingType"):
        resolver.resolve_param_types(bad)
    with pytest.raises(TypeError, match="Unknown type annotation MissingType"):
        resolver.resolve_output_types([("x", "MissingType")])
