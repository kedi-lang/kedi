"""Type-annotation resolution & validation.

A written-but-unresolvable annotation (e.g. ``MissingType``) must raise rather
than silently degrade to ``str`` / skip validation; the distinction lives at the
resolution boundary (``KediType.resolve_required``).
"""

from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.ast import TypeApply, TypeRef, TypeUnion
from kedi.lang.compiler.environment import KediRuntime
from kedi.lang.compiler.type_system import UNSUPPORTED_KEDI_TYPE_NAMES, KediType
from kedi.resolved_types import runtime_isinstance
from tests.utils_llm import run_without_llm


def test_unknown_assignment_type_raises() -> None:
    src = "[x: MissingType] = `123`\n= <`x`>\n"
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        run_without_llm(src)


def test_unknown_output_type_raises() -> None:
    src = "@p():\n  >> Produce [x: MissingType]\n  = <x>\n\n= <p()>\n"
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        run_without_llm(src)


def test_unknown_return_type_raises() -> None:
    src = "@p() -> MissingType:\n  = `object()`\n\n= <p()>\n"
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        run_without_llm(src)


def test_no_annotation_still_defaults_to_str() -> None:
    assert run_without_llm('[x] = `"hi"`\n= <x>\n') == "hi"


def test_call_assignment_validates_lhs() -> None:
    src = "@returns_text() -> str:\n  = hello\n\n[x: int] = <returns_text()>\n= <`x`>\n"
    with pytest.raises(Exception, match="valid integer|does not match"):
        run_without_llm(src)


def test_call_assignment_passes_matching_type() -> None:
    src = "@returns_int() -> int:\n  = `7`\n\n[x: int] = <returns_int()>\n= <`str(x + 1)`>\n"
    assert run_without_llm(src) == "8"


def test_native_int_into_untyped_param_raises() -> None:
    src = "@show(x):\n  = <`type(x).__name__`>\n\n= <show(`1`)>\n"
    with pytest.raises(Exception, match="Parameter 'x' does not match annotated type"):
        run_without_llm(src)


def test_str_into_untyped_param_passes() -> None:
    src = "@show(x):\n  = <`type(x).__name__`>\n\n= <show(hello)>\n"
    assert run_without_llm(src) == "str"


def test_structured_return_type_preserves_native_value() -> None:
    # A typed return is validated and kept native (not stringified), so arithmetic works.
    src = "@get_two() -> int:\n  = `2`\n\n= <`str(get_two() + 3)`>\n"
    assert run_without_llm(src) == "5"


def test_runtime_isinstance_validates_generic_arguments() -> None:
    assert runtime_isinstance("x", str) is True
    assert runtime_isinstance(1, str) is False
    assert runtime_isinstance(["x"], list[str]) is True
    assert runtime_isinstance(["x"], list[int]) is False
    assert runtime_isinstance([1, 2, 3], list[int]) is True
    assert runtime_isinstance("x", list[str]) is False
    assert runtime_isinstance(123, list[int]) is False


def test_resolve_required_distinguishes_none_from_unresolvable() -> None:
    rt = KediRuntime()
    assert KediType(None).resolve_required(rt, {}) is None
    assert KediType(TypeRef(name="None")).resolve_required(rt, {}) is type(None)
    assert KediType(TypeRef(name="int")).resolve_required(rt, {}) is int
    with pytest.raises(TypeError, match="Unknown type annotation MissingType"):
        KediType(TypeRef(name="MissingType")).resolve_required(rt, {})


@pytest.mark.parametrize("name", sorted(UNSUPPORTED_KEDI_TYPE_NAMES))
def test_unsupported_kedi_type_names_are_rejected_by_type_system(name: str) -> None:
    rt = KediRuntime()

    assert KediType(TypeRef(name=name)).resolve(rt, {}) is None
    assert KediType(name).resolve(rt, {}) is None
    with pytest.raises(TypeError, match=rf"Unsupported type annotation {name}"):
        KediType(TypeRef(name=name)).resolve_required(rt, {})
    with pytest.raises(TypeError, match=rf"Unsupported type annotation {name}"):
        KediType(name).resolve_required(rt, {})


def test_unsupported_nested_kedi_type_is_rejected() -> None:
    annotation = TypeApply(
        name="list",
        args=[TypeApply(name="tuple", args=[TypeRef(name="int")])],
    )

    rt = KediRuntime()
    assert KediType(annotation).resolve(rt, {}) is None
    with pytest.raises(TypeError, match="Unsupported type annotation tuple"):
        KediType(annotation).resolve_required(rt, {})


def test_unsupported_union_member_is_rejected() -> None:
    cases = (
        (TypeUnion(left=None, right=TypeRef(name="range")), "range"),
        (TypeUnion(left=TypeRef(name="object"), right=None), "object"),
    )

    for annotation, name in cases:
        with pytest.raises(TypeError, match=f"Unsupported type annotation {name}"):
            KediType(annotation).resolve_required(KediRuntime(), {})


def test_malformed_legacy_type_remains_unknown() -> None:
    with pytest.raises(TypeError, match=r"Unknown type annotation \["):
        KediType("[").resolve_required(KediRuntime(), {})


def test_get_adapter_stays_lenient_for_introspection() -> None:
    rt = KediRuntime()
    assert KediType(TypeRef(name="MissingType")).get_adapter(rt, {}) is None
    assert KediType(None).get_adapter(rt, {}) is None


def test_optimizer_resolvers_fail_loud_on_unknown_type() -> None:
    from kedi.optimization.services import TypeResolver

    rt = compile_program(parse_program("= ok\n"), adapter=None)
    resolver = TypeResolver(rt)

    proc = parse_program("@p(a: int, b) -> str:\n  = ok\n").procedures[0]
    assert resolver.resolve_param_types(proc) == {"a": int}
    assert resolver.resolve_output_types([("x", None), ("y", "int")]) == {
        "x": str,
        "y": int,
    }

    bad = parse_program("@p(a: MissingType) -> str:\n  = ok\n").procedures[0]
    with pytest.raises(TypeError, match="Unknown type annotation MissingType"):
        resolver.resolve_param_types(bad)
    with pytest.raises(TypeError, match="Unknown type annotation MissingType"):
        resolver.resolve_output_types([("x", "MissingType")])
