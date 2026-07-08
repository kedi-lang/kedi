from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from kedi.core import KediRuntime
from kedi.lang.ast import (
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypeLiteral,
    TypePython,
    TypeRef,
    TypeUnion,
)
from kedi.lang.compiler import type_system as type_system_module
from kedi.lang.compiler.type_system import (
    KediType,
    _convert_basic_literal,
    _default_factory,
    _is_immutable_default,
    _lookup_name,
    _parse_model_repr,
    _typeexpr_to_str,
    assign_value_from_native,
    coerce_model_value,
    detect_rhs_kind,
    is_value_compatible,
    pop_type_env,
    register_type,
)


class Person(BaseModel):
    name: str
    age: int = 0
    active: bool = False


class OtherModel(BaseModel):
    value: str


class UnknownTypeExpr(TypeExpr):
    pass


def test_structured_type_resolution_and_rendering_edges() -> None:
    rt = KediRuntime()
    type_env: dict[str, Any] = {"BadValue": object()}

    assert _lookup_name(rt, "Local", {"Local": str}) is str
    assert _lookup_name(rt, "int", {}) is int
    assert _lookup_name(rt, "missing_name_for_type_resolution", {}) is None

    assert KediType(TypeApply(name="Missing", args=[TypeRef(name="int")])).resolve(rt, {}) is None
    assert KediType(TypeApply(name="list", args=[TypeRef(name="Missing")])).resolve(rt, {}) is None
    assert (
        KediType(TypeApply(name="dict", args=[TypeRef(name="str"), TypeRef(name="int")])).resolve(
            rt,
            {},
        )
        == dict[str, int]
    )
    assert (
        KediType(TypeApply(name="BadValue", args=[TypeRef(name="int")])).resolve(rt, type_env)
        is None
    )

    assert KediType(TypeUnion(left=None, right=TypeRef(name="int"))).resolve(rt, {}) is None
    assert KediType(TypeUnion(left=TypeRef(name="str"), right=TypeRef(name="int"))).resolve(
        rt,
        {},
    ) == (str | int)
    assert (
        KediType(TypeUnion(left=TypeRef(name="BadValue"), right=TypeRef(name="int"))).resolve(
            rt,
            type_env,
        )
        is None
    )

    assert KediType(TypePython(code="list[int]")).resolve(rt, {}) == list[int]
    with pytest.raises(TypeError, match="Invalid type expression"):
        KediType(TypePython(code="missing_name_for_type_resolution")).resolve(rt, {})
    with pytest.raises(TypeError, match="does not evaluate to a valid type"):
        KediType(TypePython(code="1 + 1")).resolve(rt, {})
    assert KediType(UnknownTypeExpr()).resolve(rt, {}) is None

    assert repr(KediType(TypeRef(name="str"))).startswith("KediType(TypeRef")
    assert repr(KediType("int")) == "KediType('int')"
    assert str(KediType(TypeApply(name="list", args=[TypeRef(name="str")]))) == "list[str]"
    annotated = KediType(
        TypeApply(
            name="Annotated",
            args=[
                TypeRef(name="str"),
                TypeLiteral(value="City name", source='"City name"'),
            ],
        )
    )
    assert annotated.resolve(rt, {"Annotated": Annotated}) == Annotated[str, "City name"]
    assert str(annotated) == 'Annotated[str, "City name"]'
    assert str(KediType(TypeUnion(left=TypeRef(name="str"), right=TypeRef(name="int")))) == (
        "str | int"
    )
    assert str(KediType(TypePython(code="Person"))) == "`Person`"
    assert _typeexpr_to_str(None) == "None"
    assert _typeexpr_to_str(UnknownTypeExpr()) == "UnknownTypeExpr()"


def test_kedi_type_adapter_and_legacy_resolution_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = KediRuntime()

    assert KediType(None).is_none()
    assert KediType("").is_none()
    assert KediType(None).resolve(rt, {}) is None
    assert KediType.from_node(TypeRef(name="int"), "str").resolve(rt, {}) is int
    assert KediType(None).get_adapter(rt, {}) is None
    # get_adapter stays lenient (introspection-only); validate is strict.
    assert KediType("missing_name_for_type_resolution").get_adapter(rt, {}) is None
    # No annotation -> validate returns the value unchanged.
    assert KediType(None).validate(rt, {}, "value") == "value"
    # Written-but-unresolvable annotation -> validate raises.
    with pytest.raises(TypeError, match="Unknown type annotation"):
        KediType("missing_name_for_type_resolution").validate(rt, {}, "value")
    # resolve_required mirrors that distinction directly.
    assert KediType(None).resolve_required(rt, {}) is None
    with pytest.raises(TypeError, match="Unknown type annotation"):
        KediType("missing_name_for_type_resolution").resolve_required(rt, {})
    assert KediType("`int`").resolve(rt, {}) is int
    assert KediType("`missing_name_for_type_resolution`").resolve(rt, {}) is None
    assert KediType("missing_name_for_type_resolution").resolve(rt, {}) is None
    empty_legacy = KediType("int")
    empty_legacy._legacy_str = ""
    assert empty_legacy._legacy_resolve(rt, {}) is None
    assert str(KediType(None)) == "Any"

    class RaisingTypeAdapter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("adapter failed")

    monkeypatch.setattr(type_system_module, "TypeAdapter", RaisingTypeAdapter)
    assert KediType("int").get_adapter(rt, {}) is None


def test_model_coercion_registration_defaults_and_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = KediRuntime()
    person = Person(name="Ada", age=1)

    assert coerce_model_value(Person, person) is person
    other = OtherModel(value="kept")
    assert coerce_model_value(Person, other) is other
    assert coerce_model_value(Person, {"name": "Ada", "age": 2}).age == 2
    assert coerce_model_value(Person, '{"name": "Ada", "age": 3}').age == 3
    assert coerce_model_value(Person, "name=Ada age=4 active=true").active is True
    with pytest.raises(ValueError, match="Cannot coerce str"):
        coerce_model_value(Person, "[1, 2]")
    monkeypatch.setattr(type_system_module.json, "loads", lambda _: [1, 2])
    with pytest.raises(ValueError, match="non-object JSON"):
        coerce_model_value(Person, '{"name": "Ada"}')
    monkeypatch.undo()
    with pytest.raises(ValueError, match="Cannot coerce int"):
        coerce_model_value(Person, 1)

    assert is_value_compatible(KediType("int"), 1, rt)
    assert not is_value_compatible(KediType("int"), "1", rt)

    type_env: dict[str, Any] = {}
    with pytest.raises(TypeError, match="requires a type annotation"):
        register_type(
            rt,
            TypeDef(name="BadDefault", fields=[TypeField(name="x", default_expr="1")]),
            type_env,
        )
    with pytest.raises(TypeError, match="Unknown type annotation"):
        register_type(
            rt,
            TypeDef(
                name="BadAnnotation",
                fields=[TypeField(name="x", type_annotation="MissingType")],
            ),
            type_env,
        )

    register_type(
        rt,
        TypeDef(
            name="Box",
            fields=[
                TypeField(name="name"),
                TypeField(name="count", type_annotation="int", default_expr="1"),
                TypeField(name="items", type_annotation="list[int]", default_expr="[1]"),
            ],
        ),
        type_env,
    )
    box_model = type_env["Box"]
    first = box_model("Ada")
    second = box_model("Bob")
    assert first.name == "Ada"
    assert first.count == 1
    assert first.items == [1]
    first.items.append(2)
    assert second.items == [1]
    assert type_env["__Box_fields__"]["name"] is str


def test_literal_model_repr_default_and_native_assignment_helpers() -> None:
    rt = KediRuntime()

    assert _default_factory({"x": [1]})() == {"x": [1]}
    assert _is_immutable_default(None)
    assert _is_immutable_default(("x", 1))
    assert not _is_immutable_default(("x", []))
    assert _is_immutable_default(frozenset({1, "x"}))
    assert not _is_immutable_default(frozenset({object()}))
    assert not _is_immutable_default([])

    assert _convert_basic_literal("1", int) == 1
    assert _convert_basic_literal("1.5", float) == 1.5
    assert _convert_basic_literal("x", str) == "x"
    assert _convert_basic_literal("true", bool) is True
    assert _convert_basic_literal("0", bool) is False
    assert _convert_basic_literal("anything", bool) is True
    assert _convert_basic_literal("not-int", int) == "not-int"
    assert _convert_basic_literal("value", object) == "value"

    assert _parse_model_repr("", Person) is None
    assert _parse_model_repr('"unterminated', Person) is None
    assert _parse_model_repr("name", Person) is None
    assert _parse_model_repr("missing=value", Person) is None
    assert _parse_model_repr("name=Ada age=5 active=yes", Person) == {
        "name": "Ada",
        "age": 5,
        "active": True,
    }

    assert assign_value_from_native("x", KediType(None), rt, {}, strict_native=True) == "x"
    with pytest.raises(TypeError, match="requires explicit type annotation"):
        assign_value_from_native(1, KediType(None), rt, {}, strict_native=True)
    with pytest.raises(TypeError, match="Annotation type mismatch"):
        assign_value_from_native("1", KediType("int"), rt, {}, strict_native=True)
    assert assign_value_from_native("x", KediType("str"), rt, {}, strict_native=True) == "x"
    assert assign_value_from_native(1, KediType("int"), rt, {}, strict_native=True) == 1
    with pytest.raises(TypeError, match="Native assignment value"):
        assign_value_from_native([1], KediType("int"), rt, {}, strict_native=True)
    with pytest.raises(TypeError, match="cannot have a non-str annotation"):
        assign_value_from_native(1, KediType("int"), rt, {}, strict_native=False)
    assert assign_value_from_native(1, KediType(None), rt, {}, strict_native=False) == "1"

    assert detect_rhs_kind([]) == (False, False, False)
    pop_type_env(KediRuntime())

    rt = KediRuntime()
    rt._type_env_stack = None
    pushed = type_system_module.push_type_env(rt)
    assert pushed == {}
    assert rt._type_env_stack == [pushed]
