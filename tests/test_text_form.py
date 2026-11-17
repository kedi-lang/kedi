"""The text form of a value: what ``<x>`` renders and ``<<x>>`` attaches for plain data (#175)."""

from __future__ import annotations

import collections
import datetime
import decimal
import uuid
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from pathlib import PurePosixPath
from typing import Any, NamedTuple

import pytest
from pydantic import BaseModel

from kedi.lang.compiler.environment import substitution_text
from kedi.media import Document
from kedi.text_form import DEPTH_LIMIT, NoTextForm, text_form


class Color(StrEnum):
    RED = "red"


class Level(IntEnum):
    HIGH = 3


class Pair(Enum):
    ORIGIN = (0, 0)


class Leaky(str):
    def __repr__(self) -> str:
        raise AssertionError("a user __repr__ ran")

    def __str__(self) -> str:
        raise AssertionError("a user __str__ ran")


class LeakyInt(int):
    def __repr__(self) -> str:
        raise AssertionError("a user __repr__ ran")

    def __str__(self) -> str:
        raise AssertionError("a user __str__ ran")


class LeakyFloat(float):
    def __repr__(self) -> str:
        raise AssertionError("a user __repr__ ran")


class Point(NamedTuple):
    x: int


class Owner(BaseModel):
    name: str


@dataclass
class Line:
    qty: int


class Client:
    def __repr__(self) -> str:
        raise AssertionError("a user __repr__ ran")


UTC_NOON = datetime.datetime(2026, 11, 16, 12, 0, tzinfo=datetime.timezone.utc)


RENDERED = (
    ("çay", "çay", False),
    (Leaky("x"), "x", False),
    (True, "true", False),
    (5, "5", False),
    (LeakyInt(7), "7", False),
    (2**80, "1208925819614629174706176", False),
    (1.5, "1.5", False),
    (LeakyFloat(2.5), "2.5", False),
    (decimal.Decimal("1.10"), "1.10", False),
    (UTC_NOON, "2026-11-16T12:00:00Z", False),
    (datetime.datetime(2026, 11, 16, 12, 0), "2026-11-16T12:00:00", False),
    (datetime.date(2026, 11, 16), "2026-11-16", False),
    (datetime.time(9, 5), "09:05:00", False),
    (uuid.UUID(int=1), "00000000-0000-0000-0000-000000000001", False),
    (Color.RED, "red", False),
    (Level.HIGH, "3", False),
    (Pair.ORIGIN, "[0,0]", True),
    (
        [Color.RED, Level.HIGH, Leaky("x"), LeakyInt(1), None, "çay"],
        '["red",3,"x",1,null,"çay"]',
        True,
    ),
    ((1, "a"), '[1,"a"]', True),
    ([], "[]", True),
    ({"a": [1, None, True], Leaky("k"): {"z": 1.5}}, '{"a":[1,null,true],"k":{"z":1.5}}', True),
    ([decimal.Decimal("1.10"), UTC_NOON], '["1.10","2026-11-16T12:00:00Z"]', True),
)


# Explicit ids: pytest would otherwise call the leaky subclasses' __str__ to name the cases.
@pytest.mark.parametrize(
    ("value", "text", "structured"), RENDERED, ids=[f"rendered-{i}" for i in range(len(RENDERED))]
)
def test_values_with_a_text_form_render_from_their_data(
    value: Any, text: str, structured: bool
) -> None:
    assert text_form(value) == (text, structured)


def test_shared_values_render_and_cycles_fail() -> None:
    shared = [1]
    assert text_form([shared, shared]) == ("[[1],[1]]", True)
    loop: list[Any] = [1]
    loop.append(loop)
    with pytest.raises(NoTextForm, match="contains itself") as caught:
        text_form(loop)
    assert caught.value.path == "[1]"
    table: dict[str, Any] = {}
    table["self"] = table
    with pytest.raises(NoTextForm, match="contains itself"):
        text_form(table)


def test_depth_is_limited() -> None:
    def nested(levels: int) -> list[Any]:
        value: list[Any] = []
        for _ in range(levels):
            value = [value]
        return value

    assert text_form(nested(DEPTH_LIMIT))[1] is True
    with pytest.raises(NoTextForm, match=f"nests deeper than {DEPTH_LIMIT} levels"):
        text_form(nested(DEPTH_LIMIT + 1))


@pytest.mark.parametrize(
    ("value", "path", "reason", "hint"),
    [
        (None, "", "is None", None),
        (float("nan"), "", "is not a finite number", None),
        ([float("inf")], "[0]", "is not a finite number", None),
        (decimal.Decimal("NaN"), "", "is not a finite number", None),
        (b"x", "", "is bytes", None),
        ({"k": {1}}, '["k"]', "is of type set, whose order is not stable", "sort it first"),
        (frozenset({1}), "", "is of type frozenset, whose order is not stable", "sort it first"),
        (PurePosixPath("a/b"), "", "is a path", "render it with <`str(...)`>"),
        (Point(1), "", "is a NamedTuple (Point), which would lose its field names", "`._asdict()`"),
        (
            collections.OrderedDict(a=1),
            "",
            "is of type OrderedDict, a dict subclass",
            "`dict(...)`",
        ),
        (collections.UserList([1]), "", "is of type UserList", None),
        (type("Row", (list,), {})([1]), "", "is of type Row, a list subclass", "`list(...)`"),
        ({1: "a"}, "", "has a key of type int; dict keys must be str", None),
        ([datetime.timedelta(seconds=1)], "[0]", "is of type timedelta, which has no single", None),
        (1j, "", "is of type complex, which has no single text form", None),
        (Owner(name="Ada"), "", "is of type Owner", None),
        ([Line(qty=1)], "[0]", "is of type Line", None),
        ({"items": [1, 2, Client()]}, '["items"][2]', "is of type Client", None),
        (Line, "", "is of type type", None),
    ],
)
def test_values_without_a_text_form_name_the_part_and_the_fix(
    value: Any, path: str, reason: str, hint: str | None
) -> None:
    with pytest.raises(NoTextForm) as caught:
        text_form(value)
    assert caught.value.path == path
    assert caught.value.reason.startswith(reason)
    if hint is None:
        assert caught.value.hint is None
    else:
        assert caught.value.hint is not None and hint in caught.value.hint


def test_media_has_no_text_form() -> None:
    note = Document(media_type="text/plain", data=b"hi")
    with pytest.raises(NoTextForm) as caught:
        text_form({"note": note})
    assert caught.value.path == '["note"]'
    assert caught.value.reason == f"is {note.label}"


def test_references_also_accept_models_and_dataclasses() -> None:
    assert text_form(Owner(name="Ada"), models=True) == ('{"name":"Ada"}', True)
    assert text_form([Line(qty=1)], models=True) == ('[{"qty":1}]', True)
    with pytest.raises(NoTextForm, match="is of type set"):
        text_form([Owner(name="Ada"), {1}], models=True)


def test_describe_writes_one_sentence() -> None:
    error = NoTextForm('["k"]', "is of type set, whose order is not stable", "sort it first")
    assert str(error) == '["k"] is of type set, whose order is not stable'
    assert error.describe("order") == (
        'order["k"] is of type set, whose order is not stable. Sort it first.'
    )
    assert NoTextForm("", "is None").describe("the value") == "the value is None."


def test_substitution_errors_name_the_path_and_never_run_user_code() -> None:
    assert substitution_text([Leaky("a"), Color.RED], "tags") == '["a","red"]'
    with pytest.raises(TypeError) as caught:
        substitution_text({"items": [1, 2, Client()]}, "order")
    assert str(caught.value) == (
        '<order> renders text, but order["items"][2] is of type Client. In a model prompt pass '
        "it by reference with <<order>>; to keep the value use single-backtick Python; for text "
        "convert it explicitly, for example <`str(...)`>."
    )
    with pytest.raises(TypeError, match=r"^<`load\(\)`> renders text, but the part at \[0\] is "):
        substitution_text([Client()], "`load()`")
    with pytest.raises(
        TypeError,
        match=r"^<items> renders text, but the value is of type set, whose order is not stable\. Sort it first, for example <`sorted\(\.\.\.\)`>\. In a model prompt",
    ):
        substitution_text({1}, "items")


def test_native_values_are_never_converted_to_text() -> None:
    from kedi.lang import compile_program, parse_program

    source = (
        "```\n"
        "class Loud:\n"
        "    def __str__(self):\n"
        "        raise AssertionError('a user __str__ ran')\n"
        "    __repr__ = __str__\n"
        "```\n"
        "[made: Any] = `Loud()`\n"
        "[made] := `Loud()`\n"
        "@build() -> Any:\n  = `Loud()`\n"
        "[built: Any] = `build()`\n"
        "= `type(made).__name__ + type(built).__name__`\n"
    )
    assert compile_program(parse_program(source)).run_main() == "LoudLoud"


def test_text_named_like_a_type_stays_text() -> None:
    from kedi.lang import compile_program, parse_program

    # A model's str() form is never parsed back into the model.
    source = "~Pt(x: int, y: int)\n[Pt] = x\\=1 y\\=2\n= `type(Pt).__name__ + ':' + Pt`\n"
    assert compile_program(parse_program(source)).run_main() == "str:x=1 y=2"


class Inner(BaseModel):
    tags: set[str] = set()
    score: float = 1.0


class Report(BaseModel):
    file: PurePosixPath = PurePosixPath("a/b")
    took: datetime.timedelta = datetime.timedelta(seconds=90)
    raw: bytes = b"x"
    inner: Inner = Inner()
    items: list[Inner] = []


@dataclass
class Row:
    values: list[float]


def test_references_write_models_by_their_schema() -> None:
    assert text_form(Report(inner=Inner(tags={"a"})), models=True) == (
        '{"file":"a/b","took":"PT1M30S","raw":"x","inner":{"tags":["a"],"score":1.0},"items":[]}',
        True,
    )


@pytest.mark.parametrize(
    ("value", "path", "reason"),
    [
        (Report(items=[Inner(), Inner(tags={"a", "b"})]), ".items[1].tags", "is of type set"),
        (Report(inner=Inner(score=float("nan"))), ".inner.score", "is not a finite number"),
        ([Row(values=[1.0, float("inf")])], "[0].values[1]", "is not a finite number"),
        ({"rows": [Row(values=[decimal.Decimal("NaN")])]}, '["rows"][0].values[0]', "is not a"),
        (Owner.model_construct(name={"a": {1, 2}}), '.name["a"]', "is of type set"),
        (Owner.model_construct(name=(1, {1, 2})), ".name[1]", "is of type set"),
        (Owner.model_construct(name={Client(): {1, 2}}), ".name[<Client>]", "is of type set"),
        (Owner.model_construct(name={7: {1, 2}}), ".name[7]", "is of type set"),
    ],
)
def test_references_reject_unstable_or_non_finite_model_data(
    value: Any, path: str, reason: str
) -> None:
    with pytest.raises(NoTextForm) as caught:
        text_form(value, models=True)
    assert caught.value.path == path
    assert caught.value.reason.startswith(reason)


def test_model_data_cycles_and_depth_are_bounded() -> None:
    loop: list[Any] = []
    loop.append(loop)
    with pytest.raises(NoTextForm, match="contains itself"):
        text_form(Owner.model_construct(name=loop), models=True)
    deep: list[Any] = []
    for _ in range(DEPTH_LIMIT + 1):
        deep = [deep]
    with pytest.raises(NoTextForm, match="nests deeper"):
        text_form(Owner.model_construct(name=deep), models=True)


def test_inputs_with_a_text_form_keep_a_decision_identity() -> None:
    from kedi.decisions import fingerprint_value

    values = [
        decimal.Decimal("1.10"),
        datetime.date(2026, 11, 16),
        UTC_NOON,
        datetime.time(9, 5),
        uuid.UUID(int=1),
        Color.RED,
        Level.HIGH,
    ]
    prints = [fingerprint_value(value) for value in values]
    assert None not in prints
    assert len(set(prints)) == len(prints)
    assert fingerprint_value(decimal.Decimal("1.1")) != prints[0]
    assert fingerprint_value(datetime.datetime(2026, 11, 16)) != prints[1]


def test_filled_outputs_render_like_substitutions() -> None:
    from kedi.lang import compile_program, parse_program
    from tests.mock_adapter import MockAdapter

    source = "[r] = Decide [ok: bool] and [xs: list[str]].\n[a] = Then <ok> and <xs>.\n= <r>|<a>\n"
    result = compile_program(parse_program(source), adapter=MockAdapter()).run_main()
    assert result == 'Decide true and ["xs-a","xs-b"].|Then true and ["xs-a","xs-b"].'


def test_a_filled_output_without_text_is_a_type_error() -> None:
    from kedi.core.runtime import KediRuntime
    from kedi.lang.ast import OutputSegment, TextSegment
    from kedi.lang.compiler.evaluation import _segment_parts

    rt = KediRuntime()
    segments = [TextSegment(text="Made "), OutputSegment(name="p"), OutputSegment(name="gone")]
    with pytest.raises(TypeError, match=r"^<p> renders text, but the value is of type Owner\."):
        _segment_parts(rt, segments, {"p": Owner(name="Ada")}, context="assign")
    token = rt._current_env.set({"p": Owner(name="Ada")})
    try:
        with pytest.raises(TypeError, match="of type Owner"):
            _segment_parts(rt, segments, {}, context="assign")
    finally:
        rt._current_env.reset(token)
    assert _segment_parts(rt, segments, {"p": 3}, context="assign") == ["Made ", "3", ""]


def test_harness_tool_results_never_run_user_code() -> None:
    from kedi.agent_adapter.conversation import _portable_content
    from kedi.agent_adapter.tools.execution import tool_result_text

    assert tool_result_text("plain") == "plain"
    assert tool_result_text(None) == "null"
    from pathlib import Path

    assert tool_result_text({"day": datetime.date(2026, 11, 16), "path": Path("a")}) == (
        '{"day":"2026-11-16","path":"a"}'
    )
    with pytest.raises(Exception, match="Unable to serialize unknown type: <class '.*Client'>"):
        tool_result_text({"client": Client()})
    assert _portable_content("plain") == "plain"
    assert _portable_content({"client": Client(), "n": 1}) == '{"client": "<Client>", "n": 1}'
