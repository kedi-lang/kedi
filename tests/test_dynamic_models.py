from __future__ import annotations

from pydantic import BaseModel

from kedi.dynamic_models import build_dynamic_model
from kedi.resolved_types import ResolvedKediType


def test_build_dynamic_model_creates_typed_pydantic_model() -> None:
    fields: dict[str, tuple[ResolvedKediType, object]] = {
        "name": (str, ...),
        "count": (int, 0),
    }

    model = build_dynamic_model("SampleModel", BaseModel, fields)
    instance = model(name="Ada")

    assert instance.model_dump() == {"name": "Ada", "count": 0}
    assert model.model_fields["name"].annotation is str
