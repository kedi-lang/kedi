"""LangGraph checkpoints keep Kedi's dynamic output models as plain data (#211)."""

import logging

from pydantic import BaseModel

from kedi.agent_adapter.adapters.langchain.checkpoint import KediCheckpointSerializer, _plain
from kedi.agent_adapter.output_schema import _PortableOutputModel
from kedi.dynamic_models import build_dynamic_model


class Static(BaseModel):
    team: str


def test_dynamic_output_models_round_trip_as_data_without_a_warning(caplog):
    output = build_dynamic_model("_OutputModel", _PortableOutputModel, {"team": (str, ...)})
    serializer = KediCheckpointSerializer()
    state = {"structured_response": output(team="billing"), "writes": [("x", output(team="it"))]}

    with caplog.at_level(logging.WARNING):
        restored = serializer.loads_typed(serializer.dumps_typed(state))

    assert restored["structured_response"] == {"team": "billing"}
    assert restored["writes"][0][1] == {"team": "it"}
    assert not [record for record in caplog.records if "unregistered" in record.getMessage()]


def test_only_kedi_dynamic_models_are_flattened():
    static = Static(team="billing")
    assert _plain({"value": static})["value"] is static
    assert _plain(("a", [1])) == ("a", [1])
