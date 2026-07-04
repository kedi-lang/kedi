from __future__ import annotations

import inspect
import typing
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Mapping, get_args, get_origin

import pytest
from pydantic import BaseModel

from kedi.core import KediRuntime
from kedi.executors.pyodide import (
    PyodideExecutor,
    PyodideReference,
    _decode_type_descriptor,
    _json_value,
    _prelude_names,
    _referenced_names,
    _response_env,
    _serialize_env,
)
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import TypeRef
from kedi.lang.compiler.type_system import KediType
from kedi.promise import KediPromise


class QueueBridge:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.timeouts: list[float] = []

    def request(self, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]:
        self.requests.append(dict(payload))
        self.timeouts.append(timeout)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class MemoryExporter:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def export(self, event: Any) -> None:
        self.events.append(event)


class Payload(BaseModel):
    value: int


@dataclass
class Record:
    name: str


def test_pyodide_executor_routes_every_execution_operation() -> None:
    bridge = QueueBridge(
        {"ok": True, "result": 3, "env": {"x": 3}},
        {"ok": True, "result": "done", "env": {"x": 4}},
        {"ok": True, "result": None, "env": {"x": 5}},
        {"ok": True, "result": None, "env": {"seed": 7}},
        {"ok": True, "result": 5, "env": {}},
    )
    exporter = MemoryExporter()
    executor = PyodideExecutor(bridge, timeout=12, debug_exporter=exporter)
    runtime = KediRuntime()
    env = {"x": 1}

    assert executor.evaluate_inline(runtime, "x + 2", env) == 3
    assert env == {"x": 3}
    assert executor.execute_block(runtime, "return 'done'", env) == "done"
    assert env == {"x": 4}
    assert executor.execute_side_effects(runtime, "x += 1", env, kedi_line_offset=9) is None
    assert env == {"x": 5}
    assert executor.execute_prelude(runtime, "seed = 7") == {"seed": 7}

    assert [request["action"] for request in bridge.requests] == [
        "evaluate_inline",
        "execute_block",
        "execute_side_effects",
        "execute_prelude",
    ]
    assert bridge.requests[2]["kediLineOffset"] == 9
    assert bridge.requests[3]["syncNames"] == ["seed"]
    assert bridge.timeouts == [12, 12, 12, 12]
    assert [event.event for event in exporter.events] == [
        "execution_started",
        "execution_completed",
        "execution_started",
        "execution_completed",
        "execution_started",
        "execution_completed",
    ]
    executor.set_debug_exporter(None)
    assert executor.evaluate_inline(runtime, "x", env) == 5


def test_pyodide_execution_layers_runtime_environment_without_syncing_it_back() -> None:
    bridge = QueueBridge({"ok": True, "result": "ok", "env": {"local": "updated"}})
    executor = PyodideExecutor(bridge)
    runtime = KediRuntime()
    runtime._prelude_env = {
        "helper": PyodideReference("helper-function"),
    }
    local_env = {"local": "before"}

    assert executor.evaluate_inline(runtime, "helper(local)", local_env) == "ok"
    assert bridge.requests[0]["env"] == {
        "helper": {"__kedi_pyodide_ref__": "helper-function"},
        "local": "before",
    }
    assert bridge.requests[0]["syncNames"] == ["local"]
    assert local_env == {"local": "updated"}


def test_pyodide_executor_surfaces_bridge_and_response_errors() -> None:
    exporter = MemoryExporter()
    executor = PyodideExecutor(
        QueueBridge(
            {"ok": False, "errorType": "ValueError", "error": "bad value"},
            [],
        ),
        debug_exporter=exporter,
    )

    with pytest.raises(RuntimeError, match="ValueError: bad value"):
        executor.evaluate_inline(KediRuntime(), "bad", {})
    assert exporter.events[-1].event == "execution_failed"

    with pytest.raises(TypeError, match="response must be an object"):
        executor.evaluate_inline(KediRuntime(), "bad", {})


def test_pyodide_executor_collects_worker_stdout() -> None:
    executor = PyodideExecutor(
        QueueBridge(
            {"ok": True, "result": None, "env": {}, "stdout": "first"},
            {"ok": True, "result": None, "env": {}, "stdout": " second"},
            {"ok": True, "result": None, "env": {}, "stdout": []},
        )
    )

    executor.execute_side_effects(KediRuntime(), "print('first')", {})
    executor.execute_side_effects(KediRuntime(), "print('second')", {})

    assert executor.drain_stdout() == "first second"
    assert executor.drain_stdout() == ""
    with pytest.raises(TypeError, match="stdout must be a string"):
        executor.execute_side_effects(KediRuntime(), "print('bad')", {})


def test_builtin_module_stdout_is_collected_during_compile() -> None:
    executor = PyodideExecutor(
        QueueBridge(
            {
                "ok": True,
                "result": None,
                "env": {},
                "stdout": "The Zen of Python\n",
            }
        )
    )

    runtime = compile_program(
        parse_program("> import: this\n", source_path="<playground-test>"),
        adapter=None,
        executor=executor,
    )

    assert runtime.run_main() is None
    assert executor.drain_stdout() == "The Zen of Python\n"


def test_pyodide_prelude_exports_keep_browser_owned_objects_opaque() -> None:
    bridge = QueueBridge(
        {
            "ok": True,
            "result": None,
            "env": {
                "ModuleNotInstalledError": {
                    "__kedi_pyodide_ref__": "error-class",
                }
            },
        },
        {
            "ok": True,
            "result": None,
            "env": {
                "importlib": {
                    "__kedi_pyodide_ref__": "importlib-module",
                }
            },
        },
        {"ok": True, "result": None, "env": {}},
        {
            "ok": True,
            "type": {"kind": "literal", "values": [True]},
            "env": {},
        },
    )
    executor = PyodideExecutor(bridge)

    runtime = compile_program(
        parse_program("> import: require\n", source_path="<playground-test>"),
        adapter=None,
        executor=executor,
    )

    assert runtime.run_main() is None
    assert "require" in runtime._procedures
    assert bridge.requests[2]["env"]["ModuleNotInstalledError"] == {
        "__kedi_pyodide_ref__": "error-class"
    }


def test_pyodide_prelude_syncs_only_names_defined_by_that_prelude() -> None:
    assert _prelude_names(
        """
import importlib.util
from collections.abc import Sequence as ItemSequence

seed = 7

def package_is_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None
"""
    ) == ["importlib", "ItemSequence", "seed", "package_is_installed"]


def test_pyodide_dynamic_function_preserves_signature_and_defaults() -> None:
    executor = PyodideExecutor(QueueBridge())
    fn = executor.create_dynamic_function(
        "add",
        ["left", "right"],
        lambda **values: values["left"] + values["right"],
        {"right": 2},
    )

    assert fn(3) == 5
    assert getattr(fn, "__name__") == "add"
    assert str(inspect.signature(fn)) == "(left, right=2)"
    with pytest.raises(TypeError):
        fn()


@pytest.mark.parametrize(
    ("descriptor", "expected"),
    [
        ({"kind": "none"}, type(None)),
        ({"kind": "builtin", "name": "str"}, str),
        ({"kind": "typing", "name": "Any"}, Any),
        (
            {
                "kind": "generic",
                "origin": {"kind": "builtin", "name": "list"},
                "args": [{"kind": "builtin", "name": "int"}],
            },
            list[int],
        ),
        (
            {
                "kind": "generic",
                "origin": {"kind": "builtin", "name": "dict"},
                "args": [
                    {"kind": "builtin", "name": "str"},
                    {"kind": "builtin", "name": "int"},
                ],
            },
            dict[str, int],
        ),
    ],
)
def test_decode_pyodide_type_descriptors(descriptor: dict[str, Any], expected: Any) -> None:
    assert _decode_type_descriptor(descriptor) == expected


def test_decode_pyodide_union_and_literal_descriptors() -> None:
    union = _decode_type_descriptor(
        {
            "kind": "union",
            "args": [
                {"kind": "builtin", "name": "str"},
                {"kind": "none"},
            ],
        }
    )
    literal = _decode_type_descriptor({"kind": "literal", "values": ["a", 1]})

    assert get_args(union) == (str, type(None))
    assert get_origin(literal) is typing.Literal
    assert get_args(literal) == ("a", 1)


def test_decode_pyodide_pydantic_model_descriptor() -> None:
    model = _decode_type_descriptor(
        {
            "kind": "pydantic_model",
            "name": "LiteralOutput",
            "fields": [
                {
                    "name": "capital",
                    "annotation": {"kind": "builtin", "name": "str"},
                    "required": True,
                    "description": "Generate literal output.",
                    "examples": ["Ankara"],
                },
                {
                    "name": "confidence",
                    "annotation": {"kind": "builtin", "name": "float"},
                    "required": False,
                    "default": 1.0,
                },
            ],
        }
    )

    assert issubclass(model, BaseModel)
    assert model.__name__ == "LiteralOutput"
    assert model.model_fields["capital"].description == "Generate literal output."
    assert model.model_fields["capital"].examples == ["Ankara"]
    assert model(capital="Ankara").model_dump() == {
        "capital": "Ankara",
        "confidence": 1.0,
    }


def test_kedi_type_resolves_browser_owned_pydantic_model_reference() -> None:
    bridge = QueueBridge(
        {
            "ok": True,
            "type": {
                "kind": "pydantic_model",
                "name": "LiteralOutput",
                "fields": [
                    {
                        "name": "capital",
                        "annotation": {"kind": "builtin", "name": "str"},
                        "required": True,
                    }
                ],
            },
            "env": {},
        }
    )
    executor = PyodideExecutor(bridge)
    runtime = KediRuntime(executor=executor)

    resolved = KediType(TypeRef(name="LiteralOutput")).resolve(
        runtime,
        {"LiteralOutput": PyodideReference("literal-output")},
    )

    assert isinstance(resolved, type)
    assert issubclass(resolved, BaseModel)
    assert resolved(capital="Ankara").capital == "Ankara"
    assert bridge.requests[0]["env"]["LiteralOutput"] == {"__kedi_pyodide_ref__": "literal-output"}


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        ({"kind": "builtin", "name": "missing"}, "Unknown Pyodide builtin"),
        ({"kind": "typing", "name": "missing"}, "Unknown Pyodide typing"),
        ({"kind": "union", "args": []}, "at least one"),
        ({"kind": "generic", "origin": None, "args": []}, "origin must be an object"),
        ({"kind": "union", "args": [None]}, "union argument must be an object"),
        ({"kind": "literal", "values": None}, "values must be a list"),
        (
            {"kind": "pydantic_model", "name": "", "fields": []},
            "model name must be a non-empty string",
        ),
        (
            {
                "kind": "pydantic_model",
                "name": "Output",
                "fields": [{"name": "", "annotation": {"kind": "builtin", "name": "str"}}],
            },
            "field name must be a non-empty string",
        ),
        (
            {
                "kind": "pydantic_model",
                "name": "Output",
                "fields": [
                    {
                        "name": "value",
                        "annotation": {"kind": "builtin", "name": "str"},
                        "required": "yes",
                    }
                ],
            },
            "required flag must be a boolean",
        ),
        ({"kind": "custom"}, "Unsupported Pyodide type descriptor"),
    ],
)
def test_decode_pyodide_type_descriptor_errors(
    descriptor: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        _decode_type_descriptor(descriptor)


def test_pyodide_type_expression_uses_browser_descriptor() -> None:
    executor = PyodideExecutor(
        QueueBridge(
            {"ok": True, "result": None, "env": {}},
            {
                "ok": True,
                "type": {
                    "kind": "generic",
                    "origin": {"kind": "builtin", "name": "list"},
                    "args": [{"kind": "builtin", "name": "str"}],
                },
                "env": {},
            },
            {"ok": True, "type": "bad", "env": {}},
        )
    )

    assert executor.evaluate_type_expression(KediRuntime(), "None", {}) is None
    assert executor.evaluate_type_expression(KediRuntime(), "list[str]", {}) == list[str]
    with pytest.raises(TypeError, match="value or type descriptor"):
        executor.evaluate_type_expression(KediRuntime(), "bad", {})


def test_pyodide_serialization_is_json_safe_and_promise_aware() -> None:
    future: Future[dict[str, Any]] = Future()
    future.set_result({"used": 9, "unused": 10})
    env = {
        "payload": Payload(value=3),
        "record": Record("Ada"),
        "items": (1, 2),
        "mapping": {"ok": True},
        "used": KediPromise(future, "used"),
        "unused": KediPromise(future, "unused"),
        "__builtins__": {},
    }

    serialized = _serialize_env(env, {"payload", "record", "items", "mapping", "used"})

    assert serialized == {
        "payload": {"value": 3},
        "record": {"name": "Ada"},
        "items": [1, 2],
        "mapping": {"ok": True},
        "used": 9,
    }
    assert _json_value({"items": {1, 2}})["items"] in ([1, 2], [2, 1])
    with pytest.raises(TypeError, match="string keys"):
        _json_value({1: "bad"})
    with pytest.raises(TypeError, match="Unsupported"):
        _json_value(object())
    with pytest.raises(TypeError, match="'bad' cannot cross"):
        _serialize_env({"bad": object()}, {"bad"})
    assert _serialize_env({"ignored": object()}, set()) == {}
    assert _serialize_env({"Literal": typing.Literal}, {"Literal"}) == {}
    assert _serialize_env({"typing": typing}, {"typing"}) == {}
    assert _serialize_env({"typing_alias": typing}, set()) == {}
    assert _json_value(PyodideReference("opaque")) == {"__kedi_pyodide_ref__": "opaque"}


def test_pyodide_response_and_name_analysis_validation() -> None:
    assert _response_env(
        {
            "env": {
                "x": 1,
                "opaque": {"__kedi_pyodide_ref__": "handle"},
                "nested": [{"__kedi_pyodide_ref__": "nested-handle"}],
                "mapping": {
                    "opaque": {"__kedi_pyodide_ref__": "mapping-handle"},
                },
            }
        }
    ) == {
        "x": 1,
        "opaque": PyodideReference("handle"),
        "nested": [PyodideReference("nested-handle")],
        "mapping": {"opaque": PyodideReference("mapping-handle")},
    }
    assert _referenced_names("x + y", "evaluate_type_expression") == {"x", "y"}
    assert "value" in _referenced_names("return value", "execute_block")

    with pytest.raises(TypeError, match="env must be an object"):
        _response_env({"env": []})
    with pytest.raises(TypeError, match="env must be an object"):
        _response_env({"env": {1: "bad"}})
    with pytest.raises(TypeError, match="reference handle must be a string"):
        _response_env({"env": {"bad": {"__kedi_pyodide_ref__": 1}}})
