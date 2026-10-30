from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_adapter.tools.conversion import kedi_procedure_tool_spec
from kedi.interactive_session import InteractiveSession
from kedi.lang.procedure_introspection import get_procedure_def

pytest.importorskip("cloudpickle")


@pytest.mark.parametrize("keep_function_binding", [False, True])
@pytest.mark.parametrize("namespace_first", [False, True])
def test_exec_function_keeps_explicit_globals_identity(
    tmp_path: Path,
    keep_function_binding: bool,
    namespace_first: bool,
) -> None:
    namespace: dict[str, Any] = {"x": 1}
    exec("def f():\n    return x\n", namespace, namespace)
    function = namespace["f"]
    if not keep_function_binding:
        del namespace["f"]
    bindings = [("namespace", namespace), ("saved", function)]
    if not namespace_first:
        bindings.reverse()
    snapshot = tmp_path / "namespace.state"
    with InteractiveSession(runtime_globals=dict(bindings)) as session:
        session.dump(snapshot, format="python")
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        restored_namespace = restored._env["namespace"]
        restored_function = restored._env["saved"]
        assert restored_function.__globals__ is restored_namespace
        assert restored_function() == 1
        restored_namespace["x"] = 2
        assert restored_function() == 2
        restored.execute("[observed: int] = `saved()`")
        assert restored._env["observed"] == 2
        restored.dump(tmp_path / "namespace-again.state", format="python")
    with InteractiveSession.load(tmp_path / "namespace-again.state", trusted=True) as restored:
        assert restored._env["saved"].__globals__ is restored._env["namespace"]
        assert restored._env["saved"]() == 2


def test_saved_procedure_tool_keeps_tool_context_and_body_retries(tmp_path: Path) -> None:
    snapshot = tmp_path / "tool.state"
    with InteractiveSession() as session:
        session.execute("""```
from kedi.tool_context import inside_kedi_tool_call
attempts = []
def backend():
    attempts.append(inside_kedi_tool_call())
    if len(attempts) % 2:
        raise TimeoutError("retry the body")
    return len(attempts)
```
@lookup() -> int:
    > tool:
        retries: 1
        retry_on:
            TimeoutError
    = `backend()`
""")
        runtime = session._compiler.rt
        procedure = get_procedure_def(runtime._get_proc("lookup"))
        assert procedure is not None
        tool = kedi_procedure_tool_spec(runtime, procedure)
        assert tool.fn() == 2
        session._env.declare("saved_tool", tool.fn)
        session.dump(snapshot, format="python")
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        saved_tool = restored._env["saved_tool"]
        assert saved_tool() == 4
        restored.execute("[observed_attempts: Any] = `list(attempts)`")
        assert restored._env["observed_attempts"] == [True, True, True, True]
        restored.dump(tmp_path / "tool-again.state", format="python")
    with InteractiveSession.load(tmp_path / "tool-again.state", trusted=True) as restored:
        assert restored._env["saved_tool"]() == 6
        restored.execute("[observed_attempts: Any] = `list(attempts)`")
        assert restored._env["observed_attempts"] == [True] * 6


def test_generated_model_keeps_user_class_state_and_introspection(tmp_path: Path) -> None:
    snapshot = tmp_path / "model-class.state"
    with InteractiveSession() as session:
        session.execute("~Person(name: str)\n[person: Person] = `Person('Ada')`")
        model = type(session._env["person"])
        shared: list[str] = []
        model.greet = lambda self: f"Hello {self.name}"
        model.owner = classmethod(lambda cls: cls)
        model.label = staticmethod(lambda: "person")
        model.shared = shared
        model.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    "person_name", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
                )
            ]
        )
        expected_signature = str(inspect.signature(model))
        session._env.declare("class_payload", shared)
        session.dump(snapshot, format="python")
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        person = restored._env["person"]
        model = type(person)
        assert person.greet() == "Hello Ada"
        assert person.owner() is model
        assert person.label() == "person"
        assert model.shared is restored._env["class_payload"]
        model.shared.append("saved")
        assert restored._env["class_payload"] == ["saved"]
        assert str(inspect.signature(model)) == expected_signature
        restored.dump(tmp_path / "model-class-again.state", format="python")
    with InteractiveSession.load(tmp_path / "model-class-again.state", trusted=True) as restored:
        person = restored._env["person"]
        assert person.greet() == "Hello Ada"
        assert person.owner() is type(person)
        assert type(person).shared is restored._env["class_payload"]
        assert type(person).shared == ["saved"]
        assert str(inspect.signature(type(person))) == expected_signature


def test_module_closures_share_cells_and_recursive_aliases(tmp_path: Path) -> None:
    value = 10

    def advance() -> int:
        nonlocal value
        value += 1
        return value

    def current() -> int:
        return value

    def recursive(depth: int) -> Any:
        return recursive(depth - 1) if depth else (recursive, current, advance)

    shared_cell = current.__closure__[0]
    snapshot = tmp_path / "shared-cells.state"
    with InteractiveSession(
        runtime_globals={
            "advance": advance,
            "current": current,
            "recursive": recursive,
            "shared_cell": shared_cell,
        }
    ) as session:
        session.dump(snapshot, format="python")
    for expected in (11, 12):
        with InteractiveSession.load(snapshot, trusted=True) as restored:
            saved_advance = restored._env["advance"]
            saved_current = restored._env["current"]
            saved_recursive = restored._env["recursive"]
            cell = restored._env["shared_cell"]
            assert saved_advance.__closure__[0] is cell
            assert saved_current.__closure__[0] is cell
            assert saved_recursive(3) == (saved_recursive, saved_current, saved_advance)
            assert saved_advance() == expected
            assert saved_current() == expected
            assert cell.cell_contents == expected
            restored.dump(snapshot, format="python")
    assert current() == 10


def test_functions_keep_distinct_captured_builtins_after_globals_removal(tmp_path: Path) -> None:
    first_builtins = {"len": lambda value: 17}
    second_builtins = {"len": lambda value: 29}
    namespace: dict[str, Any] = {"__builtins__": first_builtins}
    exec("def first():\n    return len([])\n", namespace)
    namespace["__builtins__"] = second_builtins
    exec("def second():\n    return len([])\n", namespace)
    del namespace["__builtins__"]
    snapshot = tmp_path / "captured-builtins.state"
    with InteractiveSession(
        runtime_globals={
            "namespace": namespace,
            "first_builtins": first_builtins,
            "second_builtins": second_builtins,
        }
    ) as session:
        session.dump(snapshot, format="python")
    for _ in range(2):
        with InteractiveSession.load(snapshot, trusted=True) as restored:
            saved = restored._env["namespace"]
            assert "__builtins__" not in saved
            assert saved["first"].__globals__ is saved
            assert saved["second"].__globals__ is saved
            assert saved["first"].__builtins__ is restored._env["first_builtins"]
            assert saved["second"].__builtins__ is restored._env["second_builtins"]
            assert (saved["first"](), saved["second"]()) == (17, 29)
            restored.dump(snapshot, format="python")


@pytest.mark.parametrize("storage", ["slot", "class"])
@pytest.mark.parametrize("namespace_first", [False, True])
def test_function_namespaces_discovered_through_instance_state(
    tmp_path: Path, storage: str, namespace_first: bool
) -> None:
    namespace: dict[str, Any] = {"value": 10}
    exec("def read_value(self=None):\n    return value\n", namespace)
    callback = namespace.pop("read_value")

    class Holder:
        __slots__ = ("callback", "owner")

    if storage == "class":
        Holder.read_value = callback
    holder = Holder()
    holder.owner = holder
    if storage == "slot":
        holder.callback = callback
    bindings = [("namespace", namespace), ("holder", holder)]
    if not namespace_first:
        bindings.reverse()
    snapshot = tmp_path / "instance-namespace.state"
    with InteractiveSession(runtime_globals=dict(bindings)) as session:
        session.dump(snapshot, format="python")
    for _ in range(2):
        with InteractiveSession.load(snapshot, trusted=True) as restored:
            saved_holder = restored._env["holder"]
            saved_namespace = restored._env["namespace"]
            saved_callback = saved_holder.callback if storage == "slot" else saved_holder.read_value
            saved_function = saved_callback if storage == "slot" else saved_callback.__func__
            assert saved_holder.owner is saved_holder
            assert saved_function.__globals__ is saved_namespace
            saved_namespace["value"] += 1
            assert saved_callback() == saved_namespace["value"]
            restored.dump(snapshot, format="python")


def test_module_local_function_discovers_referenced_function_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    namespace: dict[str, Any] = {"value": 10}
    exec("def dependency():\n    return value\n", namespace)
    module = types.ModuleType("_snapshot_graph_dependency_module")
    module.dependency = namespace.pop("dependency")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(
        "def factory():\n    def call():\n        return dependency()\n    return call\n",
        vars(module),
    )
    snapshot = tmp_path / "function-global.state"
    with InteractiveSession(
        runtime_globals={"namespace": namespace, "call": module.factory()}
    ) as session:
        session.dump(snapshot, format="python")
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        saved_namespace = restored._env["namespace"]
        saved_call = restored._env["call"]
        assert saved_call.__globals__["dependency"].__globals__ is saved_namespace
        saved_namespace["value"] = 42
        assert saved_call() == 42


def test_cycles_cross_models_procedure_defaults_and_lexical_owner(tmp_path: Path) -> None:
    snapshot = tmp_path / "declaration-cycles.state"
    with InteractiveSession() as session:
        session.execute("""[payload: Any] = `[]`
~Node(payload: Any)
[node: Node] = `Node(payload)`
""")
        session.execute("""@read(value: Any = `payload`) -> Any:
    = `value`
""")
        payload = session._env["payload"]
        node = session._env["node"]
        procedure = session._compiler.rt._get_proc("read")
        payload.extend((payload, node, procedure, session._env))
        type(node).shared = payload
        session._env.declare("raw_procedure", procedure)
        session.dump(snapshot, format="python")
    for _ in range(2):
        with InteractiveSession.load(snapshot, trusted=True) as restored:
            payload = restored._env["payload"]
            node = restored._env["node"]
            procedure = restored._env["raw_procedure"]
            assert payload[0] is payload
            assert payload[1] is node
            assert payload[2] is procedure
            assert payload[3] is restored._env
            assert node.payload is payload
            assert type(node).shared is payload
            assert procedure.__defaults__[0] is payload
            assert procedure() is payload
            assert type(node)(payload).payload is payload
            restored.dump(snapshot, format="python")


@pytest.mark.parametrize("name", ["Literal", "Optional", "Union"])
def test_importable_typing_forms_do_not_capture_private_slot_state(
    tmp_path: Path, name: str
) -> None:
    import typing

    form = getattr(typing, name)
    snapshot = tmp_path / "typing-form.state"
    with InteractiveSession(runtime_globals={"saved_form": form}) as session:
        session.execute("~Value(item: Any)\n[value: Value] = `Value(42)`")
        session.dump(snapshot, format="python")
    with InteractiveSession.load(snapshot, trusted=True) as restored:
        assert restored._env["saved_form"] is form
        assert restored._env["value"].item == 42
