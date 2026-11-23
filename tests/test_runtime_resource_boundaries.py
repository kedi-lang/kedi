"""Regression coverage for runtime-owned sources, modules, and local types."""

from __future__ import annotations

import asyncio
import gc
import inspect
import linecache
import pickle
import sys
import weakref

import pytest

from kedi.agent_adapter import execution
from kedi.agent_adapter.execution import _run_cleanups
from kedi.core import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.compiler.type_system import register_type
from tests.mock_adapter import MockAdapter


def test_python_origin_filenames_do_not_collide_between_runtimes():
    first, second = KediRuntime(), KediRuntime()
    left = first.register_python_origin(kind="inline", code="1 / 0", span=None)
    right = second.register_python_origin(kind="inline", code="2 / 0", span=None)
    try:
        assert left != right
        assert linecache.getline(left, 1) == "1 / 0\n"
        assert linecache.getline(right, 1) == "2 / 0\n"
        asyncio.run(first.aclose())
        assert left not in linecache.cache
        assert linecache.getline(right, 1) == "2 / 0\n"
    finally:
        asyncio.run(first.aclose())
        asyncio.run(second.aclose())


def test_closing_a_restored_origin_map_does_not_remove_the_producers_source():
    producer, restored = KediRuntime(), KediRuntime()
    filename = producer.register_python_origin(kind="inline", code="1 / 0", span=None)
    # Executable snapshots restore logical origins, not physical cache ownership.
    restored._python_origins = dict(producer._python_origins)
    restored._python_origin_keys = dict(producer._python_origin_keys)
    try:
        asyncio.run(restored.aclose())
        assert linecache.getline(filename, 1) == "1 / 0\n"
    finally:
        asyncio.run(producer.aclose())


def test_local_type_fields_resolve_local_aliases_and_recursive_references():
    runtime = compile_program(
        parse_program(
            "@build() -> int:\n"
            "  [FieldType: Any] = `int`\n"
            "  ~Node(value: `FieldType`, children: list[Node])\n"
            "  = `Node(value=3, children=[Node(value=4, children=[])]).children[0].value`\n"
            "= <build()>\n"
        ),
        adapter=MockAdapter(),
    )
    try:
        assert runtime.run_main() == "4"
    finally:
        asyncio.run(runtime.aclose())


def test_type_field_resolution_uses_the_supplied_local_namespace():
    runtime = KediRuntime()
    type_env = {}
    typedef = parse_program("~Record(value: `FieldType`)\n").type_defs[0]
    try:
        register_type(runtime, typedef, type_env, {"FieldType": int})
        model = type_env["Record"]
        assert model(value=3).value == 3
        assert model.model_fields["value"].annotation is int
    finally:
        asyncio.run(runtime.aclose())


def test_synthetic_type_module_does_not_keep_a_retired_type_alive():
    runtime = compile_program(
        parse_program("~Record(value: int)\n= `Record`\n"), adapter=MockAdapter()
    )
    model = runtime.run_main()
    module_name = model.__module__
    module = sys.modules[module_name]
    source_names = (module.__file__, model.__init__.__code__.co_filename)
    reference = weakref.ref(model)
    asyncio.run(runtime.aclose())
    # Returned native types keep introspection while their caller still owns them.
    assert "class Record" in inspect.getsource(model)
    assert getattr(module, "Record") is model
    assert pickle.loads(pickle.dumps(model)) is model
    assert isinstance(pickle.loads(pickle.dumps(model(value=3))), model)
    with pytest.raises(AttributeError):
        getattr(module, "unrelated_attribute")
    del model, runtime
    gc.collect()
    assert reference() is None
    assert module_name not in sys.modules
    assert all(name not in linecache.cache for name in source_names)
    with pytest.raises(AttributeError):
        getattr(module, "Record")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("cleanup failed"), asyncio.CancelledError()])
async def test_all_deferred_cleanups_are_attempted_before_the_first_error_is_raised(failure):
    called = []

    async def fail():
        called.append("first")
        raise failure

    async def finish():
        called.append("second")

    with pytest.raises(type(failure)) as raised:
        await _run_cleanups([fail, finish])
    assert raised.value is failure
    assert called == ["first", "second"]


@pytest.mark.asyncio
async def test_host_loop_cleanup_cannot_block_the_thread_that_owns_the_loop(monkeypatch):
    host_loops = execution._HostLoops()

    def blocked_submit(coroutine, loop):
        coroutine.close()
        raise AssertionError("cleanup tried to wait on the blocked host loop")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", blocked_submit)
    with pytest.raises(RuntimeError, match="asyncio.to_thread"):
        with host_loops.use(asyncio.get_running_loop()):
            assert host_loops.defer(1, lambda: asyncio.sleep(0))
    assert not host_loops._users
    assert not host_loops._cleanups


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_owner", [None, "main", "dependency"])
async def test_runtime_close_releases_imported_resources_after_a_cleanup_failure(
    tmp_path, monkeypatch, failed_owner
):
    dependency = tmp_path / "dependency.kedi"
    dependency.write_text("@value() -> str:\n  = imported\n> export:\n  value\n")
    root = tmp_path / "main.kedi"
    runtime = compile_program(
        parse_program("> import: dependency\n= <value()>\n", source_path=str(root)),
        adapter=MockAdapter(),
    )
    assert runtime.run_main() == "imported"
    runtimes = list(dict.fromkeys(runtime._module_runtimes.values()))
    assert len(runtimes) == 2
    runtime._module_runtimes["dependency_alias"] = runtimes[-1]
    called = []
    failure = RuntimeError("coordinator cleanup failed")
    for owned in runtimes:
        original = owned._subagent_coordinator.aclose

        async def close(owned=owned, original=original):
            called.append(owned.module_name)
            await original()
            if owned.module_name == failed_owner:
                raise failure

        monkeypatch.setattr(owned._subagent_coordinator, "aclose", close)
    if failed_owner is None:
        await runtime.aclose()
    else:
        with pytest.raises(RuntimeError) as raised:
            await runtime.aclose()
        assert raised.value is failure
    assert sorted(called) == sorted(owned.module_name for owned in runtimes)
    assert all(owned.artifacts._closed for owned in runtimes)
    assert all(owned._subagent_coordinator._closed for owned in runtimes)


@pytest.mark.asyncio
async def test_synchronous_and_asynchronous_cleanups_share_error_order():
    called = []
    failure = RuntimeError("first failure")

    def fail():
        called.append("sync")
        raise failure

    async def later():
        called.append("async")
        raise ValueError("later failure")

    with pytest.raises(RuntimeError) as raised:
        await _run_cleanups([fail, later, lambda: called.append("last")])
    assert raised.value is failure
    assert called == ["sync", "async", "last"]


@pytest.mark.asyncio
async def test_closing_an_imported_runtime_does_not_close_its_importer(tmp_path):
    (tmp_path / "dependency.kedi").write_text(
        "@value() -> str:\n  = imported\n> export:\n  value\n"
    )
    runtime = compile_program(
        parse_program(
            "> import: dependency\n= <value()>\n", source_path=str(tmp_path / "main.kedi")
        ),
        adapter=MockAdapter(),
    )
    imported = next(owned for owned in runtime._module_runtimes.values() if owned is not runtime)
    try:
        await imported.aclose()
        assert imported.artifacts._closed
        assert not runtime.artifacts._closed
        assert not runtime._subagent_coordinator._closed
    finally:
        await runtime.aclose()
