"""Session-local executable graph checkpoints.

Only logical runtime fields cross this boundary.  Persistent references are
allocated before any user graph is read, so cycles may cross namespaces, lexical
owners, declarations, defaults and model instances.  No saved source is executed.
Cloudpickle is deliberately imported only by the executable snapshot entrypoints.
"""

from __future__ import annotations

import asyncio
import builtins
import contextvars
import io
import marshal
import multiprocessing.process
import pickle
import socket
import struct
import subprocess
import sys
import threading
import types
from concurrent.futures import Executor as ConcurrentExecutor, Future
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from ..core import KediRuntime
from ..decisions import DecisionReference
from ..executors.base import DefaultExecutor
from ..lang.compiler.environment import KediEnv, ScopeFrame
from ..lang.compiler.procedure_compiler import ProcedureCompiler
from ..lang.compiler.program_compiler import ProgramCompiler
from ..lang.procedure_introspection import get_procedure_def, get_procedure_metadata
from ..promise import KediPromise
from .persistence import SessionDumpError, SessionLoadError
from .resources import session_runtimes

_MAGIC = b"KEDI-PYTHON-GRAPH\x01"
# This is an allowlist, not a filtered runtime.__dict__: new live infrastructure
# must never accidentally become part of the executable graph.
_RUNTIME_FIELDS = (
    "_root_procedures",
    "_import_env",
    "_initial_globals_env",
    "_globals_env",
    "_prelude_env",
    "_root_type_env_stack",
    "_profile_registry",
    "default_agent_profile",
    "_program",
    "_source_map",
    "_source_text",
    "_python_origin_counter",
    "_python_origins",
    "_python_origin_keys",
    "_validation_tests",
    "_validation_evals",
    "conversation_state",
    "_implicit_conversation_state",
    "artifacts",
    "loop_iteration_limit",
)
_SESSION_FIELDS = (
    "_env",
    "_top_level_state",
    "_cell_number",
    "_source_names",
    "_fragments",
    "_persistence_blockers",
    "_conversation_state",
)
# These are produced afresh by the model metaclass/schema builder.  Everything
# else in the class's own namespace, including user methods and introspection
# descriptors, belongs to the saved graph rather than to Pydantic infrastructure.
_MODEL_REBUILT_ATTRIBUTES = frozenset(
    {
        "__dict__",
        "__weakref__",
        "_abc_impl",
        "__abstractmethods__",
        "__pydantic_core_schema__",
        "__pydantic_validator__",
        "__pydantic_serializer__",
        "__pydantic_complete__",
        "__pydantic_fields__",
        "__pydantic_setattr_handlers__",
        "__pydantic_parent_namespace__",
        "__pydantic_generic_metadata__",
        "__pydantic_decorators__",
        "__pydantic_computed_fields__",
        "__pydantic_custom_init__",
        "__pydantic_post_init__",
        "__init__",
        "__annotations__",
        "__module__",
        "__qualname__",
        "__doc__",
        "model_config",
    }
)
_LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))
_UNSUPPORTED = (
    io.IOBase,
    socket.socket,
    types.GeneratorType,
    types.CoroutineType,
    types.AsyncGeneratorType,
    types.FrameType,
    types.TracebackType,
    threading.Thread,
    *_LOCK_TYPES,
    threading.Event,
    threading.Condition,
    threading.Semaphore,
    threading.Barrier,
    Future,
    ConcurrentExecutor,
    multiprocessing.process.BaseProcess,
    contextvars.Context,
    contextvars.ContextVar,
    asyncio.Future,
    asyncio.AbstractEventLoop,
    subprocess.Popen,
    type(threading.local()),
)


def _cloudpickle() -> Any:
    try:
        import cloudpickle
    except ImportError as exc:
        raise SessionDumpError(
            ["Python snapshots require the optional dependency: install kedi[snapshots]"]
        ) from exc
    return cloudpickle


def _closure_values(fn: types.FunctionType) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            result[name] = cell.cell_contents
        except ValueError:
            pass
    return result


def _model_init(model: Any) -> types.FunctionType | None:
    """Recognize generated models by their constructor's actual closure graph."""
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        return None
    init = model.__dict__.get("__init__")
    if not isinstance(init, types.FunctionType):
        return None
    body = init.__globals__.get("__kedi_model_init__")
    generated = (
        isinstance(body, types.FunctionType)
        and _closure_values(body).get("model_class") is model
        and "__kedi_model_init__" in init.__code__.co_names
    )
    return init if generated else None


def _procedure_owner(fn: Any) -> tuple[KediRuntime, Any] | None:
    if not isinstance(fn, types.FunctionType) or get_procedure_def(fn) is None:
        return None
    validated = getattr(fn, "__kedi_validated_procedure_body__", None)
    body = fn.__globals__.get("_body")
    captured = _closure_values(validated) if isinstance(validated, types.FunctionType) else {}
    compiler = captured.get("self")
    if not isinstance(compiler, ProcedureCompiler):
        return None
    compiled = (
        isinstance(body, types.FunctionType)
        and "_body" in fn.__code__.co_names
        and _closure_values(body).get("_validated_body") is validated
        and captured.get("proc") is get_procedure_def(fn)
    )
    return (compiler.rt, captured.get("parent_env")) if compiled else None


def _empty_cell() -> types.CellType:
    return types.CellType()


def _function_state(fn: types.FunctionType) -> dict[str, Any]:
    return {
        "defaults": fn.__defaults__,
        "kwdefaults": fn.__kwdefaults__,
        "annotations": fn.__annotations__,
        "attributes": fn.__dict__,
        "name": fn.__name__,
        "qualname": fn.__qualname__,
        "module": fn.__module__,
        "doc": fn.__doc__,
        "type_params": getattr(fn, "__type_params__", ()),
    }


def _install_function(fn: types.FunctionType, state: dict[str, Any]) -> None:
    fn.__defaults__ = state["defaults"]
    fn.__kwdefaults__ = state["kwdefaults"]
    fn.__annotations__ = state["annotations"]
    fn.__dict__ = state["attributes"]
    fn.__name__ = state["name"]
    fn.__qualname__ = state["qualname"]
    fn.__module__ = state["module"]
    fn.__doc__ = state["doc"]
    if hasattr(fn, "__type_params__"):  # pragma: no branch - attribute exists from Python 3.12
        setattr(fn, "__type_params__", state["type_params"])


class _Inventory:
    def __init__(self, session: Any, resource_ids: dict[int, str], cloud: Any) -> None:
        self.session = session
        self.resource_ids = resource_ids
        self.by_reference = cloud.cloudpickle._should_pickle_by_reference
        self.extract_code_globals = cloud.cloudpickle._extract_code_globals
        self.validated_owners: dict[int, Any] = {}
        self.objects: list[Any] = []
        self.descriptors: list[dict[str, Any]] = []
        self.records: dict[int, dict[str, Any]] = {}
        self.ids: dict[int, int] = {}
        self.runtime_ids: dict[int, int] = {}
        self.infrastructure: dict[int, tuple[str, int]] = {}
        self.runtimes = session_runtimes(session)
        issues = list(session._persistence_blockers)
        if session._subagent_state_path is not None:
            issues.append("session owns external persistent subagent state")
        for index, rt in enumerate(self.runtimes):
            self.runtime_ids[id(rt)] = index
            if type(rt.executor) is not DefaultExecutor:
                issues.append(
                    f"runtime {index}: Python snapshots require the default local executor; "
                    f"{type(rt.executor).__qualname__} has no supported state transfer contract"
                )
            if any(rt._outstanding.values()):
                issues.append(f"runtime {index} has unfinished model work")
            coordinator = rt._subagent_coordinator
            if (
                coordinator.active_runs
                or coordinator._retained_native_runs
                or any(not owned.task.done() for owned in coordinator._tasks.values())
            ):
                issues.append(f"runtime {index} has unfinished subagent work")
            if rt._lifecycle._owners:
                issues.append(f"runtime {index} has an active invocation")
            if rt._lifecycle._closed or coordinator._closed:
                issues.append(f"runtime {index} is closed")
            for label, value in (
                ("executor", rt.executor),
                ("adapter", rt.adapter),
                ("engine", rt._engine),
            ):
                if value is not None:
                    self.infrastructure.setdefault(id(value), (label, index))
        if session._compiler.module_loader._loading:
            issues.append("module initialization is incomplete")
        if issues:
            raise SessionDumpError(issues)
        for rt in self.runtimes:
            if isinstance(rt._prelude_env, dict):
                self.register(rt._prelude_env, "namespace")

    def register(self, obj: Any, kind: str) -> int:
        known = self.ids.get(id(obj))
        if known is not None:
            return known
        index = len(self.objects)
        self.ids[id(obj)] = index
        self.objects.append(obj)
        descriptor: dict[str, Any] = {"kind": kind}
        self.descriptors.append(descriptor)
        if kind == "function":
            descriptor["code"] = marshal.dumps(obj.__code__)
            descriptor["globals"] = self.register(obj.__globals__, "namespace")
            descriptor["cells"] = [self.register(cell, "cell") for cell in obj.__closure__ or ()]
            if obj.__builtins__ is not builtins.__dict__:
                descriptor["builtins"] = self.register(obj.__builtins__, "namespace")
        elif kind == "model":
            descriptor["name"] = obj.__name__
            self.register(_model_init(obj), "function")
        elif kind == "procedure":
            validated = obj.__kedi_validated_procedure_body__
            self.validated_owners[id(validated)] = obj
            self.register(validated, "validated_body")
        elif kind == "validated_body":
            descriptor["code"] = marshal.dumps(obj.__code__)
        elif kind == "namespace":
            descriptor["namespace_type"] = (
                "kedi"
                if isinstance(obj, KediEnv)
                else "scope"
                if isinstance(obj, ScopeFrame)
                else "dict"
            )
        return index

    def persistent_id(self, obj: Any) -> Any:
        if obj is self.session:
            raise SessionDumpError(["user graph references the live interactive session"])
        token = self.resource_ids.get(id(obj))
        if token is not None:
            return ("resource", token)
        rt_id = self.runtime_ids.get(id(obj))
        if rt_id is not None:
            return ("runtime", rt_id)
        infrastructure = self.infrastructure.get(id(obj))
        if infrastructure is not None:
            return ("infrastructure", *infrastructure)
        known = self.ids.get(id(obj))
        if known is not None:
            return ("object", known)
        if isinstance(obj, _UNSUPPORTED):
            raise SessionDumpError(
                [
                    f"unsupported live Python resource: {type(obj).__module__}.{type(obj).__qualname__}"
                ]
            )
        if isinstance(obj, KediRuntime):
            raise SessionDumpError(["user graph references a runtime not owned by this session"])
        if isinstance(obj, (KediEnv, ScopeFrame)):
            return ("object", self.register(obj, "namespace"))
        if isinstance(obj, types.CellType):
            return ("object", self.register(obj, "cell"))
        if _model_init(obj) is not None:
            return ("object", self.register(obj, "model"))
        owner = _procedure_owner(obj)
        if owner is not None:
            if id(owner[0]) not in self.runtime_ids:
                raise SessionDumpError(["procedure belongs to a runtime outside this session"])
            return ("object", self.register(obj, "procedure"))
        if isinstance(obj, types.FunctionType):
            # Python-facing procedure wrappers and generated constructors must keep
            # their exact globals, not cloudpickle's copied subset of that mapping.
            if (
                isinstance(obj.__globals__, (KediEnv, ScopeFrame))
                or id(obj.__globals__) in self.ids
                or getattr(obj, "__kedi_procedure_target__", None) is not None
                or self._owns_function_namespace(obj)
            ):
                return ("object", self.register(obj, "function"))
        return None

    def _owns_function_namespace(self, fn: types.FunctionType) -> bool:
        # An exec-created function may also be exported by an importable module.
        # That alias does not make its separate globals an import-owned namespace.
        namespace = fn.__globals__
        name = namespace.get("__name__")
        module = sys.modules.get(name) if isinstance(name, str) else None
        return module is None or vars(module) is not namespace

    def discover(self, root: Any) -> None:
        """Inventory structural references before any ordinary dict is memoized.

        Custom reducers remain trusted serializer code.  This pass only walks
        stored Python state; it neither invokes reducers nor evaluates source.
        """
        from functools import partial

        from ..agent_profile.capture import _CaptureCell

        pending = [root]
        seen: set[int] = set()
        while pending:
            obj = pending.pop()
            identity = id(obj)
            if identity in seen:
                continue
            seen.add(identity)
            if (
                identity in self.resource_ids
                or identity in self.runtime_ids
                or identity in self.infrastructure
            ):
                continue
            pid = self.persistent_id(obj)
            if pid is not None and pid[0] == "object":
                index = pid[1]
                pending.extend(self.record(index).values())
                descriptor = self.descriptors[index]
                if descriptor["kind"] == "function":
                    pending.append(self.objects[descriptor["globals"]])
                    pending.extend(self.objects[cell] for cell in descriptor["cells"])
                    if "builtins" in descriptor:
                        pending.append(self.objects[descriptor["builtins"]])
                continue
            if isinstance(obj, dict):
                pending.extend(dict.keys(obj))
                pending.extend(dict.values(obj))
            elif isinstance(obj, (list, tuple, set, frozenset)):
                pending.extend(obj)
            elif isinstance(obj, types.ModuleType):
                continue
            elif isinstance(obj, types.FunctionType):
                if not self.by_reference(obj):
                    pending.extend(
                        (
                            obj.__dict__,
                            obj.__defaults__,
                            obj.__kwdefaults__,
                            obj.__annotations__,
                            obj.__closure__,
                        )
                    )
                    pending.extend(
                        obj.__globals__[name]
                        for name in self.extract_code_globals(obj.__code__)
                        if name in obj.__globals__
                    )
            elif isinstance(obj, types.MethodType):
                pending.extend((obj.__func__, obj.__self__))
            elif isinstance(obj, (staticmethod, classmethod)):
                pending.append(obj.__func__)
            elif isinstance(obj, property):
                pending.extend((obj.fget, obj.fset, obj.fdel))
            elif isinstance(obj, type):
                if not self.by_reference(obj):
                    pending.extend(vars(obj).values())
                    pending.extend(obj.__bases__)
            elif isinstance(obj, KediPromise):
                if not obj._future.done():
                    raise SessionDumpError(["Python graph contains an unfinished Kedi promise"])
                pending.append(obj.resolve())
            elif isinstance(obj, DecisionReference):
                if not obj.record_future.done():
                    raise SessionDumpError(["binding has unfinished decision evidence"])
                pending.append(obj.record_future.result())
            elif isinstance(obj, _CaptureCell):
                if obj.resolving or obj._evaluating:
                    raise SessionDumpError(["definition-site profile capture is being evaluated"])
                pending.extend((obj.value, obj._patch))
            elif isinstance(obj, partial):
                pending.extend((obj.func, obj.args, obj.keywords, obj.__dict__))
            else:
                if type(obj) in (str, bytes, int, float, complex, bool, type(None)):
                    continue
                # Reducers also encounter the instance's class and slot state.
                # Discover their functions before any aliased globals dictionary
                # can be memoized as an ordinary value.
                pending.append(type(obj))
                # Slot storage is only default-reducer state.  Custom reducers
                # can serialize an import reference instead; crawling their
                # private slots would capture implementation-only closures.
                if (
                    type(obj).__reduce__ is object.__reduce__
                    and type(obj).__reduce_ex__ is object.__reduce_ex__
                ):
                    for cls in type(obj).__mro__:
                        for descriptor in vars(cls).values():
                            if isinstance(descriptor, types.MemberDescriptorType):
                                try:
                                    pending.append(descriptor.__get__(obj, type(obj)))
                                except AttributeError:
                                    pass
                try:
                    pending.append(object.__getattribute__(obj, "__dict__"))
                except (AttributeError, TypeError):
                    pass

    def record(self, index: int) -> dict[str, Any]:
        if index not in self.records:
            self.records[index] = self._make_record(index)
        return self.records[index]

    def _make_record(self, index: int) -> dict[str, Any]:
        obj = self.objects[index]
        kind = self.descriptors[index]["kind"]
        if kind == "namespace":
            attributes = vars(obj) if isinstance(obj, (KediEnv, ScopeFrame)) else {}
            return {"items": list(dict.items(obj)), "attributes": attributes}
        if kind == "cell":
            try:
                return {"populated": True, "value": obj.cell_contents}
            except ValueError:
                return {"populated": False}
        if kind == "validated_body":
            return {"procedure": self.validated_owners[id(obj)]}
        if kind == "function":
            return _function_state(obj)
        if kind == "procedure":
            # Registration as a procedure already proved both invariants.
            procedure_owner = _procedure_owner(obj)
            metadata = get_procedure_metadata(obj)
            assert procedure_owner is not None and metadata is not None
            owner, parent = procedure_owner
            defaults = {
                name: value
                for name, source, value in zip(
                    metadata.params, metadata.param_defaults, metadata.default_values, strict=True
                )
                if source is not None
            }
            attributes = vars(obj)
            return {
                "runtime": owner,
                "parent": parent,
                "definition": get_procedure_def(obj),
                "defaults": defaults,
                "attributes": attributes,
                "function_state": {**_function_state(obj), "attributes": attributes},
            }
        assert kind == "model"
        return {
            "fields": obj.model_fields,
            "config": obj.model_config,
            "annotations": obj.__annotations__,
            "init": obj.__dict__["__init__"],
            "name": obj.__name__,
            "qualname": obj.__qualname__,
            "module": obj.__module__,
            "doc": obj.__doc__,
            "attributes": {
                name: value
                for name, value in vars(obj).items()
                if name not in _MODEL_REBUILT_ATTRIBUTES
            },
        }


def _capture_cell_new() -> Any:
    from ..agent_profile.capture import _CaptureCell

    return _CaptureCell()


def _restore_object_attributes(obj: Any, attributes: dict[str, Any]) -> None:
    obj.__dict__.update(attributes)


def _new_resolved_promise() -> KediPromise:
    return KediPromise(Future(), None)


def _restore_resolved_promise(promise: KediPromise, state: dict[str, Any]) -> None:
    value = state["value"]
    promise._future.set_result(value)
    promise._value = value
    promise._resolved = True


def _completed_decision_reference(record: Any, field_path: tuple[str, ...]) -> DecisionReference:
    future: Future[Any] = Future()
    future.set_result(record)
    return DecisionReference(future, field_path)


def _captured_procedure_reference(runtime: KediRuntime, target: Any) -> Any:
    def invoke(*args: Any) -> Any:
        return runtime._invoke_procedure(target, list(args))

    return invoke


class _SavedProcedureCompiler(ProcedureCompiler):
    def __init__(self, rt: KediRuntime, defaults: dict[str, Any]) -> None:
        super().__init__(rt)
        self._saved_defaults = defaults

    def _evaluate_param_defaults(self, proc: Any, parent_env: Any) -> dict[str, Any]:
        return self._saved_defaults


def _memory_artifact_roots(inventory: _Inventory) -> list[tuple[str, str, Any]]:
    from ..artifacts.models import ArtifactStatus

    managers: dict[int, Any] = {}
    for runtime in inventory.runtimes:
        managers[id(runtime.artifacts)] = runtime.artifacts
        for conversation in (runtime.conversation_state, runtime._implicit_conversation_state):
            if conversation is not None:
                managers[id(conversation.artifacts)] = conversation.artifacts
    conversation = inventory.session._conversation_state
    if conversation is not None:
        managers[id(conversation.artifacts)] = conversation.artifacts
    roots: list[tuple[str, str, Any]] = []
    retained = {ArtifactStatus.ACTIVE, ArtifactStatus.PENDING_RELEASE}
    for identity, manager in managers.items():
        token = inventory.resource_ids[identity]
        with manager._lock:
            for record in manager._memory_store.records(manager.session_id):
                if record.payload is None and (
                    record.status not in retained or record.codec != "json"
                ):
                    continue
                roots.append((token, record.ref_id, record.payload))
    return roots


def dumps_session(session: Any, resource_ids: dict[int, str]) -> tuple[bytes, frozenset[str]]:
    """Capture one shared logical graph, replacing infrastructure by semantic IDs.

    Also returns the modules whose objects the graph imports by reference, so
    the bundle pins exactly the distributions that loading depends on.
    """
    cloud = _cloudpickle()
    from ..agent_profile.capture import _CaptureCell

    inventory = _Inventory(session, resource_ids, cloud)
    modules: set[str] = set()

    class SessionPickler(cloud.CloudPickler):
        def persistent_id(self, obj: Any) -> Any:
            return inventory.persistent_id(obj)

        def _function_getnewargs(self, fn: types.FunctionType) -> Any:
            arguments = super()._function_getnewargs(fn)
            # Cloudpickle normally allocates separate cells for each function.
            # Persistent cells already exist before unpickling and are hydrated
            # afterward, preserving nonlocal writes across imported-module
            # closures without capturing that module's entire globals mapping.
            return (*arguments[:-1], fn.__closure__)

        def reducer_override(self, obj: Any) -> Any:
            if isinstance(obj, types.ModuleType):
                modules.add(obj.__name__)
            elif isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType)):
                module = getattr(obj, "__module__", None)
                if isinstance(module, str):
                    modules.add(module)
            if isinstance(obj, KediPromise):
                if not obj._future.done():
                    raise SessionDumpError(["Python graph contains an unfinished Kedi promise"])
                return (
                    _new_resolved_promise,
                    (),
                    {"value": obj.resolve()},
                    None,
                    None,
                    _restore_resolved_promise,
                )
            if isinstance(obj, DecisionReference):
                if not obj.record_future.done():
                    raise SessionDumpError(["binding has unfinished decision evidence"])
                return _completed_decision_reference, (
                    obj.record_future.result(),
                    obj.field_path,
                )
            if isinstance(obj, types.FunctionType):
                captured = _closure_values(obj)
                definition = captured.get("definition")
                bindings = captured.get("bindings")
                runtime = captured.get("rt")
                target = (
                    bindings.get(id(definition))
                    if isinstance(bindings, dict) and isinstance(runtime, KediRuntime)
                    else None
                )
                if target is not None and get_procedure_def(target) is definition:
                    # capture_procedure_profiles closes over an id-keyed map.
                    # Persist its semantic target, never stale producer IDs.
                    return _captured_procedure_reference, (runtime, target)
            if isinstance(obj, _CaptureCell):
                if obj.resolving or obj._evaluating:
                    raise SessionDumpError(["definition-site profile capture is being evaluated"])
                return (
                    _capture_cell_new,
                    (),
                    {"value": obj.value, "_patch": obj._patch},
                    None,
                    None,
                    _restore_object_attributes,
                )
            return super().reducer_override(obj)

    root = {
        "session": {name: getattr(session, name) for name in _SESSION_FIELDS},
        "runtimes": [
            {name: getattr(rt, name) for name in _RUNTIME_FIELDS} for rt in inventory.runtimes
        ],
        "module_programs": session._compiler.module_loader._programs,
        "modules": session._compiler.module_loader._loaded,
        "memory_artifacts": _memory_artifact_roots(inventory),
    }
    stream = io.BytesIO()
    try:
        inventory.discover(root)
        pickler = SessionPickler(stream, protocol=5)
        pickler.dump(root)
        index = 0
        while index < len(inventory.objects):
            pickler.dump((index, inventory.record(index)))
            index += 1
        header = pickle.dumps(
            {
                "version": 1,
                "runtime_count": len(inventory.runtimes),
                "objects": inventory.descriptors,
            },
            protocol=5,
        )
    except SessionDumpError:
        raise
    except Exception as exc:
        raise SessionDumpError(
            [f"Python graph capture failed ({type(exc).__name__}): {exc}"]
        ) from exc
    return _MAGIC + struct.pack(">Q", len(header)) + header + stream.getvalue(), frozenset(modules)


def _new_module_runtime(session: Any) -> KediRuntime:
    root = session._compiler
    # Construct infrastructure only.  In particular, do not compile the saved
    # program or call _ModuleLoader.load(), either of which runs initialization.
    compiler = ProgramCompiler(
        root.adapter,
        executor=root.rt.executor,
        engine=root.engine,
        module_loader=root.module_loader,
        subagent_max_depth=session._subagent_max_depth,
        subagent_max_concurrency=session._subagent_max_concurrency,
        subagent_timeout_seconds=session._subagent_timeout_seconds,
        subagent_usage_limits=session._subagent_usage_limits,
        conversation_state=session._conversation_state,
        loop_iteration_limit=session._loop_iteration_limit,
    )
    return compiler.rt


def loads_session(session: Any, payload: bytes, resources: dict[str, object]) -> None:
    """Hydrate fresh infrastructure without executing saved fragments or imports.

    The caller has already authenticated the bundle's compatibility and explicit
    trust acknowledgement.  This function is executable deserialization, not an
    independent trust boundary.
    """
    if type(session._compiler.rt.executor) is not DefaultExecutor:
        raise SessionLoadError("Python snapshots require the default local executor")
    try:
        _cloudpickle()
    except SessionDumpError as exc:
        raise SessionLoadError(
            "Python snapshots require the optional dependency: install kedi[snapshots]"
        ) from exc
    try:
        _loads_session(session, payload, resources)
    except SessionLoadError:
        raise
    except Exception as exc:
        raise SessionLoadError(
            f"Python graph hydration failed ({type(exc).__name__}): {exc}"
        ) from exc


def _loads_session(session: Any, payload: bytes, resources: dict[str, object]) -> None:
    if not payload.startswith(_MAGIC):
        raise SessionLoadError("Invalid Python session graph signature")
    start = len(_MAGIC)
    header_length = struct.unpack(">Q", payload[start : start + 8])[0]
    start += 8
    if header_length > len(payload) - start:
        raise SessionLoadError("Truncated Python session graph inventory")
    header = pickle.loads(payload[start : start + header_length])
    if header["version"] != 1 or header["runtime_count"] < 1:
        raise SessionLoadError("Unsupported Python session graph inventory")
    runtimes = [session._compiler.rt]
    runtimes.extend(_new_module_runtime(session) for _ in range(header["runtime_count"] - 1))
    descriptors = header["objects"]
    objects: list[Any] = [None] * len(descriptors)
    # Namespaces/cells/models exist before functions, and every function exists
    # before values.  Neither instance constructors nor validators run here.
    for index, descriptor in enumerate(descriptors):
        kind = descriptor["kind"]
        if kind == "namespace":
            cls = {"kedi": KediEnv, "scope": ScopeFrame, "dict": dict}[descriptor["namespace_type"]]
            objects[index] = cls()
        elif kind == "cell":
            objects[index] = _empty_cell()
        elif kind == "model":
            objects[index] = create_model(
                descriptor["name"],
                __base__=BaseModel,
                __config__=ConfigDict(strict=True, extra="forbid"),
            )
        elif kind == "procedure":
            objects[index] = types.FunctionType((lambda: None).__code__, {})
        elif kind != "function":
            if kind == "validated_body":
                code = marshal.loads(descriptor["code"])
                cells = tuple(_empty_cell() for _ in code.co_freevars)
                objects[index] = types.FunctionType(
                    code,
                    ProcedureCompiler._make_procedure_impl.__globals__,
                    code.co_name,
                    None,
                    cells or None,
                )
                continue
            raise SessionLoadError(f"Unknown Python graph object kind {kind!r}")
    for index, descriptor in enumerate(descriptors):
        if descriptor["kind"] == "function":
            namespace = objects[descriptor["globals"]]
            # A function captures f_builtins at construction, not when called.
            function_builtins = (
                objects[descriptor["builtins"]] if "builtins" in descriptor else builtins.__dict__
            )
            dict.__setitem__(namespace, "__builtins__", function_builtins)
            code = marshal.loads(descriptor["code"])
            cells = tuple(objects[cell] for cell in descriptor["cells"])
            objects[index] = types.FunctionType(code, namespace, code.co_name, None, cells or None)

    class SessionUnpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> Any:
            kind, *parts = pid
            if kind == "object":
                return objects[parts[0]]
            if kind == "runtime":
                return runtimes[parts[0]]
            if kind == "resource":
                try:
                    return resources[parts[0]]
                except KeyError as exc:
                    raise SessionLoadError(f"Missing restored resource {parts[0]!r}") from exc
            if kind == "infrastructure":
                label, index = parts
                rt = runtimes[index]
                return {"executor": rt.executor, "adapter": rt.adapter, "engine": rt._engine}[label]
            raise SessionLoadError(f"Unknown Python graph persistent reference {kind!r}")

    stream = io.BytesIO(payload[start + header_length :])
    unpickler = SessionUnpickler(stream)
    root = unpickler.load()
    records: dict[int, dict[str, Any]] = {}
    for _ in descriptors:
        index, record = unpickler.load()
        if index in records or not 0 <= index < len(objects):
            raise SessionLoadError("Duplicate or invalid Python graph record")
        records[index] = record
    if stream.read(1):
        raise SessionLoadError("Unexpected trailing Python graph data")
    for index, descriptor in enumerate(descriptors):
        obj, record = objects[index], records[index]
        kind = descriptor["kind"]
        if kind == "namespace":
            dict.clear(obj)
            dict.update(obj, record["items"])
            if isinstance(obj, (KediEnv, ScopeFrame)):
                obj.__dict__ = record["attributes"]
        elif kind == "cell" and record["populated"]:
            obj.cell_contents = record["value"]
        elif kind == "function":
            _install_function(obj, record)
    for rt, state in zip(runtimes, root["runtimes"], strict=True):
        for name in _RUNTIME_FIELDS:
            setattr(rt, name, state[name])
    # Populate model shells in place, retaining class identity in every instance,
    # annotation, captured default and generated constructor closure.
    for index, descriptor in enumerate(descriptors):
        if descriptor["kind"] != "model":
            continue
        model, record = objects[index], records[index]
        model.__pydantic_fields__ = record["fields"]
        model.__annotations__ = record["annotations"]
        model.model_config = record["config"]
        model.__init__ = record["init"]
        model.__pydantic_custom_init__ = True
        model.__name__ = record["name"]
        model.__qualname__ = record["qualname"]
        model.__module__ = record["module"]
        model.__doc__ = record["doc"]
    for index, descriptor in enumerate(descriptors):
        if descriptor["kind"] == "model":
            objects[index].model_rebuild(force=True)
            for name, value in records[index]["attributes"].items():
                setattr(objects[index], name, value)
    for index, descriptor in enumerate(descriptors):
        if descriptor["kind"] != "procedure":
            continue
        shell, record = objects[index], records[index]
        proc = record["definition"]
        # Tool retry exception expressions were already evaluated in the producer.
        # Bypass only that declaration setup, then restore the exact saved AST and
        # retry policy.  Never evaluate defaults or annotation expressions here.
        from dataclasses import replace

        saved_proc = replace(proc, tool=None)
        compiler = _SavedProcedureCompiler(record["runtime"], record["defaults"])
        fn = compiler._make_procedure_impl(saved_proc, record["parent"])
        assert isinstance(fn, types.FunctionType)
        # The implementation must execute the original definition, not a detached
        # AST copy, so its definition-site profile and lexical owner remain shared.
        validated = getattr(fn, "__kedi_validated_procedure_body__")
        for name, cell in zip(validated.__code__.co_freevars, validated.__closure__ or ()):
            if name == "proc":
                cell.cell_contents = proc
        body = fn.__globals__["_body"]
        for name, cell in zip(body.__code__.co_freevars, body.__closure__ or ()):
            if name == "proc":
                cell.cell_contents = proc
        shell.__code__ = fn.__code__
        shell.__globals__.update(fn.__globals__)
        _install_function(shell, record["function_state"])
        shell.__kedi_validated_procedure_body__ = validated
    for index, descriptor in enumerate(descriptors):
        if descriptor["kind"] != "validated_body":
            continue
        shell = objects[index]
        procedure = records[index]["procedure"]
        validated = procedure.__kedi_validated_procedure_body__
        for destination, source in zip(
            shell.__closure__ or (), validated.__closure__ or (), strict=True
        ):
            destination.cell_contents = source.cell_contents
        shell.__dict__ = validated.__dict__
        procedure.__kedi_validated_procedure_body__ = shell
        body = procedure.__globals__["_body"]
        for name, cell in zip(body.__code__.co_freevars, body.__closure__ or ()):
            if name == "_validated_body":
                cell.cell_contents = shell
    for name, value in root["session"].items():
        setattr(session, name, value)
    session._snapshot_memory_artifacts = root["memory_artifacts"]
    loader = session._compiler.module_loader
    loader._programs = root["module_programs"]
    loader._loaded = root["modules"]
    loader._loading = []
    session._compiler.default_agent_profile = runtimes[0].default_agent_profile
    session._compiler.conversation_state = session._conversation_state
    assert session._env is runtimes[0]._globals_env
