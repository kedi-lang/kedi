# Artifact Storage and Lifetime

Context reduction, producer memory, persistent storage, and history are separate concerns. Close the owning session when finished.

## Explicit Streaming

Artifact admission protects model context even for ordinary materialized tool
results, but it cannot undo producer memory already allocated by user code. A
tool that constructs a 500 MB string has already paid for that allocation before
Kedi can measure it.

Tools that need bounded producer memory must opt into `ArtifactStream`:

```python
from collections.abc import Iterator

import kedi


def chunks() -> Iterator[str]:
    with open("application.log", encoding="utf-8") as stream:
        while chunk := stream.read(64 * 1024):
            yield chunk


@kedi.tool
def read_application_log() -> kedi.ArtifactStream[str]:
    return kedi.ArtifactStream.text(chunks())
```

`ArtifactStream.text`, `ArtifactStream.bytes`, and
`ArtifactStream.json_items` accept synchronous or asynchronous sources. Kedi
does not infer stream semantics from arbitrary iterables or generators; the
explicit wrapper makes single-use ownership and failure behavior unambiguous.

The stream pipeline is:

```text
producer chunks
  -> incremental canonical encoder
  -> threshold buffer
  -> transactional store writer after threshold crossing
  -> compact reference
```

Before the threshold is crossed, Kedi buffers only enough content to decide
whether the result stays inline. Once crossed, the buffer and later chunks are
written incrementally. Session quota is reserved and grown as chunks arrive.
Failure, cancellation, invalid chunk type, or quota rejection aborts the writer,
releases reservations, closes the source, and exposes no partial artifact.

If a stream completes below the threshold, Kedi reconstructs its declared
native value. If artifacts are disabled, a stream may stay inline only up to
the 100,000-byte hard tool-result limit.

The bundled `filesystem.read_text_file` and skill `read_skill` tools use
`ArtifactStream` automatically during adapter tool calls. Their direct
Kedi/Python call contract remains `str`; the incremental transport is an
internal execution detail and does not alter their schema. Small bounded tools
such as directory and artifact metadata listing remain materialized. Sandbox
and subagent results are artifact-admitted after completion because their
underlying engines do not expose incremental result chunks.


## Serialization and Stores

### Canonical serialization

Kedi supports four payload classes:

| Value | Codec | Media type | Persistence |
| --- | --- | --- | :---: |
| `str` | UTF-8 text | `text/plain; charset=utf-8` | yes |
| `bytes` | raw bytes | `application/octet-stream` | yes |
| JSON-compatible values, Pydantic models, dataclasses | canonical JSON | `application/json` | yes |
| Other Python objects | opaque process-local object | `application/x-python-object` | memory only |

Serializable mutable values are snapshotted when admitted. JSON encoding is
canonical and records bounded RFC 6901 pointer ranges so file-backed path reads
can avoid materializing the complete document. Opaque objects use a safe type
preview, have no content digest, and cannot enter the file store.

Kedi never uses pickle and never imports an arbitrary class while loading an
artifact.

### Memory store

The memory store preserves native Python values, including opaque objects. It
reduces model context, but it is not resident-memory offload: the process still
owns the full value until release, expiry, or manager close.

### File store

The file store persists supported text, bytes, and JSON projections and does
not retain an unbounded native-value cache. Reads and literal searches operate
on the persisted projection. Paths are confined to the configured root,
symlink escapes are rejected, and payload and metadata writes are atomic.

File records are restored only for their owning session. Expired records are
not revived. A value that cannot be represented safely by a supported codec
raises `ArtifactSerializationError` instead of falling back to unsafe
persistence.


## Quotas, Expiry, and Concurrency

Admission checks both active artifact count and active payload bytes. Streaming
writes reserve count and bytes transactionally, so concurrent producers cannot
individually pass a stale quota check and overcommit the session.

The fixed TTL is measured from creation and never moves. `idle_ttl` is measured
from the last successful leased access. An artifact expires at the earlier of
the two deadlines.

Reads acquire a lease. If release or expiry occurs during an active read, the
record enters `pending_release` or `expired`, the current reader may finish, and
the payload is deleted when the final lease closes. New reads fail immediately.

Artifact states are:

```text
active -> pending_release -> released
active ------------------> released
active ------------------> expired
pending_release ----------> expired
```

Cleanup is both lazy and background-driven. Runtime operations perform an
expiry check when the configured interval has elapsed, and one process-level
cleanup service tracks active managers. Kedi does not create one cleanup thread
per runtime.

Closing the manager unregisters cleanup, closes stores and code-runtime pools,
and releases in-process accounting. An application must not reuse a closed
session.


## History and Cache Stability

`ArtifactHistory` is a thread-safe recent lifecycle log with monotonically
increasing sequence numbers. It records tool calls, tool results, artifact
creation, bounded reads, release, and expiry. By default it retains the latest
4,096 events. Eviction drops only the oldest diagnostic events; sequence numbers
are never reused. `retained_events` and `dropped_events` expose the retained and
evicted counts. This log is not a complete audit archive.

Python applications can select a different retention limit:

```python
from kedi.artifacts import ArtifactHistory, ArtifactManager

manager = ArtifactManager(history=ArtifactHistory(max_events=8192))
try:
    recent_events = manager.history.snapshot()
    omitted_events = manager.history.dropped_events
finally:
    manager.close()
```

`max_events=None` explicitly opts into unbounded retention; the application then
owns its memory growth. Do not use it for indefinite sessions without an
external lifecycle limit. Artifact checkpoints preserve the retention limit and
sequence cursor. Legacy checkpoints with a plain event list remain readable.
`as_jsonable()` exports only recent events; use `export_checkpoint()` and
`ArtifactHistory.from_checkpoint()` when restoring sequence/retention state.

The diagnostic log and provider conversation are separate. Evicting a log event
does not release a payload or remove a provider message. Compaction reads live
and expired/released reference state from the artifact manager, not from a
possibly evicted creation event. Portable conversation history contains
references and bounded tool results, never the original large payload.

Within one cache epoch, artifact lifecycle operations never delete, reorder, or
rewrite earlier model messages. Releasing or expiring a payload therefore does
not invalidate an already cached provider prefix. The old reference remains in
history, but later attempts to dereference it receive the precise released or
expired error.

Provider-native checkpoints follow the same rule: release and expiry do not
mutate their existing prefix. Conversation compaction is a separate explicit
operation that starts a new cache epoch; artifact lifecycle does not perform
hidden history compaction.

Released/expired reference metadata remains session-owned so later lookups can
distinguish those states from unknown IDs and derived provenance remains useful.
Bounding the event log does not bound an unlimited number of created artifact
references or application-owned conversation messages; close sessions when done.

Kedi remains stateless by default. Use an explicit Python
[`session()`](../python-api/artifacts-and-sessions.md) when separate calls must
share model history and artifact ownership.
