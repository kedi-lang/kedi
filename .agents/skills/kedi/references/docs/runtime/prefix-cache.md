# Provider Prefix Caching

Provider prompt caching reuses computation for a matching request prefix. It
does not skip a model invocation and is independent of Kedi's response cache.

## What Kedi Preserves

Within a compatible conversation lane, request assembly retains the order of
successful native messages and their tool-call/result relationships. Artifact
tools are present from the first artifact-enabled request. Releasing a payload
does not remove its old reference from history.

Changing a tool schema, model, settings, instructions, or enabled capabilities
can change the request prefix or continuation lane. Compaction establishes a
new cache epoch. These are not equivalent to appending a user message to an
otherwise identical request.

A Python `history_processor=` callback or native `> history:` `processor` that
returns semantically equivalent history is
a no-op: Kedi preserves the native message objects, continuation state, and
cache identity. An accepted removal, reordering, redaction, or summary changes
the prefix. Kedi then disables stale response-ID continuation for the affected
request and stages a new cache generation for each actual rewrite. Later tool
steps reuse that generation until another edit occurs; a failed or cancelled
turn does not commit it. Old server conversation IDs stay disabled on subsequent
turns in the same lane. Automatic response-ID continuation can use a new response
created after the edit. See [History](history.md#user-defined-history-processing).

`processor_condition` can gate the callback on a history token estimate or
another read-only state field. A false result avoids the processor's detached
copy and rewrite path; it does not rotate the cache epoch. A true result is
still subject to the same no-op/rewrite rules. A condition may reduce how often
processing runs, but cannot promise prefix preservation or a provider cache hit.
Python configuration emits one warning when a processor is installed; the LSP
warns at each active processor declaration, including external callbacks.

## What Kedi Cannot Guarantee

An unchanged prefix is necessary for reuse, not a promise of a cache hit.
Provider routing, minimum eligible prefix lengths, retention, and provider
implementation determine whether computation is reused. A cache key is a
routing hint, not a cache reservation. HTTP and WebSocket are transports;
neither guarantees a particular cache-read percentage.

## Measure Requests, Not Just Runs

Compare the same model, request sequence, tools, settings, and input sizes.
Record per-request input tokens and cached input tokens. The aggregate read
ratio is total cached input divided by total input, not an unweighted average
of request percentages. Include the cold first request explicitly, or label a
warm-only measurement as such.

A high ratio alone does not establish efficiency: repeatedly sending an
unnecessary large context can increase both cached tokens and total work.
Report uncached input, cached input, output, request count, task success, and
latency separately. Missing cache counters are unknown, not measured zero.

See [History](history.md), [Compaction](compaction.md), and [Application
Caches](caching.md) for the separate ownership and lifecycle rules.
