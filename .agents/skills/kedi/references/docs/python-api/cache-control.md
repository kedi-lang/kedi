# Python Cache Control

These helpers inspect process-local parse and response caches only. They do not control provider prompt caching.

## Inspect Cache State

```python
info = kedi.cache_info()
print(info.parse_entries)
print(info.response_entries)
```

`CacheInfo` is a frozen dataclass with counts for the process-memory parse and
response caches. It does not report codegen, optimized prompt, GEPA checkpoint,
or adapter-provider caches.


## Clear Caches

```python
kedi.clear_cache()
```

This clears parsed programs and completed response entries. It also advances a
cache generation: a request already in flight may finish for its current
callers, but it cannot repopulate the newly cleared cache.

Response caching is opt-in per `query` or `bind` with `cache=True`. Parse
caching is always source-hash based. Concurrent identical response misses
coalesce; failed calls are never stored. A recursive same-thread request for
the same cache key raises instead of deadlocking.
