# ragsync

Incremental reindexing and a staleness-aware semantic query cache for RAG
pipelines. Two problems most RAG stacks punt on:

1. Reindexing a corpus normally means re-embedding everything, every time.
   ragsync hashes each chunk's content and only re-embeds what actually
   changed.
2. Semantic query caches (cache an answer, serve it again for a
   near-duplicate query) normally go stale silently — nothing tells the
   cache that the chunks its answer was grounded in have since changed.
   ragsync ties every cache entry to the exact chunk hashes it depended on,
   so a corpus change invalidates only the cache entries actually affected
   by it, not the whole cache and not nothing.

## Quickstart

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

```python
from ragsync.service import RagSyncService
from ragsync.document_source import DirectorySource

service = RagSyncService(store_path="./data")
report = service.reindex(DirectorySource("./docs"))
print(report.added, report.modified, report.deleted, report.unchanged)

result = service.query("what does the config module do?", answer_fn=my_llm_call)
print(result.answer, result.cache_hit)
```

## Measured vs Claimed

Every number below states what it measures and how. Update this table
instead of writing new numbers straight into prose elsewhere (portfolio site,
resume) — keep one source of truth and link to it.

| Claim | Value | How measured | Date |
|---|---|---|---|
| _e.g. reindex speedup_ | _pending_ | _pending_ | _pending_ |
| _e.g. cache hit rate_ | _pending_ | _pending_ | _pending_ |

## Architecture

- `hashing.py` — stable content hash for a chunk (text + source id).
- `document_source.py` — pluggable source of chunks (`DirectorySource` reads
  a directory of text/markdown files and splits them).
- `embeddings.py` — all-MiniLM-L6-v2 via onnxruntime + tokenizers directly
  (no chromadb import chain — see module docstring for why).
- `vectorstore.py` — a small persisted cosine-similarity index (numpy +
  JSON), not a full vector DB.
- `indexer.py` — diffs a document source's chunk hashes against the last
  indexed state and only re-embeds what changed.
- `cache.py` — semantic query cache; each entry records which chunk hashes
  its answer depended on, so a changed chunk invalidates precisely the
  entries that depended on it.
- `service.py` — ties the above together into one importable API.
- `app.py` — thin FastAPI wrapper over `service.py`.

## Status

_What works, what's known-broken, what's untested._
