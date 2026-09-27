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
| Incremental reindex speedup | ~99x (1.72s full vs 0.017s incremental) | `scripts/benchmark.py`: 50 synthetic docs (100 chunks) reindexed from scratch, then again after editing 1 doc (1 modified, 1 deleted, 98 unchanged), local M-series laptop, real ONNX embed calls both times | 2026-09-27 |
| Semantic cache hit rate | 2/7 = 28.6% on a synthetic query set with 3 known near-duplicate pairs | Same script, `similarity_threshold=0.92` default: 7 queries, 3 of which are paraphrases of 2 earlier queries | 2026-09-27 |
| Test suite | 48 tests passing, ruff clean | `pytest -q && ruff check .` at commit time | 2026-09-27 |

The 28.6% hit rate is lower than the "3 duplicate pairs out of 7 queries" setup suggests, because `similarity_threshold=0.92` is strict by design — a looser threshold catches more paraphrases but risks serving an answer for a query that wasn't actually asking the same thing. Loosen it per-deployment if your queries are more uniformly phrased than this benchmark's.

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

**Works:** content-hash diffing (add/modify/delete detection), incremental
re-embedding, the numpy cosine-similarity store with disk persistence, the
semantic cache with hash-based invalidation (including the multi-chunk
dependency case — an entry depending on 2 chunks is invalidated if either
one changes), the FastAPI endpoints, and request validation at the API
boundary.

**Known simplifications:**
- `VectorStore` is a brute-force scan, not an ANN index — fine into the low
  thousands of chunks, not benchmarked beyond that.
- `SemanticCache` is in-memory only; it does not survive a process restart.
- `app.py`'s `_placeholder_answer` just concatenates retrieved chunk text —
  there's no LLM call wired in, by design, to keep this backend-only and
  network-independent. The real integration point is `RagSyncService.query`'s
  `answer_fn` parameter.
- `DirectorySource` only reads `.txt`/`.md` files from a local directory;
  a different `DocumentSource` implementation is a small amount of code for
  any other source (DB, API, object storage).

**Untested:** behavior under concurrent reindex + query calls (no locking is
implemented; this project assumes single-writer usage, like a periodic
reindex job).
