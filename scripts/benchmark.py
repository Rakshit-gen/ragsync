"""Measures two real numbers for the README's Measured vs Claimed table:

1. Incremental reindex speedup: time to reindex N docs from scratch vs time
   to reindex again after changing 1 of them.
2. Semantic cache hit rate on a synthetic query set with known near-duplicates.

Run from the repo root: `python3 scripts/benchmark.py`
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragsync.document_source import DirectorySource
from ragsync.service import RagSyncService

N_DOCS = 50


def answer_fn(query: str, chunks: list[dict]) -> str:
    return " ".join(c.get("text", "") for c in chunks)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="ragsync-bench-"))
    docs_dir = workdir / "docs"
    docs_dir.mkdir()
    for i in range(N_DOCS):
        (docs_dir / f"doc{i}.txt").write_text(f"Document {i} discusses topic number {i} in detail. " * 20)

    service = RagSyncService(str(workdir / "store"))

    start = time.monotonic()
    full_report = service.reindex(DirectorySource(str(docs_dir)))
    full_duration = time.monotonic() - start

    (docs_dir / "doc0.txt").write_text("Document 0 now discusses something completely different.")

    start = time.monotonic()
    incremental_report = service.reindex(DirectorySource(str(docs_dir)))
    incremental_duration = time.monotonic() - start

    print(f"Full reindex ({full_report.added} chunks added): {full_duration:.3f}s")
    print(
        f"Incremental reindex after 1 doc changed "
        f"(added={incremental_report.added} modified={incremental_report.modified} "
        f"unchanged={incremental_report.unchanged}): {incremental_duration:.3f}s"
    )
    print(f"Speedup: {full_duration / incremental_duration:.1f}x")

    queries = [
        "what does document 5 discuss?",
        "what is document 5 about?",  # near-duplicate of above
        "tell me about document 5",  # near-duplicate of above
        "what does document 12 discuss?",
        "what is document 12 about?",  # near-duplicate of above
        "what does document 30 discuss?",
        "how is the weather today?",  # unrelated to corpus, but a distinct query each time
    ]
    for q in queries:
        service.query(q, answer_fn)
    stats = service.cache_stats()
    print(f"Cache stats after {len(queries)} queries: {stats}")

    shutil.rmtree(workdir)


if __name__ == "__main__":
    main()
