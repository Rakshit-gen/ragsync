from collections.abc import Callable

from ragsync.cache import SemanticCache
from ragsync.document_source import DocumentSource
from ragsync.embeddings import get_embedder
from ragsync.indexer import Reindexer
from ragsync.models import QueryResult, ReindexReport
from ragsync.vectorstore import VectorStore

AnswerFn = Callable[[str, list[dict]], str]


class RagSyncService:
    """Ties the vector store, reindexer, and semantic cache into one
    importable API. The HTTP layer in app.py is a thin wrapper over this.
    """

    def __init__(self, store_path: str, similarity_threshold: float = 0.92, top_k: int = 5):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in (0, 1], got {similarity_threshold}")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        self._embedder = get_embedder()
        self._store = VectorStore(store_path)
        self._reindexer = Reindexer(self._store, self._embedder)
        self._cache = SemanticCache(self._embedder, similarity_threshold=similarity_threshold)
        self._top_k = top_k

    def reindex(self, source: DocumentSource) -> ReindexReport:
        """Diff the source against the current index, re-embed only what
        changed, and invalidate any cache entries grounded in changed
        content.
        """
        report = self._reindexer.reindex(source)
        invalidated = self._cache.invalidate_for_changed_hashes(report.stale_hashes)
        report.invalidated_cache_entries = invalidated
        return report

    def query(self, query_text: str, answer_fn: AnswerFn) -> QueryResult:
        """Look up a near-duplicate cached answer first; on a miss, retrieve
        the top-k chunks, call answer_fn(query_text, chunks) to generate an
        answer, and cache it keyed on the exact chunks it used.
        """
        cached = self._cache.get(query_text)
        if cached is not None:
            return QueryResult(answer=cached.answer, cache_hit=True, matched_chunk_ids=cached.matched_chunk_ids)

        query_vector = self._embedder.embed_one(query_text)
        matches = self._store.search(query_vector, top_k=self._top_k)
        chunk_ids = [chunk_id for chunk_id, _ in matches]
        chunks = [self._store.get_metadata(cid) or {} for cid in chunk_ids]

        answer = answer_fn(query_text, chunks)
        dependent_hashes = {c["content_hash"] for c in chunks if "content_hash" in c}
        self._cache.put(query_text, answer, chunk_ids, dependent_hashes)

        return QueryResult(answer=answer, cache_hit=False, matched_chunk_ids=chunk_ids)

    def cache_stats(self) -> dict:
        return self._cache.stats()
