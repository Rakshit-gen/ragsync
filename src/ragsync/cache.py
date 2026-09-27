import logging
from dataclasses import dataclass, field

from ragsync.embeddings import Embedder

logger = logging.getLogger("ragsync.cache")


@dataclass
class CacheEntry:
    query_text: str
    query_vector: list[float]
    answer: str
    matched_chunk_ids: list[str]
    dependent_hashes: set[str] = field(default_factory=set)


class SemanticCache:
    """Caches (query, answer) pairs and serves a cached answer for a
    near-duplicate query, similar to any semantic cache. The difference is
    that every entry also records the content hashes of the chunks its
    answer was actually grounded in — so when the corpus changes, only the
    entries that depended on a changed chunk get invalidated, not the whole
    cache and not nothing (see module docstring in indexer.py for why a
    plain TTL isn't enough: it either serves genuinely stale answers between
    TTL windows, or throws away perfectly good cache entries on unrelated
    corpus changes).

    In-memory only, no persistence: a cache that survives a restart still
    needs the same staleness check on load, so persisting it wouldn't save
    real work, just add a file format to maintain.
    """

    def __init__(self, embedder: Embedder, similarity_threshold: float = 0.92):
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._entries: list[CacheEntry] = []
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, query: str) -> CacheEntry | None:
        if not self._entries:
            self.misses += 1
            return None
        query_vector = self._embedder.embed_one(query)
        best_entry = None
        best_score = -1.0
        for entry in self._entries:
            score = sum(a * b for a, b in zip(query_vector, entry.query_vector))
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry is not None and best_score >= self._threshold:
            self.hits += 1
            logger.info("cache hit: query=%r score=%.4f", query, best_score)
            return best_entry
        self.misses += 1
        logger.info("cache miss: query=%r best_score=%.4f", query, best_score)
        return None

    def put(self, query: str, answer: str, matched_chunk_ids: list[str], dependent_hashes: set[str]) -> None:
        query_vector = self._embedder.embed_one(query)
        self._entries.append(
            CacheEntry(
                query_text=query,
                query_vector=query_vector,
                answer=answer,
                matched_chunk_ids=matched_chunk_ids,
                dependent_hashes=set(dependent_hashes),
            )
        )

    def invalidate_for_changed_hashes(self, changed_hashes: set[str]) -> int:
        """Drop every cache entry whose answer depended on any of the given
        content hashes. Returns how many entries were dropped.
        """
        if not changed_hashes:
            return 0
        kept = [e for e in self._entries if not (e.dependent_hashes & changed_hashes)]
        removed = len(self._entries) - len(kept)
        self._entries = kept
        self.invalidations += removed
        if removed:
            logger.info("invalidated %d cache entries for %d changed hashes", removed, len(changed_hashes))
        return removed

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "invalidations": self.invalidations,
        }
