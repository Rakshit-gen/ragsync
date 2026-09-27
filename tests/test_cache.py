from ragsync.cache import SemanticCache
from ragsync.embeddings import get_embedder


def test_miss_on_empty_cache():
    cache = SemanticCache(get_embedder())
    assert cache.get("what is the refund policy?") is None
    assert cache.stats()["misses"] == 1


def test_hit_on_near_duplicate_query():
    cache = SemanticCache(get_embedder(), similarity_threshold=0.85)
    cache.put(
        "what is the refund policy?",
        answer="Refunds within 30 days.",
        matched_chunk_ids=["policy::0"],
        dependent_hashes={"hash-a"},
    )
    hit = cache.get("what's your refund policy")
    assert hit is not None
    assert hit.answer == "Refunds within 30 days."
    assert cache.stats()["hits"] == 1


def test_miss_on_unrelated_query():
    cache = SemanticCache(get_embedder(), similarity_threshold=0.9)
    cache.put(
        "what is the refund policy?",
        answer="Refunds within 30 days.",
        matched_chunk_ids=["policy::0"],
        dependent_hashes={"hash-a"},
    )
    miss = cache.get("what is the weather forecast for tomorrow?")
    assert miss is None
    assert cache.stats()["misses"] == 1
