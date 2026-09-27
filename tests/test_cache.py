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


def test_invalidation_drops_entries_depending_on_a_changed_hash():
    cache = SemanticCache(get_embedder(), similarity_threshold=0.85)
    cache.put(
        "what is the refund policy?",
        answer="Refunds within 30 days.",
        matched_chunk_ids=["policy::0"],
        dependent_hashes={"hash-a"},
    )

    removed = cache.invalidate_for_changed_hashes({"hash-a"})

    assert removed == 1
    assert len(cache) == 0
    assert cache.get("what's your refund policy") is None
    assert cache.stats()["invalidations"] == 1


def test_invalidation_leaves_unrelated_entries_alone():
    cache = SemanticCache(get_embedder(), similarity_threshold=0.85)
    cache.put(
        "what is the refund policy?",
        answer="Refunds within 30 days.",
        matched_chunk_ids=["policy::0"],
        dependent_hashes={"hash-a"},
    )
    cache.put(
        "how do I contact support?",
        answer="Email support@example.com.",
        matched_chunk_ids=["contact::0"],
        dependent_hashes={"hash-b"},
    )

    removed = cache.invalidate_for_changed_hashes({"hash-a"})

    assert removed == 1
    assert len(cache) == 1
    hit = cache.get("how do I contact support")
    assert hit is not None
    assert hit.answer == "Email support@example.com."


def test_entry_depending_on_multiple_chunks_is_invalidated_if_any_one_changes():
    cache = SemanticCache(get_embedder(), similarity_threshold=0.85)
    cache.put(
        "summarize the refund and shipping policy",
        answer="Refunds in 30 days; free shipping over $50.",
        matched_chunk_ids=["policy::0", "shipping::0"],
        dependent_hashes={"hash-a", "hash-c"},
    )

    removed = cache.invalidate_for_changed_hashes({"hash-c"})

    assert removed == 1
    assert len(cache) == 0


def test_invalidate_with_no_changed_hashes_is_a_noop():
    cache = SemanticCache(get_embedder(), similarity_threshold=0.85)
    cache.put(
        "what is the refund policy?",
        answer="Refunds within 30 days.",
        matched_chunk_ids=["policy::0"],
        dependent_hashes={"hash-a"},
    )

    removed = cache.invalidate_for_changed_hashes(set())

    assert removed == 0
    assert len(cache) == 1
