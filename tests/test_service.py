from ragsync.document_source import DocumentSource
from ragsync.models import Chunk
from ragsync.service import RagSyncService


class FakeSource(DocumentSource):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    def load_chunks(self) -> list[Chunk]:
        return self._chunks


def fake_answer_fn(query: str, chunks: list[dict]) -> str:
    return " ".join(c.get("text", "") for c in chunks) or "no answer"


def test_reindex_then_query_returns_grounded_answer(tmp_path):
    service = RagSyncService(str(tmp_path))
    service.reindex(FakeSource([Chunk("policy", "policy::0", "Refunds are issued within 30 days.")]))

    result = service.query("what is the refund policy?", fake_answer_fn)

    assert result.cache_hit is False
    assert "Refunds" in result.answer
    assert result.matched_chunk_ids == ["policy::0"]


def test_second_identical_query_is_a_cache_hit(tmp_path):
    service = RagSyncService(str(tmp_path))
    service.reindex(FakeSource([Chunk("policy", "policy::0", "Refunds are issued within 30 days.")]))
    service.query("what is the refund policy?", fake_answer_fn)

    result = service.query("what is the refund policy?", fake_answer_fn)

    assert result.cache_hit is True
    assert service.cache_stats()["hits"] == 1


def test_reindex_that_changes_a_grounding_chunk_invalidates_the_cache_entry(tmp_path):
    service = RagSyncService(str(tmp_path))
    service.reindex(FakeSource([Chunk("policy", "policy::0", "Refunds are issued within 30 days.")]))
    service.query("what is the refund policy?", fake_answer_fn)
    assert len(service._cache) == 1

    report = service.reindex(
        FakeSource([Chunk("policy", "policy::0", "Refunds are issued within 14 days.")])
    )

    assert report.invalidated_cache_entries == 1
    assert len(service._cache) == 0

    result = service.query("what is the refund policy?", fake_answer_fn)
    assert result.cache_hit is False
    assert "14 days" in result.answer


def test_reindex_that_deletes_a_grounding_chunk_invalidates_the_cache_entry(tmp_path):
    """Cache invalidation must also cover the deletion path, not just
    modification: an answer grounded in a chunk that later vanishes
    entirely is just as stale as one grounded in a chunk that changed.
    """
    service = RagSyncService(str(tmp_path), top_k=1)
    service.reindex(FakeSource([Chunk("policy", "policy::0", "Refunds are issued within 30 days.")]))
    service.query("what is the refund policy?", fake_answer_fn)
    assert len(service._cache) == 1

    report = service.reindex(FakeSource([]))

    assert report.deleted == 1
    assert report.invalidated_cache_entries == 1
    assert len(service._cache) == 0


def test_reindex_unrelated_to_a_cached_query_leaves_it_cached(tmp_path):
    # top_k=1 so the cached answer is only ever grounded in the single most
    # relevant chunk, keeping this test's "unrelated" premise accurate.
    service = RagSyncService(str(tmp_path), top_k=1)
    service.reindex(
        FakeSource(
            [Chunk("policy", "policy::0", "Refunds are issued within 30 days."),
             Chunk("contact", "contact::0", "Email support@example.com.")]
        )
    )
    service.query("what is the refund policy?", fake_answer_fn)

    report = service.reindex(
        FakeSource(
            [Chunk("policy", "policy::0", "Refunds are issued within 30 days."),
             Chunk("contact", "contact::0", "Email help@example.com instead.")]
        )
    )

    assert report.invalidated_cache_entries == 0
    result = service.query("what is the refund policy?", fake_answer_fn)
    assert result.cache_hit is True
