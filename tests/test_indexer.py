from ragsync.document_source import DocumentSource
from ragsync.embeddings import get_embedder
from ragsync.indexer import Reindexer
from ragsync.models import Chunk
from ragsync.vectorstore import VectorStore


class FakeSource(DocumentSource):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    def load_chunks(self) -> list[Chunk]:
        return self._chunks


def test_first_reindex_adds_every_chunk(tmp_path):
    store = VectorStore(str(tmp_path))
    reindexer = Reindexer(store, get_embedder())
    source = FakeSource([Chunk("doc1", "doc1::0", "hello"), Chunk("doc1", "doc1::1", "world")])

    report = reindexer.reindex(source)

    assert report.added == 2
    assert report.modified == 0
    assert report.deleted == 0
    assert report.unchanged == 0
    assert store.ids() == {"doc1::0", "doc1::1"}


def test_second_reindex_with_no_changes_is_a_full_noop(tmp_path):
    store = VectorStore(str(tmp_path))
    reindexer = Reindexer(store, get_embedder())
    source = FakeSource([Chunk("doc1", "doc1::0", "hello"), Chunk("doc1", "doc1::1", "world")])
    reindexer.reindex(source)

    report = reindexer.reindex(source)

    assert report.added == 0
    assert report.modified == 0
    assert report.deleted == 0
    assert report.unchanged == 2


def test_modified_chunk_content_is_reembedded(tmp_path):
    store = VectorStore(str(tmp_path))
    reindexer = Reindexer(store, get_embedder())
    reindexer.reindex(FakeSource([Chunk("doc1", "doc1::0", "hello"), Chunk("doc1", "doc1::1", "world")]))

    report = reindexer.reindex(
        FakeSource([Chunk("doc1", "doc1::0", "hello there, changed"), Chunk("doc1", "doc1::1", "world")])
    )

    assert report.added == 0
    assert report.modified == 1
    assert report.deleted == 0
    assert report.unchanged == 1
    assert store.get_metadata("doc1::0")["text"] == "hello there, changed"


def test_modified_and_deleted_chunks_report_their_old_hash_as_stale(tmp_path):
    store = VectorStore(str(tmp_path))
    reindexer = Reindexer(store, get_embedder())
    reindexer.reindex(
        FakeSource(
            [
                Chunk("doc1", "doc1::0", "hello"),
                Chunk("doc1", "doc1::1", "world"),
                Chunk("doc1", "doc1::2", "goodbye"),
            ]
        )
    )
    old_hash_0 = store.get_metadata("doc1::0")["content_hash"]
    old_hash_2 = store.get_metadata("doc1::2")["content_hash"]

    report = reindexer.reindex(
        FakeSource([Chunk("doc1", "doc1::0", "hello, changed"), Chunk("doc1", "doc1::1", "world")])
    )

    assert report.stale_hashes == {old_hash_0, old_hash_2}


def test_removed_chunk_is_deleted_from_store(tmp_path):
    store = VectorStore(str(tmp_path))
    reindexer = Reindexer(store, get_embedder())
    reindexer.reindex(FakeSource([Chunk("doc1", "doc1::0", "hello"), Chunk("doc1", "doc1::1", "world")]))

    report = reindexer.reindex(FakeSource([Chunk("doc1", "doc1::0", "hello")]))

    assert report.added == 0
    assert report.modified == 0
    assert report.deleted == 1
    assert report.unchanged == 1
    assert store.ids() == {"doc1::0"}


def test_empty_corpus_reindex_is_clean(tmp_path):
    store = VectorStore(str(tmp_path))
    reindexer = Reindexer(store, get_embedder())

    report = reindexer.reindex(FakeSource([]))

    assert (report.added, report.modified, report.deleted, report.unchanged) == (0, 0, 0, 0)
    assert len(store) == 0
