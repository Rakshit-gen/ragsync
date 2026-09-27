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
