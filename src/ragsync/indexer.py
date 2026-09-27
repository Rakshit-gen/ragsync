import time

from ragsync.document_source import DocumentSource
from ragsync.embeddings import Embedder
from ragsync.hashing import chunk_hash
from ragsync.models import ReindexReport
from ragsync.vectorstore import VectorStore


class Reindexer:
    """Diffs a document source's current chunks against what's already in
    the vector store, and only re-embeds what changed.

    Each chunk's content hash is stored alongside its vector as metadata, so
    "did this chunk change" is a hash comparison, not a re-embed-and-compare.
    """

    def __init__(self, store: VectorStore, embedder: Embedder):
        self._store = store
        self._embedder = embedder

    def reindex(self, source: DocumentSource) -> ReindexReport:
        start = time.monotonic()
        chunks = source.load_chunks()
        current = {c.chunk_id: c for c in chunks}
        current_hashes = {cid: chunk_hash(c.source_id, c.text) for cid, c in current.items()}

        previous_ids = self._store.ids()

        added_ids = [cid for cid in current if cid not in previous_ids]
        deleted_ids = [cid for cid in previous_ids if cid not in current]
        modified_ids = [
            cid
            for cid in current
            if cid in previous_ids
            and (self._store.get_metadata(cid) or {}).get("content_hash") != current_hashes[cid]
        ]
        unchanged_count = len(current) - len(added_ids) - len(modified_ids)

        to_upsert_ids = added_ids + modified_ids
        if to_upsert_ids:
            texts = [current[cid].text for cid in to_upsert_ids]
            vectors = self._embedder.embed(texts)
            metadata = [
                {
                    "source_id": current[cid].source_id,
                    "text": current[cid].text,
                    "content_hash": current_hashes[cid],
                }
                for cid in to_upsert_ids
            ]
            self._store.upsert(to_upsert_ids, vectors, metadata)
        if deleted_ids:
            self._store.delete(deleted_ids)
        self._store.save()

        return ReindexReport(
            added=len(added_ids),
            modified=len(modified_ids),
            deleted=len(deleted_ids),
            unchanged=unchanged_count,
            duration_seconds=time.monotonic() - start,
        )
