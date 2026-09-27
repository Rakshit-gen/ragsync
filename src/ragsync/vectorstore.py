import json
import threading
import warnings
from pathlib import Path


class VectorStore:
    """A small persisted cosine-similarity index, not a full vector DB.

    Holds every vector in memory as a numpy array and does a brute-force
    similarity scan on search. Fine for the small-to-medium corpora this
    project targets; swap in a real vector DB if that stops being true.

    FastAPI runs sync `def` endpoints in a threadpool, so a `/reindex` and a
    `/query` can genuinely execute concurrently. upsert/delete rebuild
    `_vectors`/`_ids` by full reassignment (not in place), and search reads
    `_ids` more than once — without a lock, a write landing between those
    reads desyncs the ids returned from the scores they're paired with, or
    raises IndexError outright. Verified with a stress test before adding
    this lock: ~2400 crashes in a few seconds of concurrent upsert/delete/
    search on one store. A single lock is the right size fix here — this
    store's whole read/write volume is one process's worth of chunks.
    """

    def __init__(self, path: str):
        import numpy as np

        self._np = np
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._ids: list[str] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._metadata: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._load()

    def _vectors_file(self) -> Path:
        return self._path / "vectors.npy"

    def _meta_file(self) -> Path:
        return self._path / "meta.json"

    def _load(self) -> None:
        np = self._np
        if self._vectors_file().exists() and self._meta_file().exists():
            self._vectors = np.load(self._vectors_file())
            data = json.loads(self._meta_file().read_text())
            self._ids = data["ids"]
            self._metadata = data["metadata"]

    def save(self) -> None:
        with self._lock:
            self._np.save(self._vectors_file(), self._vectors)
            self._meta_file().write_text(json.dumps({"ids": self._ids, "metadata": self._metadata}))

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def ids(self) -> set[str]:
        with self._lock:
            return set(self._ids)

    def get_metadata(self, id_: str) -> dict | None:
        with self._lock:
            return self._metadata.get(id_)

    def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict] | None = None) -> None:
        """Insert or replace vectors by id. An id already present is
        dropped and re-added with its new vector/metadata, not merged.
        """
        np = self._np
        metadata = metadata or [{} for _ in ids]
        new_vectors = np.array(vectors, dtype=np.float32)

        with self._lock:
            replacing = set(ids)
            keep_mask = np.array([id_ not in replacing for id_ in self._ids], dtype=bool)
            kept_ids = [id_ for id_, keep in zip(self._ids, keep_mask) if keep]
            kept_vectors = self._vectors[keep_mask] if len(self._ids) else new_vectors[:0]

            self._vectors = np.concatenate([kept_vectors, new_vectors])
            self._ids = kept_ids + list(ids)
            for id_, meta in zip(ids, metadata):
                self._metadata[id_] = meta

    def delete(self, ids: list[str]) -> None:
        """Remove ids from the index. Unknown ids are silently ignored."""
        if not ids:
            return
        np = self._np
        with self._lock:
            to_delete = set(ids)
            keep_mask = np.array([id_ not in to_delete for id_ in self._ids], dtype=bool)
            self._vectors = self._vectors[keep_mask] if len(self._vectors) else self._vectors
            self._ids = [id_ for id_ in self._ids if id_ not in to_delete]
            for id_ in ids:
                self._metadata.pop(id_, None)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Brute-force cosine similarity scan. Returns (id, score) pairs
        sorted by descending score, since all stored vectors are already
        unit-normalized by the embedder.
        """
        np = self._np
        q = np.array(query_vector, dtype=np.float32)
        with self._lock:
            if len(self._ids) == 0:
                return []
            # macOS's Accelerate BLAS backend raises spurious divide-by-zero /
            # overflow / invalid-value RuntimeWarnings on some matmul shapes
            # even though the output is numerically correct (verified: no
            # NaN/Inf, values in the expected cosine-similarity range). Known
            # numpy-on-Accelerate platform quirk, not a real computation error.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
                scores = self._vectors @ q
            top_k = min(top_k, len(self._ids))
            top_indices = np.argsort(-scores)[:top_k]
            return [(self._ids[i], float(scores[i])) for i in top_indices]
