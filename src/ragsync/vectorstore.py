import json
from pathlib import Path


class VectorStore:
    """A small persisted cosine-similarity index, not a full vector DB.

    Holds every vector in memory as a numpy array and does a brute-force
    similarity scan on search. Fine for the small-to-medium corpora this
    project targets; swap in a real vector DB if that stops being true.
    """

    def __init__(self, path: str):
        import numpy as np

        self._np = np
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._ids: list[str] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._metadata: dict[str, dict] = {}
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
        self._np.save(self._vectors_file(), self._vectors)
        self._meta_file().write_text(json.dumps({"ids": self._ids, "metadata": self._metadata}))

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self) -> set[str]:
        return set(self._ids)

    def get_metadata(self, id_: str) -> dict | None:
        return self._metadata.get(id_)
