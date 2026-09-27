from abc import ABC, abstractmethod
from pathlib import Path

from ragsync.models import Chunk


class DocumentSource(ABC):
    """A source of chunks to index. Not tied to a filesystem: an
    implementation could just as easily pull from a database or an API.
    """

    @abstractmethod
    def load_chunks(self) -> list[Chunk]:
        raise NotImplementedError


def split_into_chunks(text: str, max_chars: int = 800) -> list[str]:
    """Split text on paragraph boundaries, packing consecutive paragraphs
    into chunks up to max_chars. Falls back to a hard split for any single
    paragraph longer than max_chars on its own.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class DirectorySource(DocumentSource):
    """Reads every .txt/.md file under a directory and chunks it."""

    def __init__(self, directory: str, max_chars: int = 800, extensions: tuple[str, ...] = (".txt", ".md")):
        self._directory = Path(directory)
        self._max_chars = max_chars
        self._extensions = extensions

    def load_chunks(self) -> list[Chunk]:
        chunks = []
        for path in sorted(self._directory.rglob("*")):
            if not path.is_file() or path.suffix not in self._extensions:
                continue
            source_id = str(path.relative_to(self._directory))
            text = path.read_text(encoding="utf-8")
            for i, piece in enumerate(split_into_chunks(text, self._max_chars)):
                chunks.append(Chunk(source_id=source_id, chunk_id=f"{source_id}::{i}", text=piece))
        return chunks
