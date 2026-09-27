from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One unit of source content, before embedding."""

    source_id: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk as stored in the vector index, with its content hash and vector."""

    source_id: str
    chunk_id: str
    text: str
    content_hash: str
    vector: list[float]


@dataclass
class ReindexReport:
    added: int = 0
    modified: int = 0
    deleted: int = 0
    unchanged: int = 0
    duration_seconds: float = 0.0
    invalidated_cache_entries: int = 0


@dataclass
class QueryResult:
    answer: str
    cache_hit: bool
    matched_chunk_ids: list[str] = field(default_factory=list)
