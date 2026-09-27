import hashlib


def chunk_hash(source_id: str, text: str) -> str:
    """Stable content hash for a chunk.

    Keyed on (source_id, text) rather than text alone: two different
    documents can legitimately share an identical paragraph, and those must
    be tracked as independent chunks, not deduplicated into one.
    """
    digest = hashlib.sha256()
    digest.update(source_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()
