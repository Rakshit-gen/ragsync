from ragsync.hashing import chunk_hash


def test_same_source_and_text_gives_same_hash():
    assert chunk_hash("doc1", "hello world") == chunk_hash("doc1", "hello world")


def test_different_text_gives_different_hash():
    assert chunk_hash("doc1", "hello world") != chunk_hash("doc1", "goodbye world")


def test_same_text_different_source_gives_different_hash():
    assert chunk_hash("doc1", "hello world") != chunk_hash("doc2", "hello world")


def test_hash_is_hex_sha256_length():
    h = chunk_hash("doc1", "hello world")
    assert len(h) == 64
    int(h, 16)  # raises ValueError if not valid hex
