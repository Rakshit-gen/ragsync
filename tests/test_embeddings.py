import math

from ragsync.embeddings import get_embedder


def test_embed_returns_unit_vectors():
    embedder = get_embedder()
    vectors = embedder.embed(["hello world", "goodbye world"])
    assert len(vectors) == 2
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-4


def test_similar_texts_are_closer_than_unrelated_ones():
    embedder = get_embedder()
    a, b, c = embedder.embed(
        ["the cat sat on the mat", "a cat was sitting on a mat", "quarterly revenue increased by 12 percent"]
    )

    def cosine(x, y):
        return sum(xi * yi for xi, yi in zip(x, y))

    assert cosine(a, b) > cosine(a, c)


def test_embed_one_matches_batch_embed():
    embedder = get_embedder()
    single = embedder.embed_one("hello world")
    batch = embedder.embed(["hello world"])[0]
    assert single == batch


def test_embed_empty_list_returns_empty():
    embedder = get_embedder()
    assert embedder.embed([]) == []


def test_get_embedder_returns_singleton():
    assert get_embedder() is get_embedder()
