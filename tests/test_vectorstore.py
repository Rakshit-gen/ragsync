import warnings

import numpy as np

from ragsync.vectorstore import VectorStore


def test_empty_store_search_returns_nothing(tmp_path):
    store = VectorStore(str(tmp_path))
    assert store.search([1.0, 0.0], top_k=5) == []
    assert len(store) == 0


def test_upsert_then_search_finds_closest(tmp_path):
    store = VectorStore(str(tmp_path))
    store.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        metadata=[{"n": 1}, {"n": 2}, {"n": 3}],
    )
    results = store.search([1.0, 0.0], top_k=2)
    assert results[0][0] == "a"
    assert results[0][1] > results[1][1]


def test_upsert_replaces_existing_id(tmp_path):
    store = VectorStore(str(tmp_path))
    store.upsert(ids=["a"], vectors=[[1.0, 0.0]], metadata=[{"v": 1}])
    store.upsert(ids=["a"], vectors=[[0.0, 1.0]], metadata=[{"v": 2}])
    assert len(store) == 1
    assert store.get_metadata("a") == {"v": 2}
    results = store.search([0.0, 1.0], top_k=1)
    assert results[0][0] == "a"


def test_delete_removes_id(tmp_path):
    store = VectorStore(str(tmp_path))
    store.upsert(ids=["a", "b"], vectors=[[1.0, 0.0], [0.0, 1.0]])
    store.delete(["a"])
    assert store.ids() == {"b"}
    assert len(store) == 1
    assert store.get_metadata("a") is None


def test_delete_nonexistent_id_is_a_noop(tmp_path):
    store = VectorStore(str(tmp_path))
    store.upsert(ids=["a"], vectors=[[1.0, 0.0]])
    store.delete(["does-not-exist"])
    assert len(store) == 1


def test_save_and_reload_round_trips(tmp_path):
    store = VectorStore(str(tmp_path))
    store.upsert(ids=["a", "b"], vectors=[[1.0, 0.0], [0.0, 1.0]], metadata=[{"n": 1}, {"n": 2}])
    store.save()

    reloaded = VectorStore(str(tmp_path))
    assert reloaded.ids() == {"a", "b"}
    assert reloaded.get_metadata("a") == {"n": 1}
    results = reloaded.search([1.0, 0.0], top_k=1)
    assert results[0][0] == "a"


def test_search_over_many_vectors_produces_finite_scores_and_no_uncaught_warnings(tmp_path):
    """Regression test for a macOS Accelerate BLAS quirk: matmul on some
    shapes raises spurious RuntimeWarnings even though the output is
    correct. search() suppresses that specific noise; this test checks the
    actual values stay finite and nothing else slips through unsuppressed.
    """
    rng = np.random.default_rng(0)
    store = VectorStore(str(tmp_path))
    vectors = rng.standard_normal((97, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.upsert([f"id{i}" for i in range(97)], vectors.tolist())

    query = vectors[0].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = store.search(query, top_k=5)

    assert results[0][0] == "id0"
    assert all(np.isfinite(score) for _, score in results)
