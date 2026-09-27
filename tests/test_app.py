def test_reindex_endpoint_reports_added_chunks(api_client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.txt").write_text("Refunds are issued within 30 days.")

    response = api_client.post("/reindex", json={"directory": str(docs)})

    assert response.status_code == 200
    body = response.json()
    assert body["added"] == 1
    assert body["deleted"] == 0


def test_query_endpoint_returns_answer_and_cache_flag(api_client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.txt").write_text("Refunds are issued within 30 days.")
    api_client.post("/reindex", json={"directory": str(docs)})

    response = api_client.post("/query", json={"query": "what is the refund policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["cache_hit"] is False
    assert "Refunds" in body["answer"]


def test_query_endpoint_second_call_is_cached(api_client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.txt").write_text("Refunds are issued within 30 days.")
    api_client.post("/reindex", json={"directory": str(docs)})
    api_client.post("/query", json={"query": "what is the refund policy?"})

    response = api_client.post("/query", json={"query": "what is the refund policy?"})

    assert response.json()["cache_hit"] is True


def test_cache_stats_endpoint_reports_hits_and_misses(api_client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.txt").write_text("Refunds are issued within 30 days.")
    api_client.post("/reindex", json={"directory": str(docs)})
    api_client.post("/query", json={"query": "what is the refund policy?"})
    api_client.post("/query", json={"query": "what is the refund policy?"})

    stats = api_client.get("/cache/stats").json()

    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_query_with_empty_corpus_returns_no_relevant_content(api_client):
    response = api_client.post("/query", json={"query": "anything?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "No relevant content found."
