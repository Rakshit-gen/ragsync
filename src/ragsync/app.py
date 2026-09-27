from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel

from ragsync.config import STORE_PATH
from ragsync.document_source import DirectorySource
from ragsync.service import RagSyncService

app = FastAPI(title="ragsync")
_service = RagSyncService(STORE_PATH)


def _placeholder_answer(query: str, chunks: list[dict]) -> str:
    """Joins the retrieved chunk text. Swap for a real LLM call at the
    integration point in service.query(); no LLM dependency is wired in
    here so this project stays backend-only and network-independent.
    """
    if not chunks:
        return "No relevant content found."
    return " ".join(c.get("text", "") for c in chunks)


class ReindexRequest(BaseModel):
    directory: str


class QueryRequest(BaseModel):
    query: str


@app.post("/reindex")
def reindex(request: ReindexRequest) -> dict:
    report = _service.reindex(DirectorySource(request.directory))
    result = asdict(report)
    result["stale_hashes"] = list(result["stale_hashes"])
    return result


@app.post("/query")
def query(request: QueryRequest) -> dict:
    result = _service.query(request.query, _placeholder_answer)
    return asdict(result)


@app.get("/cache/stats")
def cache_stats() -> dict:
    return _service.cache_stats()
