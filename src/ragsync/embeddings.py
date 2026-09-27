import hashlib
import os
import tarfile
from pathlib import Path

# onnxruntime's telemetry worker thread has a known crash-on-exit bug on
# macOS (a mutex it already tore down gets locked again during interpreter
# shutdown). Disabling telemetry before onnxruntime is imported avoids
# starting that thread in the first place.
os.environ.setdefault("ORT_DISABLE_TELEMETRY_EVENTS", "1")

_MODEL_DIR = Path.home() / ".cache" / "ragsync" / "onnx_models" / "all-MiniLM-L6-v2"
_MODEL_FILES_DIR = _MODEL_DIR / "onnx"  # the archive's top-level entry is an "onnx/" dir
_MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
_MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_model_downloaded() -> None:
    if (_MODEL_FILES_DIR / "model.onnx").exists() and (_MODEL_FILES_DIR / "tokenizer.json").exists():
        return
    import httpx

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    archive = _MODEL_DIR / "onnx.tar.gz"
    with httpx.stream("GET", _MODEL_URL) as resp, open(archive, "wb") as f:
        f.writelines(resp.iter_bytes(chunk_size=65536))
    if _sha256(archive) != _MODEL_SHA256:
        os.remove(archive)
        raise ValueError("all-MiniLM-L6-v2 ONNX download failed checksum verification")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(_MODEL_DIR)
    os.remove(archive)


class Embedder:
    """Runs all-MiniLM-L6-v2 via onnxruntime + tokenizers directly, instead
    of through chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2.

    chromadb's wrapper does the same inference but importing it drags in the
    whole chromadb package (grpc, opentelemetry, ~850 modules), which spiked
    peak RSS to ~780MB on a related project in this account and caused
    production OOMs. This does the same tokenize -> forward pass -> mean-pool
    -> normalize pipeline chromadb uses, without the import weight.
    """

    def __init__(self):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        _ensure_model_downloaded()

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(_MODEL_FILES_DIR / "model.onnx"), providers=["CPUExecutionProvider"], sess_options=so
        )
        self._tokenizer = Tokenizer.from_file(str(_MODEL_FILES_DIR / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=256)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        np = self._np
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = [self._tokenizer.encode(t) for t in batch]
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids)
            last_hidden_state = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )[0]
            mask = np.broadcast_to(attention_mask[..., None], last_hidden_state.shape)
            pooled = np.sum(last_hidden_state * mask, axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            norm = np.linalg.norm(pooled, axis=1)
            norm[norm == 0] = 1e-12
            all_embeddings.append((pooled / norm[:, None]).astype(np.float32))
        return np.concatenate(all_embeddings).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
