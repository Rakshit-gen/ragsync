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
        for chunk in resp.iter_bytes(chunk_size=65536):
            f.write(chunk)
    if _sha256(archive) != _MODEL_SHA256:
        os.remove(archive)
        raise ValueError("all-MiniLM-L6-v2 ONNX download failed checksum verification")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(_MODEL_DIR)
    os.remove(archive)
