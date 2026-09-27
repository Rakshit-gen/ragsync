# Development

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First test run downloads the all-MiniLM-L6-v2 ONNX model (~90MB) to
`~/.cache/ragsync/onnx_models/` and caches it there for every run after.

## Running tests

```
pytest -q
ruff check .
```

## Layout

See the README's Architecture section for what each module does. Tests
mirror the module they cover 1:1 (`test_indexer.py` tests `indexer.py`,
etc.), except `test_service.py` and `test_app.py`, which are integration
tests across the whole stack.

## Running the API locally

```
uvicorn ragsync.app:app --reload
```

`RAGSYNC_STORE_PATH` controls where the vector index persists (default
`./data`).
