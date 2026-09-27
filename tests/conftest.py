import importlib

import pytest


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """app.py builds one RagSyncService at import time, pointed at
    RAGSYNC_STORE_PATH. Reload the module per test so each test gets an
    isolated store instead of all tests sharing one process-lifetime cache.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RAGSYNC_STORE_PATH", str(tmp_path / "store"))
    import ragsync.app as app_module
    import ragsync.config as config_module

    importlib.reload(config_module)
    importlib.reload(app_module)
    return TestClient(app_module.app)
