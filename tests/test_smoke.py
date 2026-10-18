from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok_and_shape():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    for key in ("embedding", "rerank", "generation", "index"):
        assert key in body
        assert isinstance(body[key], dict)
    assert {"model", "mode"} <= set(body["embedding"].keys())
    assert {"docs", "chunks"} <= set(body["index"].keys())


def test_health_index_counts_zero_when_missing():
    body = client.get("/health").json()
    assert body["index"]["docs"] >= 0
    assert body["index"]["chunks"] >= 0


def test_index_matches_data():
    data_files = [path for path in Path("data").glob("*.txt") if path.is_file()]
    assert len(data_files) == 6
    body = client.get("/health").json()
    assert body["index"]["docs"] == 6
    assert body["index"]["chunks"] > 0
