"""/health 冒烟测试：守护接口形状与索引计数契约（P1/P2/P9）。

- 形状：status + embedding/rerank/generation/index 四个子对象齐备；
- 计数：data/ 6 份 txt 时 docs==6 且 chunks>0（防止“索引为空却全绿”的假绿）。
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok_and_shape():
    """GET /health 返回 200 且四个子对象齐备、关键字段存在。"""
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
    """索引计数恒非负（索引缺失时为 0）。"""
    body = client.get("/health").json()
    assert body["index"]["docs"] >= 0
    assert body["index"]["chunks"] >= 0


def test_index_matches_data():
    """索引与语料一致：6 份 txt → docs==6 且 chunks>0（防假绿）。"""
    data_files = [path for path in Path("data").glob("*.txt") if path.is_file()]
    assert len(data_files) == 6
    body = client.get("/health").json()
    assert body["index"]["docs"] == 6
    assert body["index"]["chunks"] > 0
