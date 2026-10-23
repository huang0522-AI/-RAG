"""索引存储：chunks.jsonl 落盘 + Chroma 向量读写，是链路的持久化底座（P2）。

- chunks.jsonl 是唯一事实来源：切块结果与 BM25 语料都从它重建；
- Chroma 只存向量与元数据（doc/title/chunk_id），服务于向量路召回；
- 向量写入前先删 collection，保证重复 ingest 幂等（向量维度会随 embedding 后端变化）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

COLLECTION_NAME = "campus"
CHUNKS_FILE = "chunks.jsonl"


def _chunks_path(storage_dir) -> Path:
    """拼出 chunks.jsonl 的完整路径。"""
    return Path(storage_dir) / CHUNKS_FILE


def write_chunks(chunks, storage_dir) -> None:
    """用覆盖方式把 chunks 写入 storage_dir/chunks.jsonl（每行一个 JSON）。

    Args:
        chunks: chunk 字典列表。
        storage_dir: 存储目录，不存在时创建。
    """
    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with _chunks_path(storage_dir).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def read_chunks(storage_dir) -> list[dict]:
    """读取 chunks.jsonl，返回 chunk 字典列表；文件不存在时返回 []。"""
    path = _chunks_path(storage_dir)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _client(storage_dir):
    """创建 Chroma 持久化客户端（关闭匿名遥测）。"""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    # 该版本即便关闭遥测仍会尝试发送并报错，直接把对应 logger 提到 CRITICAL 消噪
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(
        logging.CRITICAL
    )
    return chromadb.PersistentClient(
        path=str(storage_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(storage_dir):
    """返回（按需创建）使用 cosine 距离的 campus collection。"""
    client = _client(storage_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def upsert_vectors(chunks, embeddings, storage_dir) -> None:
    """全量重建向量索引：先删 collection 再写入，保证重复 ingest 幂等。

    Args:
        chunks: chunk 字典列表，需含 chunk_id/doc/title/text。
        embeddings: 与 chunks 等长、同序的向量列表。
        storage_dir: 存储目录。
    """
    client = _client(storage_dir)
    # 先删再建：向量维度随 embedding 后端变化，旧向量不能与新向量混用
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if not chunks:
        return
    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    # Chroma 的 metadata 不接受 None，这里统一转成字符串
    metadatas = [
        {
            "doc": str(chunk["doc"]),
            "title": str(chunk["title"]),
            "chunk_id": str(chunk["chunk_id"]),
        }
        for chunk in chunks
    ]
    collection.add(
        ids=ids,
        embeddings=[list(vector) for vector in embeddings],
        documents=documents,
        metadatas=metadatas,
    )


def count_docs(chunks) -> int:
    """统计 chunk 覆盖的文档数（按 doc 去重）。"""
    return len({chunk["doc"] for chunk in chunks})


def query_vectors(query_embedding, top_k, storage_dir) -> list[dict]:
    """向量召回 top_k，返回 [{chunk_id, doc, title, text, score}]，按 score 降序。

    Args:
        query_embedding: 查询向量。
        top_k: 期望返回条数（集合为空时返回 []）。
        storage_dir: 存储目录。

    Returns:
        召回结果列表；查询异常时记录日志并返回 []（不向上抛错）。
    """
    try:
        collection = get_collection(storage_dir)
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, total),
            include=["metadatas", "documents", "distances"],
        )
    except Exception:
        logger.exception("[store] vector query failed")
        return []

    ids = (result.get("ids") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    rows: list[dict] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        distance = distances[index] if index < len(distances) else None
        if distance is None:
            score = 0.0
        else:
            # cosine 距离转为相似度 1-distance；距离理论范围 [0,2]，
            # 转出后 clip 到 [0,1]，避免浮点误差影响后续按分数比较
            score = max(0.0, min(1.0, 1.0 - float(distance)))
        rows.append(
            {
                "chunk_id": chunk_id,
                "doc": metadata.get("doc", ""),
                "title": metadata.get("title", ""),
                "text": documents[index] if index < len(documents) else "",
                "score": score,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows
