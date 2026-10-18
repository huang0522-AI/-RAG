from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

COLLECTION_NAME = "campus"
CHUNKS_FILE = "chunks.jsonl"


def _chunks_path(storage_dir) -> Path:
    return Path(storage_dir) / CHUNKS_FILE


def write_chunks(chunks, storage_dir) -> None:
    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with _chunks_path(storage_dir).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def read_chunks(storage_dir) -> list[dict]:
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
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(
        logging.CRITICAL
    )
    return chromadb.PersistentClient(
        path=str(storage_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(storage_dir):
    client = _client(storage_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def upsert_vectors(chunks, embeddings, storage_dir) -> None:
    client = _client(storage_dir)
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
    return len({chunk["doc"] for chunk in chunks})


def query_vectors(query_embedding, top_k, storage_dir) -> list[dict]:
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
