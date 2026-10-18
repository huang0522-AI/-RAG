from __future__ import annotations

import asyncio
import logging
import time

from app import store
from app.bm25_index import bm25_search
from app.config import settings
from app.embedder import get_embedder

logger = logging.getLogger(__name__)


async def vector_search(question: str, top_k: int) -> list[dict]:
    embedder = get_embedder()
    if embedder.mode == "disabled":
        logger.warning("[retriever] vector search disabled, embedding unavailable")
        return []
    try:
        vectors = await asyncio.to_thread(embedder.encode, [question])
        query_embedding = vectors[0]
        rows = await asyncio.to_thread(
            store.query_vectors, query_embedding, top_k, settings.storage_dir
        )
    except Exception:
        logger.exception("[retriever] vector search failed")
        return []
    for row in rows:
        row["source"] = "vector"
    return rows


async def _timed_vector(question: str, top_k: int) -> tuple[list[dict], float]:
    started = time.perf_counter()
    try:
        rows = await vector_search(question, top_k)
    except Exception:
        logger.exception("[retriever] vector leg failed")
        rows = []
    return rows, (time.perf_counter() - started) * 1000.0


async def _timed_bm25(question: str, top_k: int) -> tuple[list[dict], float]:
    started = time.perf_counter()

    def _work() -> list[dict]:
        return bm25_search(question, top_k, settings.storage_dir)

    try:
        rows = await asyncio.to_thread(_work)
    except Exception:
        logger.exception("[retriever] bm25 leg failed")
        rows = []
    return rows, (time.perf_counter() - started) * 1000.0


async def hybrid_search(question: str, top_k: int) -> tuple[list[dict], dict]:
    (vector_rows, vector_ms), (bm25_rows, bm25_ms) = await asyncio.gather(
        _timed_vector(question, top_k),
        _timed_bm25(question, top_k),
    )

    k = settings.rrf_k
    fused: dict[str, dict] = {}
    for rank, row in enumerate(vector_rows, start=1):
        entry = fused.setdefault(
            row["chunk_id"],
            {
                "chunk_id": row["chunk_id"],
                "doc": row["doc"],
                "title": row["title"],
                "text": row["text"],
                "rrf_score": 0.0,
                "ranks": {"vector": None, "bm25": None},
            },
        )
        entry["rrf_score"] += 1.0 / (k + rank)
        entry["ranks"]["vector"] = rank
    for rank, row in enumerate(bm25_rows, start=1):
        entry = fused.setdefault(
            row["chunk_id"],
            {
                "chunk_id": row["chunk_id"],
                "doc": row["doc"],
                "title": row["title"],
                "text": row["text"],
                "rrf_score": 0.0,
                "ranks": {"vector": None, "bm25": None},
            },
        )
        entry["rrf_score"] += 1.0 / (k + rank)
        entry["ranks"]["bm25"] = rank

    results = sorted(fused.values(), key=lambda row: row["rrf_score"], reverse=True)
    timing = {
        "embed_ms": 0,
        "vector_ms": int(round(vector_ms)),
        "bm25_ms": int(round(bm25_ms)),
        "candidates": len(fused),
    }
    return results[:top_k], timing


async def retrieve(question: str, top_k: int = 5) -> tuple[list[dict], dict]:
    from app.reranker import get_reranker

    candidates, timing = await hybrid_search(question, 20)
    reranker = get_reranker()
    rerank_ms = 0.0
    if candidates:
        started = time.perf_counter()
        try:
            results = await asyncio.to_thread(
                reranker.rerank, question, candidates, top_k
            )
        except Exception:
            logger.exception("[retriever] rerank failed, using fused order")
            results = candidates[:top_k]
        rerank_ms = (time.perf_counter() - started) * 1000.0
    else:
        results = []
    timing["rerank_ms"] = int(round(rerank_ms))
    timing["rerank_mode"] = reranker.mode
    return results, timing
