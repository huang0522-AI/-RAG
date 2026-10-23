"""检索编排：向量路 + 关键词路并行 → RRF 融合 → 精排，是链路的核心检索层（P4/P5）。

- 两路并行：向量编码/查询是阻塞调用，BM25 是纯 CPU，用 asyncio 并发缩短总耗时；
- 融合用 RRF（只看名次、两路等权），避免两路分数量纲不同带来的加权调参；
- retrieve() 是 /ask 唯一允许调用的检索入口，保证「融合 → 精排」的顺序不被绕过。
"""

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
    """向量路：编码 query → Chroma 召回 top_k；不可用或异常时返回 []。

    Args:
        question: 用户问题。
        top_k: 召回条数。

    Returns:
        结果列表（每条带 source="vector"）；embedding disabled 或出错时返回 []，不抛错。
    """
    embedder = get_embedder()
    if embedder.mode == "disabled":
        logger.warning("[retriever] vector search disabled, embedding unavailable")
        return []
    try:
        # encode 与 query 都是同步阻塞调用，放 to_thread 里避免阻塞事件循环
        vectors = await asyncio.to_thread(embedder.encode, [question])
        query_embedding = vectors[0]
        rows = await asyncio.to_thread(
            store.query_vectors, query_embedding, top_k, settings.storage_dir
        )
    except Exception:
        # 向量路失败只降级为自身空结果，不影响关键词路
        logger.exception("[retriever] vector search failed")
        return []
    for row in rows:
        row["source"] = "vector"
    return rows


async def _timed_vector(question: str, top_k: int) -> tuple[list[dict], float]:
    """执行向量路并计时；异常时返回空结果与已耗时。"""
    started = time.perf_counter()
    try:
        rows = await vector_search(question, top_k)
    except Exception:
        logger.exception("[retriever] vector leg failed")
        rows = []
    return rows, (time.perf_counter() - started) * 1000.0


async def _timed_bm25(question: str, top_k: int) -> tuple[list[dict], float]:
    """在线程中执行 BM25 并计时；异常时返回空结果与已耗时。"""
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
    """两路并行检索后做 RRF 融合，返回降序结果与分阶段计时。

    RRF: score(d) = Σ 1/(k + rank)，rank 从 1 开始（1 为该路第 1 名）；
    k=60 起平滑作用，压制单路靠前名次的权重，让两路贡献更均衡。

    Args:
        question: 用户问题。
        top_k: 融合后返回条数。

    Returns:
        (结果列表, timing)；timing 含 embed_ms/vector_ms/bm25_ms/candidates。
    """
    # gather 让向量路与 BM25 路真正并行，总耗时约取二者较大值而非相加
    (vector_rows, vector_ms), (bm25_rows, bm25_ms) = await asyncio.gather(
        _timed_vector(question, top_k),
        _timed_bm25(question, top_k),
    )

    k = settings.rrf_k
    fused: dict[str, dict] = {}
    # 用 chunk_id 去重合并：同一块被两路同时召回时 RRF 分数叠加
    for rank, row in enumerate(vector_rows, start=1):
        entry = fused.setdefault(
            row["chunk_id"],
            {
                "chunk_id": row["chunk_id"],
                "doc": row["doc"],
                "title": row["title"],
                "text": row["text"],
                "rrf_score": 0.0,
                # 保留两路名次：调试/答辩时要能说清「这条是靠哪一路召回的」
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
    """检索入口：混合检索取 20 条候选，再精排取 top_k。

    这是 /ask 唯一允许调用的检索函数，保证链路顺序（融合 → 精排）不被绕过。

    Args:
        question: 用户问题。
        top_k: 精排后返回条数。

    Returns:
        (结果列表, timing)；timing 额外含 rerank_ms 与 rerank_mode。
    """
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
        # 候选为空时跳过精排，但仍回填 rerank_mode 供 /health 与 trace 使用
        results = []
    timing["rerank_ms"] = int(round(rerank_ms))
    timing["rerank_mode"] = reranker.mode
    return results, timing
