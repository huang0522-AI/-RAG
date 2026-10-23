"""/health 的状态来源：把三档后端的真实 mode 与索引规模汇总给接口（P1/P2）。

- /health 绝不能触发模型下载：仅在索引存在时惰性初始化一次，用布尔位缓存；
- 未初始化或初始化失败时返回 disabled/unknown，不抛错、不阻塞响应；
- embedding/rerank 的 mode 取实际生效值，generation 的 mode 由 key 决定。
"""

from __future__ import annotations

from pathlib import Path

from app import embedder, reranker
from app.config import settings
from app.generator import get_generator

UNKNOWN = "unknown"
# 用一次性布尔位缓存：避免每次 /health 都重复探测（探测可能触发模型加载）
_embedding_initialized = False
_rerank_initialized = False


def _index_exists() -> bool:
    """判断 chunks.jsonl 是否存在（即索引是否已构建）。"""
    return (Path(settings.storage_dir) / "chunks.jsonl").exists()


def _init_embedding() -> None:
    """在索引存在时惰性初始化 embedding 单例；只做一次，失败静默忽略。"""
    global _embedding_initialized
    if _embedding_initialized:
        return
    _embedding_initialized = True
    # 索引不存在就不探测：此时向量路本就用不到，探测只会拖慢 /health
    if not _index_exists():
        return
    try:
        embedder.get_embedder()
    except Exception:
        pass


def _init_rerank() -> None:
    """在索引存在时惰性初始化 reranker 单例；只做一次，失败静默忽略。"""
    global _rerank_initialized
    if _rerank_initialized:
        return
    _rerank_initialized = True
    if not _index_exists():
        return
    try:
        reranker.get_reranker()
    except Exception:
        pass


def embedding_status() -> dict[str, str]:
    """返回 embedding 的 {model, mode}；尚未初始化时给出 disabled。"""
    _init_embedding()
    if embedder.get_embedder.cache_info().currsize == 0:
        return {"model": settings.embedding_model, "mode": "disabled"}
    instance = embedder.get_embedder()
    return {
        "model": instance.active_model or settings.embedding_model,
        "mode": instance.mode,
    }


def rerank_status() -> dict[str, str]:
    """返回 rerank 的 {model, mode}；尚未初始化时给出 disabled。"""
    _init_rerank()
    if reranker.get_reranker.cache_info().currsize == 0:
        return {"model": settings.reranker_model, "mode": "disabled"}
    instance = reranker.get_reranker()
    return {
        "model": instance.active_model or settings.reranker_model,
        "mode": instance.mode,
    }


def generation_status() -> dict[str, str]:
    """返回 generation 的 {model, mode}（由 key 决定 online/extractive）。"""
    instance = get_generator()
    return {"model": settings.deepseek_model, "mode": instance.mode}
