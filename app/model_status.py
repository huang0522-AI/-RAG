from __future__ import annotations

from pathlib import Path

from app import embedder, reranker
from app.config import settings
from app.generator import get_generator

UNKNOWN = "unknown"
_embedding_initialized = False
_rerank_initialized = False


def _index_exists() -> bool:
    return (Path(settings.storage_dir) / "chunks.jsonl").exists()


def _init_embedding() -> None:
    global _embedding_initialized
    if _embedding_initialized:
        return
    _embedding_initialized = True
    if not _index_exists():
        return
    try:
        embedder.get_embedder()
    except Exception:
        pass


def _init_rerank() -> None:
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
    _init_embedding()
    if embedder.get_embedder.cache_info().currsize == 0:
        return {"model": settings.embedding_model, "mode": "disabled"}
    instance = embedder.get_embedder()
    return {
        "model": instance.active_model or settings.embedding_model,
        "mode": instance.mode,
    }


def rerank_status() -> dict[str, str]:
    _init_rerank()
    if reranker.get_reranker.cache_info().currsize == 0:
        return {"model": settings.reranker_model, "mode": "disabled"}
    instance = reranker.get_reranker()
    return {
        "model": instance.active_model or settings.reranker_model,
        "mode": instance.mode,
    }


def generation_status() -> dict[str, str]:
    instance = get_generator()
    return {"model": settings.deepseek_model, "mode": instance.mode}
