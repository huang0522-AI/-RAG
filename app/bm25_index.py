from __future__ import annotations

import logging
import threading
import time

import jieba
from rank_bm25 import BM25Okapi

from app import store

logger = logging.getLogger(__name__)

STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "吗",
    "呢",
    "啊",
    "怎么",
    "什么",
    "多少",
    "请问",
}

_index_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}
_cache_lock = threading.Lock()


def _keep(token: str) -> bool:
    if not token:
        return False
    if token in STOPWORDS:
        return False
    if len(token) >= 2:
        return True
    if len(token) == 1 and token.isascii() and token.isalnum():
        return True
    return False


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in jieba.lcut_for_search(text or "", HMM=True):
        token = raw.strip()
        if _keep(token):
            tokens.append(token)
    return tokens


def _build(storage_dir: str) -> tuple[BM25Okapi | None, list[dict]]:
    chunks = store.read_chunks(storage_dir)
    if not chunks:
        return None, []
    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    if not any(corpus):
        return None, chunks
    return BM25Okapi(corpus, k1=1.5, b=0.75), chunks


def build_index(storage_dir: str) -> None:
    started = time.perf_counter()
    index, chunks = _build(storage_dir)
    with _cache_lock:
        _index_cache[storage_dir] = (index, chunks)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(f"[bm25] indexed chunks={len(chunks)} elapsed={elapsed_ms}ms")


def _get_index(storage_dir: str) -> tuple[BM25Okapi | None, list[dict]]:
    with _cache_lock:
        cached = _index_cache.get(storage_dir)
    if cached is not None:
        return cached
    build_index(storage_dir)
    with _cache_lock:
        return _index_cache.get(storage_dir, (None, []))


def bm25_search(question: str, top_k: int, storage_dir: str) -> list[dict]:
    index, chunks = _get_index(storage_dir)
    if index is None or not chunks:
        return []
    tokens = tokenize(question)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = [
        (position, float(score))
        for position, score in enumerate(scores)
        if float(score) > 0.0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    results: list[dict] = []
    for position, score in ranked[:top_k]:
        chunk = chunks[position]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc": chunk["doc"],
                "title": chunk["title"],
                "text": chunk["text"],
                "score": score,
                "source": "bm25",
            }
        )
    return results


def invalidate_cache() -> None:
    with _cache_lock:
        _index_cache.clear()
