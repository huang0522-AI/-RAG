"""关键词检索（BM25）：链路的关键词路，负责专有词/数字的精确命中（P4）。

- 分词用 jieba 搜索模式，文档侧与 query 侧共用同一个 tokenize 以保证可比；
- 索引按 storage_dir 缓存在进程内存，语料重建后必须调用 invalidate_cache；
- 过滤掉 score<=0 的候选，避免把与 query 无共同 token 的块带进融合。
"""

from __future__ import annotations

import logging
import threading
import time

import jieba
from rank_bm25 import BM25Okapi

from app import store

logger = logging.getLogger(__name__)

# 只收高频虚词；数字与专有名词（如“8000”“奖学金”）必须保留，
# 否则关键词路会漏掉含关键事实的块
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

# 缓存键含 storage_dir：不同存储目录的索引互不干扰
_index_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}
# 构建可能被多个线程并发触发（经 asyncio.to_thread 调用），加锁保护缓存
_cache_lock = threading.Lock()


def _keep(token: str) -> bool:
    """判断 token 是否保留：去停用词，保留长度 >=2 的词或单字符的英文/数字。"""
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
    """把文本切成 BM25 用的 token 列表（文档侧与 query 侧共用以保证可比）。

    Args:
        text: 原文。

    Returns:
        过滤停用词、按长度规则保留后的 token 列表。
    """
    tokens: list[str] = []
    # 用搜索模式分词：文档写“校园网络”、用户问“校园网”时，精确模式两边切分不一致，
    # 会导致 BM25 token 交集为空；搜索模式同时产出长词与子词，能把两边对齐
    for raw in jieba.lcut_for_search(text or "", HMM=True):
        token = raw.strip()
        if _keep(token):
            tokens.append(token)
    return tokens


def _build(storage_dir: str) -> tuple[BM25Okapi | None, list[dict]]:
    """读取 chunks 并构建 (BM25Okapi, chunks)；无有效语料时返回 (None, ...)。"""
    chunks = store.read_chunks(storage_dir)
    if not chunks:
        return None, []
    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    if not any(corpus):
        return None, chunks
    # k1=1.5、b=0.75 为 BM25 常用默认：k1 控制词频饱和，b 控制文档长度归一
    return BM25Okapi(corpus, k1=1.5, b=0.75), chunks


def build_index(storage_dir: str) -> None:
    """构建索引并写入进程内缓存，打印建索引用时。"""
    started = time.perf_counter()
    index, chunks = _build(storage_dir)
    with _cache_lock:
        _index_cache[storage_dir] = (index, chunks)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(f"[bm25] indexed chunks={len(chunks)} elapsed={elapsed_ms}ms")


def _get_index(storage_dir: str) -> tuple[BM25Okapi | None, list[dict]]:
    """取缓存索引，缺失时惰性构建后再读缓存。"""
    with _cache_lock:
        cached = _index_cache.get(storage_dir)
    if cached is not None:
        return cached
    build_index(storage_dir)
    with _cache_lock:
        return _index_cache.get(storage_dir, (None, []))


def bm25_search(question: str, top_k: int, storage_dir: str) -> list[dict]:
    """BM25 检索 top_k，返回 [{chunk_id, doc, title, text, score, source}]，score 降序。

    Args:
        question: 用户问题。
        top_k: 返回条数上限。
        storage_dir: 索引所在目录（同时作为缓存键）。

    Returns:
        命中列表；索引为空或 query 无有效 token 时返回 []。
    """
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
        # 分数为 0 表示该块与 query 无共同 token，不是候选，直接丢弃
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
    """清空进程内索引缓存；ingest 重建 chunks 后调用，避免继续用旧语料。"""
    with _cache_lock:
        _index_cache.clear()
