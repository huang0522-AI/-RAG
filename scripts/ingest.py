"""索引入库 CLI：读语料 → 切块 → 落盘 → 建向量/BM25 索引（P2）。

- 幂等：chunks.jsonl 覆盖写、向量索引先删 collection 再建，重复执行结果一致；
- embedder disabled 时仍写 chunks.jsonl（关键词路可用），只跳过向量写入；
- 退出码：正常返回 0；外部模型不可用由各层内部降级处理，不中断入库。
"""

from __future__ import annotations

import argparse
import time

from app import store
from app.chunking import build_chunks, load_documents
from app.config import settings


def run(rebuild: bool = False) -> dict:
    """执行一次入库，返回 {rebuilt, docs, chunks, elapsed_ms}。

    Args:
        rebuild: 保留参数；当前实现每次都全量重建，取值不影响结果。

    Returns:
        入库统计字典；elapsed_ms 为毫秒。
    """
    started = time.perf_counter()
    load_documents(settings.data_dir)
    chunks = build_chunks(
        settings.data_dir, settings.chunk_size, settings.chunk_overlap
    )
    # 先落盘 chunks.jsonl：它是唯一事实来源，BM25 与向量索引都从这里重建
    store.write_chunks(chunks, settings.storage_dir)

    from app.bm25_index import invalidate_cache

    # 语料变了必须让 BM25 缓存失效，否则会继续用旧索引
    invalidate_cache()

    vector_on = False
    from app.embedder import get_embedder

    embedder = get_embedder()
    if embedder.mode != "disabled" and chunks:
        embeddings = embedder.encode([chunk["text"] for chunk in chunks])
        store.upsert_vectors(chunks, embeddings, settings.storage_dir)
        vector_on = True
    else:
        # embedder 不可用时跳过向量写入，但 chunks 已落盘，关键词路照常可用
        print("[ingest] vector index skipped")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    docs = store.count_docs(chunks)
    print(
        f"[ingest] docs={docs} chunks={len(chunks)} "
        f"vector={'on' if vector_on else 'off'} elapsed={elapsed_ms}ms"
    )
    return {
        "rebuilt": True,
        "docs": docs,
        "chunks": len(chunks),
        "elapsed_ms": elapsed_ms,
    }


def main() -> int:
    """CLI 入口：解析 --rebuild 并执行入库，正常返回退出码 0。"""
    parser = argparse.ArgumentParser(description="Build the campus QA index.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the index")
    args = parser.parse_args()
    run(rebuild=args.rebuild)
    # 入库不因外部模型不可用而失败：退出码固定 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
