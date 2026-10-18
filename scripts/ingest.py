from __future__ import annotations

import argparse
import time

from app import store
from app.chunking import build_chunks, load_documents
from app.config import settings


def run(rebuild: bool = False) -> dict:
    started = time.perf_counter()
    load_documents(settings.data_dir)
    chunks = build_chunks(
        settings.data_dir, settings.chunk_size, settings.chunk_overlap
    )
    store.write_chunks(chunks, settings.storage_dir)

    from app.bm25_index import invalidate_cache

    invalidate_cache()

    vector_on = False
    from app.embedder import get_embedder

    embedder = get_embedder()
    if embedder.mode != "disabled" and chunks:
        embeddings = embedder.encode([chunk["text"] for chunk in chunks])
        store.upsert_vectors(chunks, embeddings, settings.storage_dir)
        vector_on = True
    else:
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
    parser = argparse.ArgumentParser(description="Build the campus QA index.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the index")
    args = parser.parse_args()
    run(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
