from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from scripts.ingest import run as run_ingest

from app import model_status
from app.config import settings
from app.generator import get_generator
from app.retriever import retrieve
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    IndexStatus,
    IngestResponse,
    ModelStatus,
    Trace,
)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] campus-qa-rag skeleton ready")
    yield


app = FastAPI(title="Campus QA RAG MVP", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": "invalid request"})


@app.exception_handler(Exception)
async def _unhandled_handler(request, exc):
    logger.exception("[api] unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


def _index_counts() -> tuple[int, int]:
    chunks_path = Path(settings.storage_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        return 0, 0
    docs: set[str] = set()
    chunks = 0
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = record.get("doc")
            if isinstance(doc, str) and doc:
                docs.add(doc)
    return len(docs), chunks


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    docs, chunks = _index_counts()
    return HealthResponse(
        status="ok",
        embedding=ModelStatus(**model_status.embedding_status()),
        rerank=ModelStatus(**model_status.rerank_status()),
        generation=ModelStatus(**model_status.generation_status()),
        index=IndexStatus(docs=docs, chunks=chunks),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    result = run_ingest()
    return IngestResponse(**result)


def _to_citations(rows: list[dict]) -> list[Citation]:
    citations = []
    for position, row in enumerate(rows, start=1):
        score = row.get("rerank_score")
        if score is None:
            score = row.get("rrf_score", row.get("score", 0.0))
        citations.append(
            Citation(
                id=position,
                doc=row["doc"],
                chunk_id=row["chunk_id"],
                score=float(score),
                snippet=row["text"][:200],
                score_kind=row.get("rerank_score_kind") or "rrf",
            )
        )
    return citations


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    chunks_path = Path(settings.storage_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        raise HTTPException(status_code=503, detail="index not found")

    started = time.perf_counter()
    rows, timing = await retrieve(request.question, request.top_k)
    contexts = [{"text": row["text"]} for row in rows]

    generator = get_generator()
    gen_started = time.perf_counter()
    answer = generator.generate(request.question, contexts)
    gen_ms = int(round((time.perf_counter() - gen_started) * 1000))

    citations = _to_citations(rows)
    total_ms = int(round((time.perf_counter() - started) * 1000))
    trace = Trace(
        total_ms=total_ms,
        embed_ms=int(timing.get("embed_ms", 0)),
        vector_ms=int(timing.get("vector_ms", 0)),
        bm25_ms=int(timing.get("bm25_ms", 0)),
        rerank_ms=int(timing.get("rerank_ms", 0)),
        gen_ms=gen_ms,
        fusion="rrf",
        rerank_mode=timing.get("rerank_mode", "disabled"),
        gen_mode=generator.mode,
        candidates=int(timing.get("candidates", 0)),
        kept=len(citations),
    )
    return AskResponse(answer=answer, citations=citations, trace=trace)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
