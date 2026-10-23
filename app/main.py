"""HTTP 入口：/health、/ingest、/ask 与静态页，串起检索→精排→生成（P5/P6）。

- /ask 固定顺序：校验索引 → retrieve（融合 + 精排）→ generate → 组装 citations/trace；
- 错误码：参数非法 400、索引缺失 503、未处理异常 500，均不回传堆栈；
- trace 的分阶段耗时来自各层自测，total_ms 为 /ask 内端到端的墙钟时间。
"""

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
    """进程启动时打印一行就绪日志（不在此加载模型，避免拖慢启动）。"""
    print("[startup] campus-qa-rag skeleton ready")
    yield


app = FastAPI(title="Campus QA RAG MVP", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc):
    """请求体校验失败（如空问题、超 500 字）→ 400。"""
    return JSONResponse(status_code=400, content={"detail": "invalid request"})


@app.exception_handler(Exception)
async def _unhandled_handler(request, exc):
    """兜底异常 → 500；堆栈只进服务端日志，响应体不含任何细节。"""
    logger.exception("[api] unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


def _index_counts() -> tuple[int, int]:
    """从 chunks.jsonl 统计 (docs, chunks)；文件不存在返回 (0, 0)。"""
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
    """GET /health：返回三档后端 mode 与索引规模。"""
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
    """POST /ingest：复用 CLI 入库逻辑重建索引。"""
    result = run_ingest()
    return IngestResponse(**result)


def _to_citations(rows: list[dict]) -> list[Citation]:
    """把检索结果转成 citations，id 从 1 开始，与 prompt 里的 [n] 一一对应。

    score 优先取精排分 rerank_score；精排关闭时为 None，回退取 rrf_score，再兜底 score。
    """
    citations = []
    for position, row in enumerate(rows, start=1):
        score = row.get("rerank_score")
        if score is None:
            score = row.get("rrf_score", row.get("score", 0.0))
        citations.append(
            Citation(
                # id 从 1 开始：与 answer 中的 [n] 对齐，点 [n] 即可定位到第 n 条来源
                id=position,
                doc=row["doc"],
                chunk_id=row["chunk_id"],
                score=float(score),
                snippet=row["text"][:200],
                # 标注分数含义；无精排分时用 "rrf"
                score_kind=row.get("rerank_score_kind") or "rrf",
            )
        )
    return citations


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """POST /ask：检索 → 生成 → 组装 answer/citations/trace。

    Args:
        request: {question, top_k}；question 长度 1–500，top_k 默认 5。

    Returns:
        AskResponse（answer + citations + trace）。

    Raises:
        HTTPException: 索引不存在时返回 503。
    """
    chunks_path = Path(settings.storage_dir) / "chunks.jsonl"
    # 索引不存在直接 503，避免在空索引上白跑一次检索
    if not chunks_path.exists():
        raise HTTPException(status_code=503, detail="index not found")

    # total_ms 从进入 /ask 起算（含检索与精排）到生成结束的端到端墙钟时间
    started = time.perf_counter()
    rows, timing = await retrieve(request.question, request.top_k)
    contexts = [{"text": row["text"]} for row in rows]

    generator = get_generator()
    gen_started = time.perf_counter()
    answer = generator.generate(request.question, contexts)
    # gen_ms 只覆盖生成阶段，与上面的 total_ms 区分
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
    # 静态页挂在根路径：/ 由 StaticFiles(html=True) 返回 index.html，供演示页访问
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
