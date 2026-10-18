from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    id: int
    doc: str
    chunk_id: str
    score: float
    snippet: str
    score_kind: str


class Trace(BaseModel):
    total_ms: int
    embed_ms: int
    vector_ms: int
    bm25_ms: int
    rerank_ms: int
    gen_ms: int
    fusion: str
    rerank_mode: str
    gen_mode: str
    candidates: int
    kept: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: Trace


class IngestRequest(BaseModel):
    pass


class IngestResponse(BaseModel):
    rebuilt: bool
    docs: int
    chunks: int
    elapsed_ms: int


class ModelStatus(BaseModel):
    model: str
    mode: str


class IndexStatus(BaseModel):
    docs: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    embedding: ModelStatus
    rerank: ModelStatus
    generation: ModelStatus
    index: IndexStatus
