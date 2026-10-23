"""接口数据模型（pydantic v2），字段名与 docs/CONTRACT.md 第四节一一对应。

- Citation / Trace / AskResponse 对应 POST /ask；
- IngestResponse 对应 POST /ingest；HealthResponse 对应 GET /health；
- 所有 *_ms 字段单位均为毫秒，且为整数。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """单条来源；id 从 1 起，与 answer 中的 [n] 对应。"""

    id: int
    doc: str
    chunk_id: str
    score: float
    snippet: str
    # 分数含义标注：rank（名次分）/ probability / relevance / rrf
    score_kind: str


class Trace(BaseModel):
    """本次 /ask 的分阶段计时与档位快照；各 *_ms 均为毫秒整数。"""

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
    """提问请求：question 1–500 字、top_k 1–10（默认 5）。

    top_k 上限 10：单次答案只展示少量证据，给太大既稀释答案又拉高精排/生成成本。
    """

    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """拒绝纯空白问题（空问题按契约返回 400）。"""
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    """提问响应：答案 + 来源 + 链路 trace。"""

    answer: str
    citations: list[Citation]
    trace: Trace


class IngestRequest(BaseModel):
    """入库请求：无参数（每次全量重建索引）。"""

    pass


class IngestResponse(BaseModel):
    """入库响应；elapsed_ms 为毫秒。"""

    rebuilt: bool
    docs: int
    chunks: int
    elapsed_ms: int


class ModelStatus(BaseModel):
    """单个后端的模型名与当前 mode（必须是实际生效值）。"""

    model: str
    mode: str


class IndexStatus(BaseModel):
    """索引规模：文档数与块数。"""

    docs: int
    chunks: int


class HealthResponse(BaseModel):
    """健康检查响应：三个后端状态 + 索引规模。"""

    status: str
    embedding: ModelStatus
    rerank: ModelStatus
    generation: ModelStatus
    index: IndexStatus
