from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _to_int(raw: str | None, default: int) -> int:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _to_bool(raw: str | None, default: bool) -> bool:
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    deepseek_model: str = field(
        default_factory=lambda: _env("DEEPSEEK_MODEL", "deepseek-chat")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    embedding_backend: str = field(
        default_factory=lambda: _env("EMBEDDING_BACKEND", "auto")
    )
    embedding_api_base: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1"
        )
    )
    embedding_api_key: str = field(
        default_factory=lambda: _env("EMBEDDING_API_KEY", "")
    )
    embedding_api_model: str = field(
        default_factory=lambda: _env("EMBEDDING_API_MODEL", "BAAI/bge-m3")
    )
    embedding_hash_dim: int = field(
        default_factory=lambda: _to_int(os.getenv("EMBEDDING_HASH_DIM"), 512)
    )
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )
    rerank_backend: str = field(
        default_factory=lambda: _env("RERANK_BACKEND", "auto")
    )
    rerank_api_base: str = field(
        default_factory=lambda: _env(
            "RERANK_API_BASE", "https://api.siliconflow.cn/v1"
        )
    )
    rerank_api_key: str = field(default_factory=lambda: _env("RERANK_API_KEY", ""))
    rerank_api_model: str = field(
        default_factory=lambda: _env(
            "RERANK_API_MODEL", "BAAI/bge-reranker-v2-m3"
        )
    )
    rerank_llm_model: str = field(
        default_factory=lambda: _env("RERANK_LLM_MODEL", "deepseek-chat")
    )
    rerank_llm_enabled: bool = field(
        default_factory=lambda: _to_bool(os.getenv("RERANK_LLM_ENABLED"), True)
    )
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    storage_dir: str = field(default_factory=lambda: _env("STORAGE_DIR", "storage"))
    chunk_size: int = field(
        default_factory=lambda: _to_int(os.getenv("CHUNK_SIZE"), 350)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _to_int(os.getenv("CHUNK_OVERLAP"), 80)
    )
    qa_max_chunk: int = field(
        default_factory=lambda: _to_int(os.getenv("QA_MAX_CHUNK"), 800)
    )
    top_k_vector: int = field(
        default_factory=lambda: _to_int(os.getenv("TOP_K_VECTOR"), 20)
    )
    top_k_bm25: int = field(
        default_factory=lambda: _to_int(os.getenv("TOP_K_BM25"), 20)
    )
    top_k_rerank: int = field(
        default_factory=lambda: _to_int(os.getenv("TOP_K_RERANK"), 5)
    )
    rrf_k: int = field(default_factory=lambda: _to_int(os.getenv("RRF_K"), 60))


settings = Settings()
