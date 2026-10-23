"""文本向量化：三档后端可切换，是链路向量路的入口（P3）。

- 判定顺序 api → local → hashing → disabled，按可用性自动降级；
- 只向上层暴露 encode()/mode，且 mode 必须返回“实际生效”的后端，禁止写死；
- hashing 是零依赖降级路径：保证链路可跑通，但没有语义/同义召回能力。
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

# 后端标识，直接作为 get_embedder().mode 与 /health 中 embedding.mode 的取值
_BACKEND_API = "api"
_BACKEND_LOCAL = "local"
_BACKEND_HASHING = "hashing"
_BACKEND_DISABLED = "disabled"


def _sentence_transformers_available() -> bool:
    """探测 sentence-transformers 是否可导入（决定 local 后端是否可用）。"""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


class Embedder:
    """可切换后端的向量化器，对外只暴露 mode/dim/active_model 与 encode()。"""

    def __init__(self):
        self._backend = _BACKEND_DISABLED
        self._model = None
        self._active_model = None
        self._dim = None
        self._api_key = settings.embedding_api_key.strip()
        self._hash_dim = max(1, settings.embedding_hash_dim)
        self._select_backend()

    def _select_backend(self) -> None:
        """按配置与可用性选择后端：显式指定优先，auto 时按 api→local→hashing 降级。"""
        requested = (settings.embedding_backend or "auto").strip().lower()
        has_api = bool(self._api_key)
        if requested == _BACKEND_DISABLED:
            self._activate_disabled()
        elif requested == _BACKEND_HASHING:
            self._activate_hashing()
        elif requested == _BACKEND_API:
            if has_api:
                self._activate_api()
            elif _sentence_transformers_available() and self._try_local():
                pass
            else:
                self._activate_hashing(reason="api requested but no key")
        elif requested == _BACKEND_LOCAL:
            if _sentence_transformers_available() and self._try_local():
                pass
            else:
                self._activate_hashing(reason="local requested but unavailable")
        else:
            if has_api:
                self._activate_api()
            elif _sentence_transformers_available() and self._try_local():
                pass
            else:
                self._activate_hashing(reason="auto: no api key, no local model")

    def _try_local(self) -> bool:
        """尝试加载本地 SentenceTransformer；成功返回 True，失败返回 False。"""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(settings.embedding_model, device="cpu")
            dim = model.get_sentence_embedding_dimension()
            self._model = model
            self._backend = _BACKEND_LOCAL
            self._active_model = settings.embedding_model
            self._dim = dim
            print(
                f"[embedder] backend={_BACKEND_LOCAL} "
                f"model={self._active_model} dim={dim}"
            )
            return True
        except Exception:
            logger.exception("[embedder] local backend unavailable")
            return False

    def _activate_api(self) -> None:
        """启用远程 embedding 后端；维度要等首次调用才知道，故先置 None。"""
        self._backend = _BACKEND_API
        self._model = None
        self._active_model = settings.embedding_api_model
        self._dim = None
        print(
            f"[embedder] backend={_BACKEND_API} "
            f"model={self._active_model} dim=unknown"
        )

    def _activate_disabled(self) -> None:
        """关闭向量路（仅供显式配置 disabled 时使用）。"""
        self._backend = _BACKEND_DISABLED
        self._model = None
        self._active_model = None
        self._dim = None
        print(f"[embedder] backend={_BACKEND_DISABLED}")

    def _activate_hashing(self, reason: str | None = None) -> None:
        """降级到 hashing 后端；reason 会被记入日志，便于排查是哪一档触发降级。"""
        self._backend = _BACKEND_HASHING
        self._model = None
        self._active_model = "hashing-blake2b"
        self._dim = self._hash_dim
        if reason:
            logger.warning("[embedder] falling back to hashing: %s", reason)
        print(
            f"[embedder] backend={_BACKEND_HASHING} "
            f"model={self._active_model} dim={self._dim}"
        )

    @property
    def mode(self) -> str:
        """当前实际生效的后端（api/local/hashing/disabled），/health 直接采用此值。"""
        return self._backend

    @property
    def dim(self):
        """当前后端输出的向量维度；api 后端在首次调用前可能为 None。"""
        return self._dim

    @property
    def active_model(self):
        """当前后端实际使用的模型名（hashing 为 hashing-blake2b）。"""
        return self._active_model

    def encode(self, texts) -> list[list[float]]:
        """把文本列表编码为向量列表（与输入等长、同序）。

        Args:
            texts: 待编码文本列表。

        Returns:
            向量列表。api 后端运行中失败会自动降级到 hashing 后重算；disabled 时抛错。

        Raises:
            RuntimeError: 后端为 disabled 时。
        """
        items = list(texts)
        if self._backend == _BACKEND_LOCAL:
            return self._encode_local(items)
        if self._backend == _BACKEND_API:
            try:
                return self._encode_api(items)
            except Exception:
                logger.exception(
                    "[embedder] api backend failed, falling back to hashing"
                )
                self._activate_hashing(reason="api call failed")
                return self._encode_hashing(items)
        if self._backend == _BACKEND_HASHING:
            return self._encode_hashing(items)
        # 无可用后端时明确抛错，而不是回退成随机/零向量冒充 embedding
        raise RuntimeError("embedder is disabled")

    def _encode_local(self, items: list[str]) -> list[list[float]]:
        """用 SentenceTransformer 编码；normalize_embeddings 与向量库的 cosine 距离配套。"""
        vectors = self._model.encode(
            items, normalize_embeddings=True, batch_size=32
        )
        return [[float(value) for value in vector] for vector in vectors]

    def _encode_api(self, items: list[str]) -> list[list[float]]:
        """调用远程 /embeddings，按返回的 index 归位；空文本先过滤。"""
        indexed = [
            (position, text)
            for position, text in enumerate(items)
            if text and text.strip()
        ]
        if not indexed:
            dim = self._dim or self._hash_dim
            return [[0.0] * dim for _ in items]
        vectors: list[list[float] | None] = [None] * len(items)
        dim = self._dim
        for start in range(0, len(indexed), 32):
            batch = indexed[start : start + 32]
            payload = {
                "model": self._active_model,
                "input": [text for _, text in batch],
            }
            data = self._api_post(payload)
            # 远程按批返回且不保证顺序，这里按 index 排序后再归位到原位置
            for item in sorted(data.get("data", []), key=lambda row: row["index"]):
                position = batch[item["index"]][0]
                vector = [float(value) for value in item["embedding"]]
                vectors[position] = vector
                dim = len(vector)
        dim = dim or 0
        self._dim = dim
        # 被过滤掉的空文本用零向量占位，保证返回值与输入等长同序
        for position, vector in enumerate(vectors):
            if vector is None:
                vectors[position] = [0.0] * dim
        return [vector for vector in vectors if vector is not None]

    def _api_post(self, payload: dict) -> dict:
        """POST OpenAI 兼容 /embeddings；超时 30s，失败重试 1 次后抛出。"""
        import httpx

        url = settings.embedding_api_base.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = httpx.post(
                    url, json=payload, headers=headers, timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
        raise last_error if last_error else RuntimeError("embedding api failed")

    def _encode_hashing(self, items: list[str]) -> list[list[float]]:
        """对每条文本计算哈希向量（结果经 LRU 缓存，重复文本不重复计算）。"""
        return [list(self._hash_vector_cached(text)) for text in items]

    @lru_cache(maxsize=8192)
    def _hash_vector_cached(self, text: str) -> tuple[float, ...]:
        """缓存单条文本的哈希向量；缓存键含实例，避免了跨实例串用。"""
        return tuple(self._hash_vector(text))

    def _hash_vector(self, text: str) -> list[float]:
        """计算一条文本的 hashing 向量：n-gram 计数 → log1p → L2 归一化。

        全过程只用 hashlib/math，纯 CPU 且确定性（同文本必得同向量）。
        """
        # 忽略大小写与空白，减少同义形态差异（对 hashing 只做到字符串级归一）
        normalized = "".join(text.lower().split())
        counts = [0.0] * self._hash_dim
        # 同时取 2-gram 与 3-gram：兼顾两字词与三字词/短语
        for size in (2, 3):
            if len(normalized) < size:
                continue
            for position in range(len(normalized) - size + 1):
                gram = normalized[position : position + size]
                # blake2b 稳定哈希 8 字节，再取模落到固定维度，得到稀疏计数
                digest = hashlib.blake2b(
                    gram.encode("utf-8"), digest_size=8
                ).digest()
                counts[int.from_bytes(digest, "big") % self._hash_dim] += 1.0
        # log1p 压缩高频 gram 的计数，避免个别常见 gram 主导整个向量
        weighted = [math.log1p(value) for value in counts]
        # L2 归一化，使其与向量库的 cosine 距离配套使用
        norm = math.sqrt(sum(value * value for value in weighted))
        if norm == 0.0:
            return [0.0] * self._hash_dim
        return [value / norm for value in weighted]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """进程内单例（lru_cache），避免重复加载模型或重复探测后端。"""
    return Embedder()


def reported_mode() -> str:
    """返回当前实际生效的 embedding 模式。"""
    return get_embedder().mode
