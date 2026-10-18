from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

_BACKEND_API = "api"
_BACKEND_LOCAL = "local"
_BACKEND_HASHING = "hashing"
_BACKEND_DISABLED = "disabled"


def _sentence_transformers_available() -> bool:
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


class Embedder:
    def __init__(self):
        self._backend = _BACKEND_DISABLED
        self._model = None
        self._active_model = None
        self._dim = None
        self._api_key = settings.embedding_api_key.strip()
        self._hash_dim = max(1, settings.embedding_hash_dim)
        self._select_backend()

    def _select_backend(self) -> None:
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
        self._backend = _BACKEND_API
        self._model = None
        self._active_model = settings.embedding_api_model
        self._dim = None
        print(
            f"[embedder] backend={_BACKEND_API} "
            f"model={self._active_model} dim=unknown"
        )

    def _activate_disabled(self) -> None:
        self._backend = _BACKEND_DISABLED
        self._model = None
        self._active_model = None
        self._dim = None
        print(f"[embedder] backend={_BACKEND_DISABLED}")

    def _activate_hashing(self, reason: str | None = None) -> None:
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
        return self._backend

    @property
    def dim(self):
        return self._dim

    @property
    def active_model(self):
        return self._active_model

    def encode(self, texts) -> list[list[float]]:
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
        raise RuntimeError("embedder is disabled")

    def _encode_local(self, items: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            items, normalize_embeddings=True, batch_size=32
        )
        return [[float(value) for value in vector] for vector in vectors]

    def _encode_api(self, items: list[str]) -> list[list[float]]:
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
            for item in sorted(data.get("data", []), key=lambda row: row["index"]):
                position = batch[item["index"]][0]
                vector = [float(value) for value in item["embedding"]]
                vectors[position] = vector
                dim = len(vector)
        dim = dim or 0
        self._dim = dim
        for position, vector in enumerate(vectors):
            if vector is None:
                vectors[position] = [0.0] * dim
        return [vector for vector in vectors if vector is not None]

    def _api_post(self, payload: dict) -> dict:
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
        return [list(self._hash_vector_cached(text)) for text in items]

    @lru_cache(maxsize=8192)
    def _hash_vector_cached(self, text: str) -> tuple[float, ...]:
        return tuple(self._hash_vector(text))

    def _hash_vector(self, text: str) -> list[float]:
        normalized = "".join(text.lower().split())
        counts = [0.0] * self._hash_dim
        for size in (2, 3):
            if len(normalized) < size:
                continue
            for position in range(len(normalized) - size + 1):
                gram = normalized[position : position + size]
                digest = hashlib.blake2b(
                    gram.encode("utf-8"), digest_size=8
                ).digest()
                counts[int.from_bytes(digest, "big") % self._hash_dim] += 1.0
        weighted = [math.log1p(value) for value in counts]
        norm = math.sqrt(sum(value * value for value in weighted))
        if norm == 0.0:
            return [0.0] * self._hash_dim
        return [value / norm for value in weighted]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def reported_mode() -> str:
    return get_embedder().mode
