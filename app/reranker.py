from __future__ import annotations

import importlib.util
import json
import logging
import math
import re
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

_BACKEND_LOCAL = "local"
_BACKEND_API = "api"
_BACKEND_LLM = "llm"
_BACKEND_DISABLED = "disabled"
_BACKENDS = (_BACKEND_LOCAL, _BACKEND_API, _BACKEND_LLM, _BACKEND_DISABLED)

LLM_POOL = 20
LLM_SNIPPET = 180
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _cross_encoder_available() -> bool:
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


class Reranker:
    def __init__(self):
        self._backend = _BACKEND_DISABLED
        self._model = None
        self._active_model = None
        self._api_key = (
            settings.rerank_api_key.strip() or settings.embedding_api_key.strip()
        )
        self._deepseek_key = settings.deepseek_api_key.strip()
        self._chain = self._build_chain(settings.rerank_backend)
        self._chain_index = -1
        self._advance_to_available()

    def _build_chain(self, requested: str) -> list[str]:
        req = (requested or "auto").strip().lower()
        if req in _BACKENDS:
            return list(_BACKENDS[_BACKENDS.index(req):])
        return list(_BACKENDS)

    def _advance_to_available(self) -> None:
        while self._chain_index + 1 < len(self._chain):
            self._chain_index += 1
            name = self._chain[self._chain_index]
            available, reason = self._activate(name)
            if available:
                return
            logger.warning("[reranker] backend '%s' unavailable: %s", name, reason)

    def _activate(self, name: str) -> tuple[bool, str | None]:
        if name == _BACKEND_LOCAL:
            return self._activate_local()
        if name == _BACKEND_API:
            return self._activate_api()
        if name == _BACKEND_LLM:
            return self._activate_llm()
        self._force_disabled()
        return True, None

    def _activate_local(self) -> tuple[bool, str | None]:
        if not _cross_encoder_available():
            return False, "sentence-transformers not importable"
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                settings.reranker_model, max_length=512, device="cpu"
            )
            self._model = model
            self._backend = _BACKEND_LOCAL
            self._active_model = settings.reranker_model
            print(
                f"[reranker] backend={_BACKEND_LOCAL} model={self._active_model}"
            )
            return True, None
        except Exception as exc:
            logger.exception("[reranker] local backend load failed")
            return False, f"load failed: {exc}"

    def _activate_api(self) -> tuple[bool, str | None]:
        if not self._api_key:
            return False, "no RERANK_API_KEY / EMBEDDING_API_KEY"
        self._model = None
        self._backend = _BACKEND_API
        self._active_model = settings.rerank_api_model
        print(f"[reranker] backend={_BACKEND_API} model={self._active_model}")
        return True, None

    def _activate_llm(self) -> tuple[bool, str | None]:
        if not settings.rerank_llm_enabled:
            return False, "RERANK_LLM_ENABLED=false"
        if not self._deepseek_key:
            return False, "no DEEPSEEK_API_KEY"
        self._model = None
        self._backend = _BACKEND_LLM
        self._active_model = settings.rerank_llm_model
        print(f"[reranker] backend={_BACKEND_LLM} model={self._active_model}")
        return True, None

    def _force_disabled(self) -> None:
        self._model = None
        self._backend = _BACKEND_DISABLED
        self._active_model = None
        print(f"[reranker] backend={_BACKEND_DISABLED}")

    @property
    def mode(self) -> str:
        return self._backend

    @property
    def active_model(self):
        return self._active_model

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        while True:
            try:
                if self._backend == _BACKEND_LOCAL:
                    return self._rerank_local(query, candidates, top_k)
                if self._backend == _BACKEND_API:
                    return self._rerank_api(query, candidates, top_k)
                if self._backend == _BACKEND_LLM:
                    return self._rerank_llm(query, candidates, top_k)
                return self._rerank_disabled(candidates, top_k)
            except Exception:
                logger.exception(
                    "[reranker] backend '%s' failed at runtime, degrading",
                    self._backend,
                )
                previous = self._chain_index
                self._advance_to_available()
                if self._chain_index == previous:
                    self._force_disabled()

    def _rerank_disabled(self, candidates: list[dict], top_k: int) -> list[dict]:
        results = []
        for candidate in candidates[:top_k]:
            row = dict(candidate)
            row["rerank_score"] = None
            row["rerank_score_kind"] = "rank"
            results.append(row)
        return results

    def _rerank_local(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        pairs = [(query, candidate.get("text", "")) for candidate in candidates]
        scores = self._model.predict(pairs)
        scored = []
        for candidate, raw in zip(candidates, scores):
            row = dict(candidate)
            row["rerank_score"] = 1.0 / (1.0 + math.exp(-float(raw)))
            row["rerank_score_kind"] = "probability"
            scored.append(row)
        scored.sort(key=lambda row: row["rerank_score"], reverse=True)
        return scored[:top_k]

    def _rerank_api(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        import httpx

        url = settings.rerank_api_base.rstrip("/") + "/rerank"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._active_model,
            "query": query,
            "documents": [candidate.get("text", "") for candidate in candidates],
            "top_n": top_k,
            "return_documents": False,
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        results = response.json().get("results", [])
        scored = []
        for item in results:
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue
            row = dict(candidates[index])
            row["rerank_score"] = float(item.get("relevance_score", 0.0))
            row["rerank_score_kind"] = "relevance"
            scored.append(row)
        if not scored:
            raise ValueError("rerank api returned no usable results")
        scored.sort(key=lambda row: row["rerank_score"], reverse=True)
        return scored[:top_k]

    def _rerank_llm(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        pool = candidates[:LLM_POOL]
        listing = "\n".join(
            f"{position}. {candidate.get('text', '')[:LLM_SNIPPET]}"
            for position, candidate in enumerate(pool, start=1)
        )
        prompt = (
            "你是检索结果重排器，只输出 JSON，不要输出任何解释或答案。\n"
            f"问题：{query}\n"
            f"候选（编号从 1 开始）：\n{listing}\n"
            f"请按与问题的相关性从高到低输出候选编号，最多 {top_k} 个。\n"
            '只输出形如 {"ranking":[3,7,1]} 的 JSON。'
        )
        content = self._llm_chat(prompt)
        order = self._parse_ranking(content, len(pool))
        if not order:
            order = list(range(len(pool)))
        ordered = order[:top_k]
        total = max(1, len(order))
        results = []
        for rank, index in enumerate(ordered, start=1):
            row = dict(pool[index])
            row["rerank_score"] = 1.0 - (rank - 1) / total
            row["rerank_score_kind"] = "rank"
            results.append(row)
        return results

    def _parse_ranking(self, content: str, pool_size: int) -> list[int]:
        text = content.strip()
        match = _FENCE.search(text)
        if match:
            text = match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        ranking = payload.get("ranking")
        if not isinstance(ranking, list):
            return []
        order: list[int] = []
        for item in ranking:
            if not isinstance(item, int):
                continue
            index = item - 1
            if 0 <= index < pool_size and index not in order:
                order.append(index)
        return order

    def _llm_chat(self, prompt: str) -> str:
        import httpx

        url = settings.deepseek_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self._deepseek_key}"}
        payload = {
            "model": self._active_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 120,
            "stream": False,
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=20.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
