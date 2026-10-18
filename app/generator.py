from __future__ import annotations

import logging
import re
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

MODE_ONLINE = "online"
MODE_EXTRACTIVE = "extractive"

_SENTENCE = re.compile(r"[^。！？!?]+[。！？!?]?")
MAX_ANSWER_CHARS = 200
EXTRACTIVE_TOP = 3
EXTRACTIVE_SENTENCES = 2


def _first_sentences(text: str, count: int) -> str:
    sentences = [
        piece.strip() for piece in _SENTENCE.findall(text or "") if piece.strip()
    ]
    return "".join(sentences[:count])


def build_prompt(question: str, contexts: list[dict]) -> str:
    lines = []
    for position, context in enumerate(contexts, start=1):
        lines.append(f"[{position}] {context.get('text', '').strip()}")
    block = "\n".join(lines)
    return (
        "你是校园问答助手，请严格按以下规则回答：\n"
        "1. 只依据【资料】回答，不得编造资料中没有的数字、时间或事实。\n"
        "2. 每个事实性陈述后用 [编号] 标注来源，编号对应资料序号。\n"
        "3. 若【资料】中没有答案，只回答「未找到相关信息」，不要猜测，也不要推荐资料外的内容。\n"
        f"4. 回答简洁，不超过 {MAX_ANSWER_CHARS} 字。\n\n"
        f"【资料】\n{block}\n\n"
        f"【问题】{question}\n"
        "【回答】"
    )


def extractive_answer(question: str, contexts: list[dict]) -> str:
    parts = []
    for position, context in enumerate(contexts[:EXTRACTIVE_TOP], start=1):
        snippet = _first_sentences(context.get("text", ""), EXTRACTIVE_SENTENCES)
        if snippet:
            parts.append(f"[{position}] {snippet}")
    if not parts:
        return "未找到相关信息"
    return " ".join(parts)


class Generator:
    def __init__(self):
        self._key = settings.deepseek_api_key.strip()
        self._base = settings.deepseek_base_url.rstrip("/")
        self._model = settings.deepseek_model
        self._mode = MODE_ONLINE if self._key else MODE_EXTRACTIVE
        print(f"[generator] mode={self._mode} model={self._model}")

    @property
    def mode(self) -> str:
        return self._mode

    def generate(self, question: str, contexts: list[dict]) -> str:
        if self._mode == MODE_ONLINE:
            try:
                return self._generate_online(question, contexts)
            except Exception:
                logger.exception(
                    "[generator] online generation failed, falling back to extractive"
                )
                self._mode = MODE_EXTRACTIVE
        return extractive_answer(question, contexts)

    def _generate_online(self, question: str, contexts: list[dict]) -> str:
        import httpx

        prompt = build_prompt(question, contexts)
        url = self._base + "/chat/completions"
        headers = {"Authorization": f"Bearer {self._key}"}
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
            "stream": False,
        }
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = httpx.post(
                    url, json=payload, headers=headers, timeout=30.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_error = exc
        raise last_error if last_error else RuntimeError("deepseek request failed")


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return Generator()
