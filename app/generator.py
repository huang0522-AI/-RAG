"""答案生成：online（DeepSeek）→ extractive（抽取式）两档，是链路最后一步（P6）。

- online 不可用或失败时降级为 extractive，直接把最相关原文片段拼成答案；
- 只把 top5 证据拼进 prompt，要求逐句标注 [n] 来源；
- 对外只返回答案文本：key、完整 prompt、上游异常都不进入结果。
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

MODE_ONLINE = "online"
MODE_EXTRACTIVE = "extractive"

_SENTENCE = re.compile(r"[^。！？!?]+[。！？!?]?")
# 回答长度上限，与 prompt 规则 4 一致（单位：字）
MAX_ANSWER_CHARS = 200
# 抽取式取前 3 条候选、每条取前 2 句
EXTRACTIVE_TOP = 3
EXTRACTIVE_SENTENCES = 2


def _first_sentences(text: str, count: int) -> str:
    """取文本前 count 句（按句末标点切分）。"""
    sentences = [
        piece.strip() for piece in _SENTENCE.findall(text or "") if piece.strip()
    ]
    return "".join(sentences[:count])


def build_prompt(question: str, contexts: list[dict]) -> str:
    """把问题与证据片段拼成提示词；四条规则各防一类问题。

    规则 1 防编造、规则 2 保证可溯源、规则 3 防越界硬答、规则 4 防啰嗦。

    Args:
        question: 用户问题。
        contexts: 证据片段列表，每项取 text 字段。

    Returns:
        prompt 文本。
    """
    lines = []
    for position, context in enumerate(contexts, start=1):
        lines.append(f"[{position}] {context.get('text', '').strip()}")
    block = "\n".join(lines)
    return (
        # 规则1：资料里没有的事实（尤其数字/时间）最易幻觉，先封死编造
        "你是校园问答助手，请严格按以下规则回答：\n"
        "1. 只依据【资料】回答，不得编造资料中没有的数字、时间或事实。\n"
        # 规则2：没有 [n] 就无法把答案对回来源，citation 形同虚设
        "2. 每个事实性陈述后用 [编号] 标注来源，编号对应资料序号。\n"
        # 规则3：越界问题要显式拒答，否则模型倾向推荐资料外内容
        "3. 若【资料】中没有答案，只回答「未找到相关信息」，不要猜测，也不要推荐资料外的内容。\n"
        # 规则4：限制长度，避免生成又长又散
        f"4. 回答简洁，不超过 {MAX_ANSWER_CHARS} 字。\n\n"
        f"【资料】\n{block}\n\n"
        f"【问题】{question}\n"
        "【回答】"
    )


def extractive_answer(question: str, contexts: list[dict]) -> str:
    """抽取式回答：取前 EXTRACTIVE_TOP(3) 条候选、每条前 EXTRACTIVE_SENTENCES(2) 句拼接。

    这里按检索顺序取句子，不做 token 交集/长度加权等二次打分；
    片段保留 [n] 前缀，使其与 citations 的编号对应。

    Args:
        question: 用户问题（当前实现未参与选句，仅为签名一致）。
        contexts: 证据片段列表。

    Returns:
        拼接后的答案；无可用片段时返回「未找到相关信息」。
    """
    parts = []
    for position, context in enumerate(contexts[:EXTRACTIVE_TOP], start=1):
        snippet = _first_sentences(context.get("text", ""), EXTRACTIVE_SENTENCES)
        if snippet:
            # 保留 [n] 前缀，让抽取式回答也能对回 citations
            parts.append(f"[{position}] {snippet}")
    if not parts:
        return "未找到相关信息"
    return " ".join(parts)


class Generator:
    """两档生成器：有 key 走 online，否则走 extractive。"""

    def __init__(self):
        self._key = settings.deepseek_api_key.strip()
        self._base = settings.deepseek_base_url.rstrip("/")
        self._model = settings.deepseek_model
        # mode 由 key 是否存在决定，保证 /health 与 trace 反映真实档位
        self._mode = MODE_ONLINE if self._key else MODE_EXTRACTIVE
        print(f"[generator] mode={self._mode} model={self._model}")

    @property
    def mode(self) -> str:
        """当前生成模式（online/extractive）。"""
        return self._mode

    def generate(self, question: str, contexts: list[dict]) -> str:
        """生成答案：online 优先，失败则切到 extractive 并保持降级。

        Args:
            question: 用户问题。
            contexts: 证据片段列表 [{text}]，允许为空（例如检索无命中）。

        Returns:
            答案文本；key、prompt、上游异常都不会出现在返回值里。
        """
        if self._mode == MODE_ONLINE:
            try:
                return self._generate_online(question, contexts)
            except Exception:
                # 失败不抛给调用方：记录异常并永久降级为抽取式
                logger.exception(
                    "[generator] online generation failed, falling back to extractive"
                )
                self._mode = MODE_EXTRACTIVE
        # contexts 为空也不短路：交给规则3，让模型回答「未找到相关信息」
        return extractive_answer(question, contexts)

    def _generate_online(self, question: str, contexts: list[dict]) -> str:
        """调用 DeepSeek 兼容 /chat/completions（temperature=0、max_tokens=300、超时 30s）。"""
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
        # 失败重试 1 次（range(2)）：网络抖动可自愈；仍失败则抛出，由上层降级
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
    """进程内单例（lru_cache）。"""
    return Generator()
