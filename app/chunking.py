"""文本切块：把语料切成可检索的 chunk，是链路的入库前置步骤（P2）。

- 两条路径：Q&A 语料（含【问题】/【答复】标记）走 split_qa，其余走 split_text；
- Q&A 路径保证一条问答不跨块合并，超长时切分并在每块重复问题前缀；
- 通用路径按空行分段、按句末标点切句，块间保留 overlap 重叠；
- 所有 span 的 start/end 都是相对原文的字符偏移，不是块内偏移。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import settings

# 小于 30 字的块信息量过低，容易变成噪声片段，直接丢弃
MIN_CHUNK = 30

_SENT_END = re.compile(r"[。！？!?]+")
_PARA_SEP = re.compile(r"\n[ \t\r]*\n+")
_FILENAME_PREFIX = re.compile(r"^\d+_")

_QA_QUESTION = "【问题】"
_QA_ANSWER = "【答复】"
# Q&A 正文常用分号并列多个要点，切句时把分号也算句末，避免要点被割裂
_QA_SENT_END = re.compile(r"[。！？；!?;]+")
# 前缀只取问题前 120 字即可定位上下文；取太长会挤占正文的字符预算
_QUESTION_PREFIX_LEN = 120


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """去掉 span 两端的空白字符，返回新的 (start, end)。"""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """按空行切分，返回各非空段落的 (start, end) 列表。"""
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _PARA_SEP.finditer(text):
        if match.start() > pos:
            spans.append((pos, match.start()))
        pos = match.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """在段落范围内按句末标点切句，返回句子的 (start, end) 列表。"""
    spans: list[tuple[int, int]] = []
    for para_start, para_end in _paragraph_spans(text):
        paragraph = text[para_start:para_end]
        cursor = 0
        for match in _SENT_END.finditer(paragraph):
            spans.append((para_start + cursor, para_start + match.end()))
            cursor = match.end()
        if cursor < len(paragraph):
            spans.append((para_start + cursor, para_end))
    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        start, end = _trim_span(text, start, end)
        if end > start:
            trimmed.append((start, end))
    return trimmed


def _split_overlong(
    spans: list[tuple[int, int]], chunk_size: int, overlap: int
) -> list[tuple[int, int]]:
    """把超过 chunk_size 的 span 硬切成带 overlap 的子 span（仅单句超长时才用）。"""
    step = max(1, chunk_size - overlap)
    out: list[tuple[int, int]] = []
    for start, end in spans:
        if end - start <= chunk_size:
            out.append((start, end))
            continue
        pos = start
        while pos < end:
            stop = min(pos + chunk_size, end)
            out.append((pos, stop))
            if stop >= end:
                break
            pos += step
    return out


def _chunk_spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """通用路径的块级 (start, end)：贪心拼句，并按句边界回退 overlap。"""
    sentences = _split_overlong(_sentence_spans(text), chunk_size, overlap)
    total = len(sentences)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < total:
        chunk_start = sentences[index][0]
        chunk_end = sentences[index][1]
        cursor = index + 1
        while cursor < total and sentences[cursor][1] - chunk_start <= chunk_size:
            chunk_end = sentences[cursor][1]
            cursor += 1
        spans.append((chunk_start, chunk_end))
        if cursor >= total:
            break
        # 从块尾回退 overlap 字，但把下一块起点对齐到句子边界，
        # 这样既有重叠、又不会把一句话从中间切断
        threshold = chunk_end - overlap
        next_index = cursor
        for candidate in range(index, cursor):
            if sentences[candidate][1] > threshold:
                next_index = candidate
                break
        if next_index <= index:
            next_index = index + 1
        if next_index > cursor:
            next_index = cursor
        index = next_index
    return spans


def split_text(text: str, chunk_size: int = 350, overlap: int = 80) -> list[str]:
    """通用切块：分段切句后贪心拼块，相邻块按句边界保留 overlap 重叠。

    非 Q&A 语料走这条路径；默认 350/80 取自 settings 的 CHUNK_SIZE/CHUNK_OVERLAP。

    Args:
        text: 原文。
        chunk_size: 单块最大字符数。
        overlap: 相邻块希望保留的重叠字符数。

    Returns:
        去首尾空白、长度不小于 MIN_CHUNK 的文本块列表。
    """
    if not text or not text.strip():
        return []
    result: list[str] = []
    for start, end in _chunk_spans(text, chunk_size, overlap):
        piece = text[start:end].strip()
        if len(piece) < MIN_CHUNK:
            continue
        result.append(piece)
    return result


def _title_from_filename(filename: str) -> str:
    """由文件名生成 title：去掉扩展名与开头两位数字加下划线。"""
    stem = Path(filename).stem
    return _FILENAME_PREFIX.sub("", stem)


def _qa_units(text: str) -> list[tuple[int, int]]:
    """提取问答单元：以空行分隔、以【问题】开头且含【答复】的块的 (start, end)。"""
    units: list[tuple[int, int]] = []
    for start, end in _paragraph_spans(text):
        start, end = _trim_span(text, start, end)
        if end <= start:
            continue
        block = text[start:end]
        if block.startswith(_QA_QUESTION) and _QA_ANSWER in block:
            units.append((start, end))
    return units


def _qa_question(unit: str) -> str:
    """从问答单元里取出问题文本（【问题】与【答复】之间的部分）。"""
    start = unit.find(_QA_QUESTION) + len(_QA_QUESTION)
    end = unit.find(_QA_ANSWER)
    if end == -1:
        end = len(unit)
    return unit[start:end].strip()


def _sentence_spans_in(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """在 [start, end) 内按句末标点切句（Q&A 专用，句末含分号）。"""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _QA_SENT_END.finditer(text, start, end):
        spans.append((cursor, match.end()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    trimmed: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        span_start, span_end = _trim_span(text, span_start, span_end)
        if span_end > span_start:
            trimmed.append((span_start, span_end))
    return trimmed


def _qa_chunk_spans(text: str, max_len: int) -> list[tuple[str, int, int]]:
    """Q&A 分块主逻辑：返回 (块文本, 原文 start, 原文 end)。

    单元不超长时整条成一块；超长时切句拼接，每块都补上问题前缀。
    """
    results: list[tuple[str, int, int]] = []
    for unit_start, unit_end in _qa_units(text):
        unit = text[unit_start:unit_end]
        if len(unit) <= max_len:
            results.append((unit, unit_start, unit_end))
            continue
        question = _qa_question(unit)
        # 超长单元被切成多块后，后半段本身不含问题，必须在每块头部重复问题前缀，
        # 否则检索命中的那块看不出在回答什么问题
        prefix = _QA_QUESTION + question[:_QUESTION_PREFIX_LEN] + "\n" + _QA_ANSWER
        budget = max(1, max_len - len(prefix))
        answer_at = unit.find(_QA_ANSWER)
        body_start = unit_start + (answer_at + len(_QA_ANSWER) if answer_at != -1 else 0)
        sentences = _split_overlong(
            _sentence_spans_in(text, body_start, unit_end), budget, 0
        )
        current: list[tuple[int, int]] = []
        current_len = 0

        def flush() -> None:
            """把已累积的句子拼成一块并清空累积。"""
            nonlocal current, current_len
            if not current:
                return
            body = " ".join(text[a:b].strip() for a, b in current)
            # start/end 取该块正文首句起点与末句终点，均为原文偏移（前缀是合成文本，不计入）
            results.append((prefix + body, current[0][0], current[-1][1]))
            current = []
            current_len = 0

        for seg_start, seg_end in sentences:
            segment = text[seg_start:seg_end].strip()
            if not segment:
                continue
            if current and current_len + len(segment) > budget:
                flush()
            current.append((seg_start, seg_end))
            current_len += len(segment)
        flush()
    return results


def split_qa(text: str, max_len: int = 800) -> list[str]:
    """Q&A 感知切块：一个问答尽量成一块，保证块内问题上下文不丢失。

    默认 800 对应 settings.QA_MAX_CHUNK；单条问答通常几百字，800 能在不切分
    大多数问答的前提下控制单块长度。

    Args:
        text: 原文，需含【问题】/【答复】标记。
        max_len: 单块最大字符数。

    Returns:
        文本块列表，每块都同时含【问题】与【答复】；不含标记时返回 []。
    """
    if not text or _QA_QUESTION not in text:
        return []
    return [chunk for chunk, _, _ in _qa_chunk_spans(text, max_len)]


def load_documents(data_dir) -> list[dict]:
    """读取 data_dir 下所有 .txt，按文件名排序返回文档列表。

    Args:
        data_dir: 语料目录。

    Returns:
        [{"doc": 文件名, "text": 内容}]；目录不存在时返回 []。读取用 UTF-8、忽略解码错误。
    """
    directory = Path(data_dir)
    documents: list[dict] = []
    if not directory.exists():
        return documents
    for path in sorted(directory.glob("*.txt")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        documents.append({"doc": path.name, "text": text})
    return documents


def build_chunks(
    data_dir, chunk_size: int = 350, overlap: int = 80
) -> list[dict]:
    """把语料目录切成 chunk 列表（链路入库的输入）。

    含【问题】标记的文件走 Q&A 路径（split_qa，长度上限取 settings.QA_MAX_CHUNK），
    其余走通用路径（split_text）。两条路径的取舍理由是：Q&A 语料一条就是一个完整
    语义单元，按通用规则切会切断问答；而非 Q&A 长文没有这种结构，按段落切更合适。

    Args:
        data_dir: 语料目录。
        chunk_size: 通用路径的单块最大字符数。
        overlap: 通用路径相邻块的重叠字符数（Q&A 路径不使用）。

    Returns:
        chunk 字典列表，字段为 chunk_id、doc、title、text、start、end；
        其中 start/end 是相对原文的字符偏移（非块内偏移）。
    """
    chunks: list[dict] = []
    for document in load_documents(data_dir):
        filename = document["doc"]
        text = document["text"]
        stem = Path(filename).stem
        title = _title_from_filename(filename)
        # 以是否含【问题】标记区分 Q&A 语料与通用长文
        if _QA_QUESTION in text:
            pieces = _qa_chunk_spans(text, settings.qa_max_chunk)
        else:
            pieces = [
                (text[start:end].strip(), start, end)
                for start, end in _chunk_spans(text, chunk_size, overlap)
            ]
        kept = [piece for piece in pieces if len(piece[0].strip()) >= MIN_CHUNK]
        for order, (piece_text, start, end) in enumerate(kept, start=1):
            chunks.append(
                {
                    "chunk_id": f"{stem}#{order:03d}",
                    "doc": filename,
                    "title": title,
                    "text": piece_text.strip(),
                    "start": start,
                    "end": end,
                }
            )
    return chunks
