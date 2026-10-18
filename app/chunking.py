from __future__ import annotations

import re
from pathlib import Path

from app.config import settings

MIN_CHUNK = 30

_SENT_END = re.compile(r"[。！？!?]+")
_PARA_SEP = re.compile(r"\n[ \t\r]*\n+")
_FILENAME_PREFIX = re.compile(r"^\d+_")

_QA_QUESTION = "【问题】"
_QA_ANSWER = "【答复】"
_QA_SENT_END = re.compile(r"[。！？；!?;]+")
_QUESTION_PREFIX_LEN = 120


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
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
    stem = Path(filename).stem
    return _FILENAME_PREFIX.sub("", stem)


def _qa_units(text: str) -> list[tuple[int, int]]:
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
    start = unit.find(_QA_QUESTION) + len(_QA_QUESTION)
    end = unit.find(_QA_ANSWER)
    if end == -1:
        end = len(unit)
    return unit[start:end].strip()


def _sentence_spans_in(text: str, start: int, end: int) -> list[tuple[int, int]]:
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
    results: list[tuple[str, int, int]] = []
    for unit_start, unit_end in _qa_units(text):
        unit = text[unit_start:unit_end]
        if len(unit) <= max_len:
            results.append((unit, unit_start, unit_end))
            continue
        question = _qa_question(unit)
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
            nonlocal current, current_len
            if not current:
                return
            body = " ".join(text[a:b].strip() for a, b in current)
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
    if not text or _QA_QUESTION not in text:
        return []
    return [chunk for chunk, _, _ in _qa_chunk_spans(text, max_len)]


def load_documents(data_dir) -> list[dict]:
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
    chunks: list[dict] = []
    for document in load_documents(data_dir):
        filename = document["doc"]
        text = document["text"]
        stem = Path(filename).stem
        title = _title_from_filename(filename)
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
