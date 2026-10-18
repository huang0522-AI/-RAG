from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
LAST_RUN_PATH = EVAL_DIR / "last_run.json"

MUST_CONTAIN_SCORE = 0.7
CITATIONS_SCORE = 0.3
PASS_RATIO = 0.8
STAGES = ("total_ms", "embed_ms", "vector_ms", "bm25_ms", "rerank_ms", "gen_ms")


def load_questions() -> list[dict]:
    if not QUESTIONS_PATH.exists():
        print(f"[eval] missing question bank: {QUESTIONS_PATH}")
        print("[eval] 请先提供 eval/questions.json 后再运行。")
        sys.exit(2)
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("questions", [])
    return data


def is_out_of_scope(item: dict) -> bool:
    if item.get("out_of_scope") is True:
        return True
    return str(item.get("scope", "")).lower() == "out_of_scope"


def keyword_groups(item: dict) -> list[list[str]]:
    raw = item.get("must_contain") or []
    groups: list[list[str]] = []
    for entry in raw:
        if isinstance(entry, list):
            groups.append([str(word) for word in entry])
        else:
            groups.append([str(entry)])
    return groups


def score_answer(item: dict, answer: str, citations: list[dict]) -> tuple[float, list[list[str]]]:
    groups = keyword_groups(item)
    missed = [group for group in groups if not any(word in answer for word in group)]
    score = MUST_CONTAIN_SCORE if not missed else 0.0

    wanted = [str(name) for name in (item.get("citations_should_include") or [])]
    docs = [str(c.get("doc", "")) for c in citations]
    if wanted and any(name in doc for name in wanted for doc in docs):
        score += CITATIONS_SCORE
    return score, missed


def ask(client: httpx.Client, base_url: str, question: str) -> tuple[dict | None, int, str]:
    started = time.perf_counter()
    try:
        response = client.post(
            base_url.rstrip("/") + "/ask", json={"question": question}, timeout=60.0
        )
    except Exception as exc:
        return None, int((time.perf_counter() - started) * 1000), f"{type(exc).__name__}: {exc}"
    elapsed = int((time.perf_counter() - started) * 1000)
    if response.status_code != 200:
        return None, elapsed, f"HTTP {response.status_code}"
    try:
        return response.json(), elapsed, ""
    except Exception as exc:
        return None, elapsed, f"bad json: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Campus QA RAG evaluation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    questions = load_questions()
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    health = None
    with httpx.Client() as client:
        try:
            health = client.get(args.base_url.rstrip("/") + "/health", timeout=10.0).json()
        except Exception as exc:
            print(f"[eval] health check failed: {exc}")

        rows = []
        total_score = 0.0
        max_score = 0.0
        scored = 0
        stage_sums = {stage: 0 for stage in STAGES}
        stage_counts = {stage: 0 for stage in STAGES}

        for index, item in enumerate(questions):
            qid = item.get("id", index + 1)
            question = item.get("question", "")
            payload, elapsed_ms, error = ask(client, args.base_url, question)

            if is_out_of_scope(item):
                answer = "" if payload is None else payload.get("answer", "")
                print(f"[eval] (out_of_scope) id={qid} {question} -> {answer!r} {error}")
                rows.append({
                    "id": qid, "question": question, "out_of_scope": True,
                    "answer": answer, "error": error, "elapsed_ms": elapsed_ms,
                    "score": None,
                })
            elif payload is None:
                max_score += 1.0
                scored += 1
                print(f"[eval] id={qid} FAIL request error: {error}")
                rows.append({
                    "id": qid, "question": question, "out_of_scope": False,
                    "answer": "", "error": error, "elapsed_ms": elapsed_ms,
                    "score": 0.0, "missed_groups": [g for g in keyword_groups(item)],
                    "docs": [], "gen_mode": None,
                })
            else:
                answer = payload.get("answer", "")
                citations = payload.get("citations", [])
                trace = payload.get("trace", {})
                score, missed = score_answer(item, answer, citations)
                total_score += score
                max_score += 1.0
                scored += 1
                for stage in STAGES:
                    if stage in trace and isinstance(trace[stage], (int, float)):
                        stage_sums[stage] += trace[stage]
                        stage_counts[stage] += 1
                docs = [c.get("doc", "") for c in citations]
                rows.append({
                    "id": qid, "question": question, "out_of_scope": False,
                    "answer": answer, "error": error, "elapsed_ms": elapsed_ms,
                    "score": score, "missed_groups": missed, "docs": docs,
                    "gen_mode": trace.get("gen_mode"), "trace": trace,
                })

            if index < len(questions) - 1 and args.sleep > 0:
                time.sleep(args.sleep)

    threshold = PASS_RATIO * max_score if max_score else 0.0
    passed = total_score >= threshold
    title = "id | 得分 | 耗时ms | gen_mode | 引用文档 | 未命中关键词组"
    print("\n" + title)
    print("-" * len(title))
    for row in rows:
        if row["out_of_scope"]:
            print(f"{row['id']} | out_of_scope | {row['elapsed_ms']} | - | - | -")
            continue
        docs = ",".join(sorted({d.split("_")[0] for d in row.get("docs", [])}))
        missed = ";".join("/".join(group) for group in row.get("missed_groups", [])) or "-"
        print(
            f"{row['id']} | {row['score']:.1f} | {row['elapsed_ms']} | "
            f"{row.get('gen_mode')} | {docs or '-'} | {missed}"
        )

    avg_stage = {
        stage: round(stage_sums[stage] / stage_counts[stage])
        for stage in STAGES
        if stage_counts[stage]
    }
    print("\n===== 汇总 =====")
    print(f"总分: {total_score:.2f} / {max_score:.2f}")
    print(f"通过率: {(total_score / max_score if max_score else 0) * 100:.1f}%")
    print(f"通过线: {threshold:.2f} (80%)")
    print(f"结果: {'PASS' if passed else 'FAIL'}")
    print(f"各阶段平均耗时(ms): {avg_stage}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "health": health,
        "total_score": round(total_score, 4),
        "max_score": round(max_score, 4),
        "threshold": round(threshold, 4),
        "passed": passed,
        "scored_questions": scored,
        "avg_stage_ms": avg_stage,
        "questions": rows,
    }
    LAST_RUN_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[eval] wrote {LAST_RUN_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
