# 校园RAG · Campus RAG MVP

> 面向校园场景的检索增强问答：**问题 → Embedding → 混合检索 → 精排 → 生成**，每条答案标注来源，并返回可观测的链路耗时。

一个刻意做窄的 RAG 实现。只保留一条可解释的检索链路，不做路由、不做多轮、不做 Agent 编排。目标不是功能多，而是每一跳都能被单独验证、单独替换、单独解释。

---

## 它解决什么

校园信息散落在招生简章、生活指南、收费文件、专业介绍等多份文档里。通用大模型对「A 食堂早餐几点开始」「国家奖学金多少钱」这类问题，要么答不知道，要么编造。

本项目把文档切块建索引，用**向量召回 + 关键词召回**双路取候选，融合后精排，最后约束大模型只依据检索结果作答，并标出来源片段。

---

## 链路

```
question
  │
  ├─[A] 向量路：Embedding → Chroma (cosine) → top 20
  │
  └─[B] 关键词路：jieba 分词 → BM25 → top 20
        │
        └─[C] RRF 融合 (k=60，两路等权) → top 20
              │
              └─[D] 精排 (LLM listwise / CrossEncoder / API) → top 5
                    │
                    └─[E] 只把 top 5 拼进 prompt → LLM 生成
                          │
                          └─ answer + citations + trace
```

`[A]` 与 `[B]` 并行执行，耗时在 `trace` 中分开记录。

---

## 快速开始

### 1. 依赖

```bash
python3 -m pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

最少需要：

```ini
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

> 注意：`DEEPSEEK_BASE_URL` 必须是 `https://api.deepseek.com`，`platform.deepseek.com` 是网页控制台，不是 API 端点。

### 3. 建索引

```bash
python3 -m scripts.ingest
```

### 4. 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000/`。

---

## 接口

### `GET /health`

```json
{
  "status": "ok",
  "embedding": {"model": "hashing-blake2b", "mode": "hashing"},
  "rerank": {"model": "deepseek-chat", "mode": "llm"},
  "generation": {"model": "deepseek-chat", "mode": "online"},
  "index": {"docs": 5, "chunks": 29}
}
```

三个 `mode` 都反映**实际生效**的后端，不是配置里的期望值。

### `POST /ingest`

```json
{"rebuilt": true, "docs": 5, "chunks": 29, "elapsed_ms": 120}
```

### `POST /ask`

```bash
curl -s -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"A食堂早餐几点开始？"}'
```

```json
{
  "answer": "A 食堂早餐开放时间为 07:00—09:00。[1]",
  "citations": [
    {
      "id": 1,
      "doc": "02_校园生活指南.txt",
      "chunk_id": "02_校园生活指南#001",
      "score": 1.0,
      "score_kind": "rank",
      "snippet": "A 食堂位于学生公寓 3 栋旁，开放时间为早餐 07:00—09:00……"
    }
  ],
  "trace": {
    "total_ms": 1285,
    "embed_ms": 0,
    "vector_ms": 12,
    "bm25_ms": 1,
    "rerank_ms": 781,
    "gen_ms": 492,
    "fusion": "rrf",
    "rerank_mode": "llm",
    "gen_mode": "online",
    "candidates": 26,
    "kept": 5
  }
}
```

错误码：空问题 → `400`；索引未构建 → `503`；异常 → `500`（不回传堆栈）。

---

## 后端矩阵（本项目最重要的一处设计）

三个环节都可以独立替换，且降级状态在 `/health` 与 `trace` 中可见，**不做静默降级**。

| 环节 | 可选后端 | 当前默认 | 说明 |
|---|---|---|---|
| Embedding | `api` / `local` / `hashing` | `hashing` | 运行环境无法安装 torch，默认用确定性哈希向量保证链路完整 |
| Rerank | `llm` / `api` / `local` / `disabled` | `llm` | 用 DeepSeek 做 listwise 重排，返回名次分而非概率 |
| Generation | `online` / `extractive` | `online` | 无 key 或调用失败时退回抽取式回答，答案仍带引用 |

### 为什么默认是 hashing

`hashing` 后端是 **字符 2-gram / 3-gram + blake2b 哈希 + log1p + L2 归一化**，是确定性的，但不具备语义能力。它的作用是在没有 torch、下载模型受限的环境里，保证四段链路仍然完整可跑、可测试、可演示。

要换成真实语义模型：

```ini
EMBEDDING_BACKEND=api
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
EMBEDDING_API_MODEL=BAAI/bge-m3
```

或本地安装 `sentence-transformers` 后设 `EMBEDDING_BACKEND=local`。

**更换后端后必须重跑 `python3 -m scripts.ingest`**，因为向量维度会变。

### 精排为什么可以用 LLM

`bge-reranker` 需要 torch，本环境装不上。`llm` 后端让模型只输出一个 JSON 排序数组，不生成答案，避免把重排变成二次生成。代价是每次多一次约 0.5–1.5 秒的调用，`trace.rerank_ms` 会如实记录。

---

## 评测

```bash
python3 -m eval.run_eval
```

- 12 道题，覆盖 5 份文档，逐题检查关键词与引用文档；
- 2 道越界题，只人工确认是否拒答；
- 满分 12，通过线 9.6（80%）；
- 结果写入 `eval/last_run.json`，可用于迭代前后对比。

打分只看 HTTP 响应，不直接读索引；不修改题库、不放宽规则。

---

## 设计取舍

**为什么不用 LlamaIndex**
编排层只有四个阶段。裸写之后，每个阶段的输入输出、耗时、降级原因都能直接打印，依赖也更少。面试或答辩时，链路可以逐跳解释，而不是「框架里发生了一些事」。

**为什么用 RRF 而不是加权求和**
向量分与 BM25 分量纲不同，加权需要调参且对语料敏感。RRF 只用名次：`score = Σ 1/(k + rank)`，两路等权，k=60。不需要归一化，也不需要为每份语料重新调权重。

**为什么保留 extractive 降级**
现场演示最常见的失败不是逻辑错误，而是网络和密钥。`extractive` 用 token 重叠度选句，输出仍带引用编号，保证「没有模型也能给出有依据的答案」。

**为什么不做多轮、不改写、不做路由**
这些能力会显著增加链路长度和不可解释性。本项目的目标是证明「检索质量 + 来源可追溯」，不是覆盖所有 RAG 形态。边界写在 `docs/CONTRACT.md` 的禁止项里。

---

## 目录结构

```
app/
  config.py        配置与环境变量
  schemas.py       请求 / 响应模型
  chunking.py      文档切块
  store.py         chunks.jsonl + Chroma
  embedder.py      三档 embedding 后端
  bm25_index.py    jieba + BM25
  reranker.py      四档精排后端
  generator.py     prompt 组装 + online / extractive
  retriever.py     并行召回 + RRF + 精排入口
  main.py          FastAPI 路由
scripts/ingest.py  重建索引
data/              5 份校园语料
eval/              题库、评测脚本、最近一次结果
tests/             冒烟测试
web/index.html     单文件演示页（零外链）
docs/CONTRACT.md   项目契约与 Changelog
```

---

## 已知限制

1. 语料只有 5 份，覆盖范围有限；
2. 默认 `hashing` 后端没有语义能力，同义表达会漏召回；
3. 精排走 LLM，单次请求增加 0.5–1.5 秒；
4. 单进程内存索引，未做并发优化与多 worker；
5. 无鉴权、无限流，仅适合内网或演示；
6. 越界问题的拒答依赖 prompt 约束，不是硬性校验；
7. 未做多轮对话与查询改写。

---

## License

MIT
