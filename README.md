# 校园RAG · 校园问答 RAG MVP

面向现场演示的校园问答检索增强生成服务：一条固定链路、可观测的降级、克制的代码。

## 启动

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 key（把真实值写入 .env，.env 已被 .gitignore 忽略）
cp .env.example .env

# 建索引（首次或语料变更后执行）
python3 -m scripts.ingest

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl -s localhost:8000/health
```

## 链路

```
question
 ├─ 向量路：Embedding（api / local / hashing，当前 hashing） → Chroma cosine → top 20
 └─ 关键词路：jieba lcut_for_search → BM25 → top 20
        │  (两路并行)
        ▼
      RRF 融合(k=60，等权) → top 20
        ▼
      精排（local / api / llm listwise，当前 llm） → top 5
        ▼
      仅 5 段拼入 prompt → DeepSeek 生成
        ▼
      answer + citations + trace
```

## 为什么链路只保留这四段

- **锚点路由**：命中哪份文档由「向量 + BM25」自然涌现，多一层路由会引入额外误判且难以解释。
- **复合问题拆分**：演示语料单跳即可回答，拆分会把 5 步链路放大，trace 也随之难读。
- **多轮记忆**：现场问答是单问单答，记忆会带来状态与串扰，MVP 阶段保持无状态。
- **MCP**：本项目不调用外部工具，接入协议只会增加依赖与不确定性。
- **答案去重**：只有 5 段证据且已经过 RRF 与精排排序，二次去重收益低，还会掩盖真实排序。

## 技术选型

| 层 | 选型 | 说明 |
|----|------|------|
| Web 框架 | FastAPI + uvicorn(8000) | 轻量、pydantic v2 校验、自带 OpenAPI |
| 切块/落盘 | 自研 `chunking` + `chunks.jsonl` | 句界切块、重叠可配置，落盘可读 |
| 向量库 | ChromaDB `PersistentClient` | 单目录持久化，cosine 距离 |
| 关键词 | rank-bm25 + jieba `lcut_for_search` | 纯 Python，无 CPU 推理依赖 |
| 精排 | CrossEncoder / rerank API / LLM listwise | 三档可切换 |
| 生成 | httpx 调 DeepSeek `/chat/completions` | OpenAI 兼容，不引入 SDK |
| 前端 | 原生 HTML/CSS/JS 单文件 | 零外链、零构建 |
| 测试 | pytest | 只测本地逻辑，不触外部服务 |

**为什么不用 LlamaIndex**：链路要可观测——每一段（向量、BM25、RRF、精排、生成）都能单独打印耗时与中间结果；同时希望依赖少、无隐藏抽象，所以用几十行自研编排替代框架。同理未引入 LangChain。

## 三档后端（本项目当前真实状态）

| 环节 | 可选后端 | 当前生效 | 原因 |
|------|----------|----------|------|
| Embedding | `api` / `local` / `hashing` | **hashing** | 本环境无 torch、无 sentence-transformers、无 embedding API key，自动降级到哈希向量（blake2b 字符 n-gram，dim=512）以保证链路完整 |
| Rerank | `local` / `api` / `llm` / `disabled` | **llm** | `DEEPSEEK_API_KEY` 就绪，复用其做 listwise 重排，score 为名次分 `rerank_score_kind=rank` |
| Generation | `online` / `extractive` | **online** | `DEEPSEEK_API_KEY` 就绪；失败自动重试 1 次后降级为抽取式 |

`auto` 的降级顺序：embedding 无自动链（显式配置）；rerank 为 `local → api → llm → disabled`；generation 为 `online → extractive`。降级原因都会写入日志与 `/health`。

## 接口示例

健康检查：

```bash
curl -s localhost:8000/health
```

```json
{"status":"ok","embedding":{"model":"hashing-blake2b","mode":"hashing"},"rerank":{"model":"deepseek-chat","mode":"llm"},"generation":{"model":"deepseek-chat","mode":"online"},"index":{"docs":5,"chunks":29}}
```

重建索引：

```bash
curl -s -X POST localhost:8000/ingest
```

```json
{"rebuilt":true,"docs":5,"chunks":29,"elapsed_ms":2861}
```

提问：

```bash
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"A食堂早餐几点开始？"}'
```

```json
{"answer":"A食堂早餐07:00开始[1][2]。","citations":[{"id":1,"doc":"02_校园生活指南.txt","chunk_id":"02_校园生活指南#001","score":1.0,"snippet":"..."}],"trace":{"total_ms":1839,"embed_ms":0,"vector_ms":14,"bm25_ms":1,"rerank_ms":953,"gen_ms":872,"fusion":"rrf","rerank_mode":"llm","gen_mode":"online","candidates":20,"kept":5}}
```

## 实测结果

四个必演示问题（本机实测，`rerank_mode=llm`、`gen_mode=online`）：

| 问题 | answer | 首条引用 |
|------|--------|----------|
| A食堂早餐几点开始？ | A食堂早餐07:00开始[1][2]。 | 02_校园生活指南.txt |
| 国家奖学金一年多少钱？ | 国家奖学金标准为每人每年 8000 元，一次性发放。[1] | 03_奖助学金与收费.txt |
| 宿舍晚上几点断电？ | 学生公寓周日到周四 23:00 断电，周五、周六 24:00 断电……[1][2][3] | 02_校园生活指南.txt |
| 学校附近哪家火锅店最好吃？ | 未找到相关信息。 | —（无资料命中） |

各阶段平均耗时（4 题 warm 实测，ms）：

| total | embed | vector | bm25 | rerank | gen |
|-------|-------|--------|------|--------|-----|
| ≈1894 | 0 | ≈12 | ≈1 | ≈1030 | ≈851 |

自动评测：`eval/questions.json` 已随仓库提供（12 道断言题 + 2 道越界题），通过线 80%（9.6 / 12）。运行 `python3 -m eval.run_eval`，脚本会把总分、通过率与逐题明细写入 `eval/last_run.json`。

> 本仓库**不附带未经实测的评测分数**。上表是 4 道演示题的人工记录；完整 12 题的总分请在本地跑出后填入，避免把估算值当成结果。

## 已知限制

1. 语料只有 5 份校园文档，覆盖面有限，问题稍偏就可能召回不到。
2. 接口无鉴权，任何能访问端口的人都可以调用 `/ask` 与 `/ingest`。
3. 单进程 uvicorn，没有并发控制与水平扩展，重算索引时不可用。
4. Embedding 目前是 `hashing` 哈希向量，**没有语义能力**，语义命中弱；正确性命中主要靠 BM25 与 LLM 精排兜底。
5. rerank 与生成依赖外部 DeepSeek，网络延迟明显（单题 rerank ≈1s、生成 ≈0.85s），且受外部服务可用性影响。
6. 越界问题（如「附近哪家火锅店最好吃」）靠 prompt 第 3 条规则约束拒答，没有独立的拒答/相关性分类器。
7. `embed_ms` 恒为 0：向量编码耗时并入 `vector_ms`（`vector_search` 未拆分计时）。

## 目录结构

```
app/        config / schemas / chunking / store / embedder / bm25_index / reranker / generator / retriever / model_status / main
scripts/    ingest（CLI 入库）
data/       5 份校园语料（只读）
eval/       run_eval.py + questions.json（12 题题库）+ last_run.json（运行后生成）
tests/      pytest（不触外部服务）
web/        index.html（单文件演示页）
docs/       CONTRACT.md 项目契约
storage/    chunks.jsonl + Chroma 持久化
```

## 现场演示脚本（约 90 秒）

```bash
# 1) 先亮状态（约 5s）
curl -s localhost:8000/health

# 2) 打开演示页提问「A食堂早餐几点开始？」（约 20s，展示 answer/来源/耗时）
#    浏览器打开 http://localhost:8000/

# 3) 问「国家奖学金一年多少钱？」（约 15s，展示引用 03）

# 4) 问越界题「学校附近哪家火锅店最好吃？」（约 15s，展示「未找到相关信息」）

# 5) 跑自动评测看总分（约 35s）
python3 -m eval.run_eval
```
