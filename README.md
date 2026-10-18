# 广东省教育政策问答 RAG

面向现场演示的广东省教育政策问答检索增强生成服务：一条固定链路、可观测的降级、克制的代码，每条答案可追溯来源。

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

## 语料

`data/` 下 6 份广东教育政策问答，共 **68 条问答**（`【问题】…【答复】…` 结构），覆盖学生资助、教师资格与职称、招生考试与升学、就业与应届生身份、基础教育管理：

| 文件 | 问题条数 |
|------|----------|
| 01_学生资助与奖学.txt | 12 |
| 02_教师资格与职称.txt | 15 |
| 03_招生考试与升学.txt | 24 |
| 04_就业与应届生身份.txt | 10 |
| 05_基础教育管理.txt | 6 |
| 06_综合问答.txt | 1 |

切块对 Q&A 做了适配（`split_qa`，`QA_MAX_CHUNK=800`）：一条问答不跨块合并，超长单元被切分时每块都重复问题前缀，保证任何一块都能看出在回答什么问题；当前 6 份语料共切出 **69 个 chunk**。

## 链路

```
question
 ├─ 向量路：Embedding(bge-small-zh-v1.5) → Chroma cosine → top 20
 └─ 关键词路：jieba 分词 → BM25 → top 20
        │  (两路并行)
        ▼
      RRF 融合(k=60，等权) → top 20
        ▼
      CrossEncoder(bge-reranker-v2-m3) 精排 → top 5
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
| 切块/落盘 | 自研 `chunking`（Q&A 感知）+ `chunks.jsonl` | 一条问答一块、可配置重叠，落盘可读 |
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

`auto` 的降级顺序：rerank 为 `local → api → llm → disabled`；generation 为 `online → extractive`。降级原因都会写入日志与 `/health`。

## 接口示例

健康检查：

```bash
curl -s localhost:8000/health
```

```json
{"status":"ok","embedding":{"model":"hashing-blake2b","mode":"hashing"},"rerank":{"model":"deepseek-chat","mode":"llm"},"generation":{"model":"deepseek-chat","mode":"online"},"index":{"docs":6,"chunks":69}}
```

重建索引：

```bash
curl -s -X POST localhost:8000/ingest
```

```json
{"rebuilt":true,"docs":6,"chunks":69,"elapsed_ms":1885}
```

提问（真实响应，snippet 已截断）：

```bash
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"国家励志奖学金证书怎么下载？"}'
```

```json
{
  "answer": "2023-2024学年：访问广东省教育厅通知（https://edu.gd.gov.cn/zwgknew/gsgg/content/post_4644358.html），通过附件名单查询下载电子证书[1]；该链接已于2025年6月12日恢复，点击“旧版入口”蓝字跳转个人信息输入页即可[2]。2024-2025学年：登录粤省事APP或微信小程序，依次点击办事→主题→教育→（更多服务）证书查询→选择2025年份，或直接搜索“证书查询-教育”[1]。",
  "citations": [
    {"id": 1, "doc": "01_学生资助与奖学.txt", "chunk_id": "01_学生资助与奖学#005", "score": 1.0, "snippet": "【问题】怎么找不到2023~24和2024~25两个学年的奖学金证书下载地址…", "score_kind": "rank"},
    {"id": 2, "doc": "01_学生资助与奖学.txt", "chunk_id": "01_学生资助与奖学#001", "score": 0.8, "snippet": "【问题】你好，23-24年国家励志奖学金证书 网址无法打开…", "score_kind": "rank"},
    {"id": 3, "doc": "01_学生资助与奖学.txt", "chunk_id": "01_学生资助与奖学#009", "score": 0.6, "snippet": "【问题】2022-2023学年度国家励志奖学金证书丢失了…", "score_kind": "rank"},
    {"id": 4, "doc": "02_教师资格与职称.txt", "chunk_id": "02_教师资格与职称#008", "score": 0.4, "snippet": "【问题】本人在《2025年广东省国家中小学智慧教育平台应用优质教育教学教研案例征集》…", "score_kind": "rank"},
    {"id": 5, "doc": "01_学生资助与奖学.txt", "chunk_id": "01_学生资助与奖学#012", "score": 0.2, "snippet": "【问题】请问2025年度本科生国家奖学金获得者的名单将会在什么时候公示…", "score_kind": "rank"}
  ],
  "trace": {"total_ms": 1669, "embed_ms": 0, "vector_ms": 12, "bm25_ms": 1, "rerank_ms": 539, "gen_ms": 1117, "fusion": "rrf", "rerank_mode": "llm", "gen_mode": "online", "candidates": 30, "kept": 5}
}
```

## 实测结果

自动评测（`python3 -m eval.run_eval`，12 题 + 2 道 out_of_scope）：

```
总分: 12.00 / 12.00        通过率: 100.0%        结果: PASS
各阶段平均耗时(ms): total 1801 · embed 0 · vector 15 · bm25 1 · rerank 834 · gen 952
```

四个必演示问题（本机实测，`rerank_mode=llm`、`gen_mode=online`）：

| 问题 | answer（摘要） | 首条引用 |
|------|----------------|----------|
| 国家励志奖学金证书怎么下载？ | 含「旧版入口」「粤省事」查询路径 | 01_学生资助与奖学.txt |
| 广东教资认定普通话要几级？ | 二级乙等及以上 | 02_教师资格与职称.txt |
| 集体三等功可以免试专升本吗？ | 需上传「立功登记表」及立功证书 | 03_招生考试与升学.txt |
| 2027年广东省高考报名人数是多少？ | 未找到相关信息（不编造人数） | —（无资料命中） |

## 已知限制

1. 语料为问答摘录，政策有时效性，不能替代官方答复。
2. `hashing` embedding 无语义能力，同义问法依赖 BM25 与 LLM 精排兜底。
3. 单条问答较长（最长约 900 字），精排与生成成本高于短文本。
4. 语料只有 6 份、68 条问答，覆盖面有限，问题稍偏就可能召回不到。
5. 接口无鉴权，任何能访问端口的人都可以调用 `/ask` 与 `/ingest`。
6. 单进程 uvicorn，没有并发控制与水平扩展，重算索引时不可用。
7. rerank 与生成依赖外部 DeepSeek，网络延迟明显（单题 rerank ≈0.8s、生成 ≈1.0s），且受外部服务可用性影响。
8. 越界问题靠 prompt 第 3 条规则约束拒答，没有独立的拒答/相关性分类器。
9. `embed_ms` 恒为 0：向量编码耗时并入 `vector_ms`（`vector_search` 未拆分计时）。

## 目录结构

```
app/        config / schemas / chunking / store / embedder / bm25_index / reranker / generator / retriever / model_status / main
scripts/    ingest（CLI 入库）
data/       6 份教育政策问答语料（只读）
eval/       run_eval.py + questions.json + last_run.json
tests/      pytest（不触外部服务）
web/        index.html（单文件演示页）
docs/       CONTRACT.md 项目契约
storage/    chunks.jsonl + Chroma 持久化
```

## 现场演示脚本（约 90 秒）

```bash
# 1) 先亮状态（约 5s）
curl -s localhost:8000/health

# 2) 打开演示页提问「国家励志奖学金证书怎么下载？」（约 20s，展示 answer/来源/耗时）
#    浏览器打开 http://localhost:8000/

# 3) 问「广东教资认定普通话要几级？」（约 15s，展示引用 02）

# 4) 问越界题「2027年广东省高考报名人数是多少？」（约 15s，展示「未找到相关信息」）

# 5) 跑自动评测看总分（约 35s）
python3 -m eval.run_eval
```
