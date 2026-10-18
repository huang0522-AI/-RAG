# 项目契约

一、唯一允许的链路（任何额外环节都算越界）
question
 ├─ 向量路：Embedding(bge-small-zh-v1.5) → Chroma cosine → top 20
 └─ 关键词路：jieba 分词 → BM25 → top 20
两路必须并行 → RRF 融合(k=60，等权) → top 20 → CrossEncoder(bge-reranker-v2-m3) 精排 → top 5
→ 只把这 5 段拼进 prompt → DeepSeek 生成 → answer + citations + trace

二、技术栈（锁定）
FastAPI + uvicorn(8000) / sentence-transformers / ChromaDB PersistentClient / rank-bm25 + jieba / openai SDK 调 DeepSeek / pydantic v2 / pytest。
不引入 LangChain、LlamaIndex、Streamlit、MCP、任何数据库（Chroma 除外）、任何前端框架。

三、固定目录
app/{config,schemas,chunking,store,embedder,bm25_index,reranker,generator,retriever,main}.py
scripts/ingest.py  data/  eval/  tests/  web/index.html  docs/CONTRACT.md

四、接口契约（字段名不得改）
GET /health →
{"status":"ok","embedding":{"model":"...","mode":"bge|disabled"},"rerank":{"model":"...","mode":"cross_encoder|disabled"},"generation":{"model":"deepseek-chat","mode":"online|extractive"},"index":{"docs":5,"chunks":42}}

POST /ingest → {"rebuilt":true,"docs":5,"chunks":42,"elapsed_ms":1234}

POST /ask 请求 {"question":"...","top_k":5}
响应 {"answer":"...","citations":[{"id":1,"doc":"02_校园生活指南.txt","chunk_id":"02_校园生活指南#003","score":0.91,"snippet":"...","score_kind":"relevance"}],
"trace":{"total_ms":1012,"embed_ms":9,"vector_ms":12,"bm25_ms":3,"rerank_ms":84,"gen_ms":880,"fusion":"rrf","rerank_mode":"cross_encoder","gen_mode":"online","candidates":40,"kept":5}}

错误码：空问题或超 500 字 → 400；索引不存在 → 503；异常 → 500，不回传堆栈。

五、三级降级（必须实现，且必须在 /health 与 trace 里可见，禁止静默降级）
Embedding：bge-small-zh-v1.5 → 环境变量指定模型 → 关闭向量路（mode=disabled，仅 BM25）
Rerank：bge-reranker-v2-m3 → bge-reranker-base → 跳过精排（mode=disabled，用 RRF 分数截断）
Generation：DeepSeek → 重试 1 次(30s) → 抽取式回答（取 top3 chunk 前两句拼接，mode=extractive）
模型下载失败不得导致服务启动失败。

六、硬性禁止
不做多轮/记忆；不做 query 改写/意图识别/锚点路由/问题拆分/Agent 循环；不做流式；不做答案去重后处理；不做登录鉴权；不改 data/ 与 eval/questions.json；不硬编码 API Key；不伪造测试结果，没跑通就写"未验证"并贴报错。

七、完成定义
python -m scripts.ingest 退出码 0；uvicorn 可启动；/health 200；/ask 返回 answer+citations；pytest 全绿。

八、每轮回复格式（强制）
## 改动（文件 + 一句话）
## 自测（实际执行的命令 + 真实输出摘要，或"未验证 + 报错原文"）
## 契约核对（4 个勾选项）
## 风险与遗留（最多 3 条）

## Changelog v1.2

背景：云端容器下载 torch 仅约 145KB/s，196MB wheel 需 20 分钟以上，且随后还要下载模型，本地语义后端在演示环境不可用。为「不依赖 torch 也能完整跑通 P2，同时保留接真实语义模型的能力」，embedding 与 rerank 改为可切换后端。

### embedding.mode 取值含义（扩展为四值）
- `api`：启用远程 OpenAI 兼容 `/embeddings` 接口。当 `EMBEDDING_API_KEY` 非空且 `EMBEDDING_BACKEND` 为 `auto`/`api` 时启用。batch=32、超时 30s、失败重试 1 次；返回按 `index` 排序；空文本先过滤。
- `local`：`sentence-transformers` 可 import 时启用，沿用 `SentenceTransformer`，`device="cpu"`。
- `hashing`：前两者不可用时的零依赖降级路径。字符 2-gram/3-gram → `hashlib.blake2b(digest_size=8)` → 对 `EMBEDDING_HASH_DIM` 取模计数 → `log1p` → L2 归一化（零向量返回全零），带进程内 LRU 缓存，确定性可复现。
- `disabled`：无可用后端。`mode` 只返回实际生效的值，禁止写死；`api` 运行时失败会自动落到 `hashing`，并在日志与 `/health` 中体现。

### rerank.mode 取值含义（扩展为三值）
- `api`：`RERANK_API_KEY` 非空（为空时复用 `EMBEDDING_API_KEY`）且 `RERANK_BACKEND` 为 `auto`/`api` 时启用，调用 `POST {RERANK_API_BASE}/rerank`，取 `results[].index` 与 `relevance_score` 降序映射回候选。
- `local`：`CrossEncoder` + sigmoid。
- `disabled`：原样返回前 top_k 条，`rerank_score` 置 `null`。

### 新增环境变量
- `EMBEDDING_BACKEND=auto|api|local|hashing`，`EMBEDDING_API_BASE`，`EMBEDDING_API_KEY`，`EMBEDDING_API_MODEL`，`EMBEDDING_HASH_DIM=512`
- `RERANK_BACKEND=auto|api|local|disabled`，`RERANK_API_BASE`，`RERANK_API_KEY`，`RERANK_API_MODEL`

### 运维约定
- 更换 embedding 后端后必须重跑 `python -m scripts.ingest`：向量维度会随之变化，旧索引与新后端的向量不可混用。
- `hashing` 是降级路径，只保证链路完整可演示，不保证语义检索质量；接入 `api`/`local` 后应重跑 ingest 再评测。

## Changelog v1.3

### rerank.mode 取值含义（扩展为四值）
- `local`：`CrossEncoder(model, max_length=512, device="cpu")`，`predict` 后 sigmoid 归一化，`rerank_score_kind="probability"`。
- `api`：`POST {RERANK_API_BASE}/rerank`，按 `results[].index` 映射回候选，`relevance_score` 排序，`rerank_score_kind="relevance"`。
- `llm`：listwise 重排。只对候选前 20 条、每条截断 180 字，要求模型只输出 `{"ranking":[3,7,1]}`（编号从 1 开始、按相关性降序、最多 top_k）；`temperature=0`、`max_tokens=120`、`timeout=20s`；非法项丢弃、全非法退回原顺序。**score 为名次分 `1-(rank-1)/max(1,len)`，不是概率**，`rerank_score_kind="rank"`。
- `disabled`：原样返回前 top_k 条，`rerank_score=null`，`rerank_score_kind="rank"`，`mode="disabled"`。

### auto 选择顺序与可观测性
- `RERANK_BACKEND=auto` 时固定按 `local → api → llm → disabled` 依次尝试；显式指定某档时从其开始往后再降级。
- 每一级不可用都 `logger.warning` 记录原因（如 `sentence-transformers not importable`、`no RERANK_API_KEY / EMBEDDING_API_KEY`、`no DEEPSEEK_API_KEY`），运行期异常 `logger.exception` 后自动降级到下一档，禁止静默。
- `/health` 的 `rerank.mode` 与 `/ask` trace 的 `rerank_mode` 取同源（`get_reranker().mode`），必须一致；`RERANK_LLM_ENABLED` 为 `RERANK_LLM_MODEL`（默认 `deepseek-chat`）的开关。

### 关键词路分词（修复 v1.2 遗留）
- `app/bm25_index.py::tokenize` 由 `jieba.lcut(text, cut_all=False)` 改为 `jieba.lcut_for_search(text, HMM=True)`：搜索模式同时产出长词与子词，解决「校园网」对「校园网络」切分不一致导致的 token 交集为空。文档侧与 query 侧共用同一 `tokenize`，不引入词表、不做同义词扩展。

## Changelog v1.4

补记 P6 实际落地内容（此前编号从 v1.3 直接跳到 v1.5，缺 v1.4）。

### build_prompt 四条规则定稿
1. 只依据【资料】回答，不得编造资料中没有的数字、时间或事实；
2. 每个事实性陈述后用 `[编号]` 标注来源，编号对应资料序号；
3. 【资料】中没有答案时只回答「未找到相关信息」，不做猜测、不推荐资料外内容；
4. 回答不超过 200 字。规则之外不加任何后处理步骤。

### citations 与 [n] 的对应
- `/ask` 的 `citations[].id` 从 1 开始，与 prompt 中资料的 `[n]` 顺序严格一致，保证 answer 里的 `[n]` 能在 citations 中一一找到。

### /ask 的 trace 字段与错误码
- trace 字段：`total_ms, embed_ms, vector_ms, bm25_ms, rerank_ms, gen_ms, fusion, rerank_mode, gen_mode, candidates, kept`，其中耗时为整数，`fusion` 固定 `"rrf"`。
- 错误码：空问题或超 500 字 → 400；索引不存在 → 503；异常 → 500。

### 500 不回传堆栈
- 未处理异常统一返回 `{"detail":"internal server error"}`，堆栈只进服务端日志，不进入响应体。

## Changelog v1.5

### 生成链路落地（本轮新增）
- 新增 `app/generator.py` 与 `POST /ask`（此前仓库缺少生成模块与入口）。online 走 `httpx` 直连 DeepSeek `/chat/completions`（`temperature=0`、`max_tokens=300`、`timeout=30s`、失败重试 1 次），extractive 取 top3 chunk 前两句拼接；`build_prompt` 四条规则：只用资料、逐句标注来源 `[n]`、无答案即答「未找到相关信息」、不超过 200 字。不需要额外后处理。

### DEEPSEEK_BASE_URL
- 必须为 `https://api.deepseek.com`；**禁止** `https://platform.deepseek.com/v1`（那是网页控制台，非 API 端点）。API 路径为 `/chat/completions`。

### generation.mode 由 key 是否存在决定
- `DEEPSEEK_API_KEY` 非空 → `online`；为空 → `extractive`。两种都必须可演示：online 失败自动降级为 `extractive` 并在 `/health` 与 trace 的 `gen_mode` 中体现，不得删除 extractive 分支。

### rerank 的 llm 后端
- 复用 `DEEPSEEK_API_KEY`；`auto` 顺序为 `local → api → llm → disabled`。

### key 存放与脱敏
- `DEEPSEEK_API_KEY` 只允许存在于环境变量或根目录 `.env`（`.gitignore` 已忽略 `.env`）；`.env.example` 保持空值。key 不得写入任何 `.py`/README/测试/日志/响应体；需要打印时只输出前 6 位 + `***` + 后 4 位。

## Changelog v1.6

### 前端零外链
- `web/index.html` 为单文件演示页：样式内联 `<style>`、脚本内联 `<script>`，不引用任何 CDN、外部字体或图片、前端框架；请求 `POST /ask`、`GET /health` 使用相对路径，超时 60 秒，错误以红色条显示 HTTP 状态与 detail，不静默失败。

### eval 打分规则与通过线
- `eval/run_eval.py` 读取 `eval/questions.json`（只读，运行时不改）逐题 `POST /ask`，单题失败不中断，题间 `--sleep` 默认 0.5s。
- 打分：每个 `must_contain` 内层数组命中任意关键词即算该组命中，全部组命中记 0.7，否则 0；任一 citation 的 `doc` 命中 `citations_should_include` 任意字符串记 0.3；满分 1.0/题。`out_of_scope` 题不参与打分，仅打印供人工确认。
- 通过线：总分 ≥ 80%（12 题时为 9.6）。退出码 PASS=0、FAIL=1。评分只基于 HTTP 响应，脚本不读 `data/` 或索引。

### eval/last_run.json 作为迭代基线
- 每次运行把 `timestamp`、`base_url`、`/health` 快照、逐题明细、总分写入 `eval/last_run.json`，用于后续改动的前后对比。

## Changelog v1.7

### Citation 新增 score_kind
- `/ask` 的每条 citation 新增字段 `score_kind`，取值 `"rank" | "probability" | "relevance" | "rrf"`；来源为 reranker 的 `rerank_score_kind`，为 `null` 时取 `"rrf"`。
- 该字段是本轮 `/ask` 响应唯一新增项；其余字段、错误码与计算逻辑保持不变。
- 前端按 `score_kind` 决定分数呈现，避免把名次分显示成像概率的数字：
  - `rank` → 文案「精排名次 N」，进度条长度 = `score`；
  - `probability` / `relevance` → 文案「相关性 0.91」（两位小数），进度条长度 = `score`；
  - 其他（含 `rrf`）→ 文案「RRF 分数」，保留 4 位小数。

### 前端重做（P8，仅 `web/index.html`）
- 浅色学术风（暖白底 `#F6F4EF`、藏青 `#12344D` 主色、铜金 `#B07A3C` 点缀），高对比、投影可读；系统字体栈，零外链、零框架。
- 顶部栏 72px：标题 + 副标题 + 三个真实状态徽章（Embedding / 精排 / 生成，来自 `/health`）+ 索引行。
- 主体两栏：左栏提问/示例/回答/来源；右栏 sticky「检索链路」面板，四阶段耗时条形图 + `candidates → kept` + 融合方式，数据全部来自本次请求 trace。
- 空答案（含「未找到/没有找到」）时来源区默认折叠为「检索候选 N 条（未作为回答依据）」，避免「没答案却列一堆来源」的自相矛盾。
