# 论文调研 Agent 系统

[![CI](https://github.com/X-0240/paper_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/X-0240/paper_agent/actions/workflows/ci.yml)

把「单篇论文问答」和「多篇论文综述」做在同一条链路上：问题进来先按关键词路由——普通问答走
**混合检索 → 条件重排 → 带引用的回答**这条短路；对比/综述类问题交给一个**手写 ReAct Agent**，
用 6 个工具依次完成检索、读章节、建卡、事实抽取、冲突裁决和综述生成。

**主指标是证据级检索命中率**：生产口径 HitRate@5 = **85.1%（172/202）**，
语料 **300 篇论文 / 16121 个切片**。所有数字都是本机实测值，口径与消融见 [`src/DESIGN.md`](src/DESIGN.md)。

## 这个项目解决什么问题

- **回答要能溯源到章节，而不是"看起来像"**。综述里的引用不是模型写的，是代码从事实记录（facts）里
  确定性生成的；章节名映射不到真实 `chunk_id` 的事实会被直接丢弃（`src/paper_agent/agent/tools.py` 的 `_resolve_chunk_id`）。
- **综述要把分歧摆出来，不是把摘要拼起来**。输出结构固定为共识 / 分歧 / 被推翻结论 / 开放问题，
  有争议的结论必须回原文取证裁决（`verify_claim`），而不是由模型投票决定。
- **中文问英文知识库要能命中**。检索前运行时生成英文 query，同批 202 题配对实测命中率
  72.3% → 85.1%（McNemar p<0.0001）；关掉重排是 81.2%，三个开关的独立贡献都做过单变量消融。
- **单 Agent 还是多 Agent 用数据定**。3 题综述上 3-Agent 判定全胜，代价是 4-6 倍 token
  （对照表在 `src/experiments/README.md`）；生产链路因此用单 Agent + 6 工具，旧 3-Agent 链路保留在仓库里，
  只为让这组对照能复现。

## 这个仓库能跑什么（克隆前必读）

**本仓库只含代码与配置模板。以下内容不在仓库里，所以克隆后无法直接跑通完整服务：**

| 不在仓库里 | 原因 | 怎么补 |
|---|---|---|
| 300 篇论文 PDF 与切片 | 论文正文有版权，且体积大 | 按 `evaluation/v300/manifest.json` 的 arXiv id 重新下载 |
| FAISS 索引 + BM25 语料 | 体积大（`*.faiss` 被忽略） | `python -m paper_agent.retrieval.build_merged_faiss` 重建；规模记录见 `evaluation/v300/index_counts.json`（300 篇 / 16121 切片） |
| Embedding 与 Cross-Encoder 权重 | 本地约 16GB | 自行下载 bge-m3 与 bge-reranker-v2-m3，路径写进 `.env` |
| `.env` 密钥 | 敏感信息不入库 | 复制 `.env.example` 再填自己的 DeepSeek key |

**克隆后能直接跑的**：CI 那批纯逻辑测试（6 个文件、30 条，不需要模型与索引）——

```bash
cd src && python -m pytest tests/test_task_router.py tests/test_agent_react.py tests/test_llm_cost.py \
  tests/test_web_search.py tests/test_evaluation_metrics.py tests/test_dialog_anchor.py -q
```

本地全量收集为 **127 条**，其中 4 条端到端用例需要完整服务已启动，见下面「测试」一节。

## 快速开始

```bash
cd src                      # 代码在 src/paper_agent/ 包里，命令都在 src/ 下执行
conda activate GPU          # 换成你自己的环境名
pip install -r requirements.txt
cp .env.example .env        # 填 DEEPSEEK_API_KEY，以及索引/模型/语料的本地路径

python -m pytest tests -q                                   # 全量回归（需要模型与索引）
uvicorn paper_agent.api_server:app --host 127.0.0.1 --port 8000   # 启动服务，浏览器打开 http://127.0.0.1:8000
python scripts/run_acceptance.py                            # 6 条固定 query 的验收脚本
```

前提是本机已经有索引、语料与两个模型权重（见上一节）。服务启动时会预热 Embedding、FAISS、BM25 与
重排模型，约 45 秒；之后每次请求都不再等模型。

## 架构

### 1）主链（一个问题怎么被处理）

```
HTTP / WebUI
  │  ① JWT 鉴权 → 进程内令牌桶限流（默认 10 次/分/账号）→ 过载准入（超过 MAX_INFLIGHT 直接 503）
  ▼
orchestration.task_router.classify_task(question)      # 纯规则路由，不调模型
  ├─ simple  ─ orchestration.pipeline.simple_answer_with_sources_async
  │             混合检索（本地 retrieval.retrieval_service 与外网 retrieval.web_search 并行）
  │             → 拼上下文 → 流式回答（回答带来源，来源在检索完成时就先推送）
  └─ survey  ─ orchestration.pipeline.survey_pipeline
                agent.survey_agent.run_survey → agent.agent_react.react（手写 ReAct 循环）
                  search_papers → read_section → build_paper_card
                    → analyze_paper_relations（抽事实 + 生成待核实冲突）
                    → verify_claim（回原文裁决冲突）→ write_review（引用由 facts 生成）
                → review_to_markdown(state.review)

落库（两条路径共用）：infra.store → SQLite 六张表：threads / messages / tasks / costs / tool_calls / schema_version
```

> 上图中的模块名前缀相对 `src/paper_agent/`；接口层与配置是包根下的 `api_server.py`、`config.py`、`state.py`。

- 路由是关键词规则（`对比/比较/区别/综述/演进/…` 走 survey），不是模型分类，所以路由本身零成本、可单元测试。
- survey 路径即使模型提前收尾，编排层也会兜底补抽事实并生成综述，避免返回空结果；
  预算不足时降级生成，仍给出带引用的结论。

### 2）检索链路

```
问题（中文）→ 需要时运行时生成英文检索 query
   → BM25 + 向量双路粗召回（生产 RETRIEVAL_CANDIDATES=50）
   → 条件重排：粗排 Top-1 与 Top-K 分差不够大时才跑 Cross-Encoder（bge-reranker-v2-m3）
   → 去重、稳定 chunk_id、Top-5 交付
```

- 语义召回缓存：同义问法直接复用上次的 `chunk_id` 列表（**只缓存召回，不缓存答案**），阈值 0.85；
  索引快照变化即整体失效。缓存命中与近似未命中的追溯在 `GET /metrics/semantic_hits`。
- 重排缓存按 `query + 索引快照` 做进程内 LRU，索引更新后自动失效。
- 检索只有一个入口：所有调用方都走 `retrieval.retrieval_service.get_service()`，不允许各自拼召回逻辑。

### 3）综述链路（单 ReAct Agent + 6 工具）

| 工具 | 作用 | 关键约束 |
|---|---|---|
| `search_papers` | 检索候选论文 | 每会话最多 3 次，防止反复换词刷检索 |
| `read_section` | 读某篇论文的指定章节原文 | 只能读已检索到的论文 |
| `build_paper_card` | 生成论文卡片 | 带提示词版本号的缓存，提示词改了自动失效 |
| `analyze_paper_relations` | 多篇论文抽事实 + 生成待核实冲突 | 事实必须能映射到真实 `chunk_id`，映射不到就丢弃 |
| `verify_claim` | 拿两条事实回原文裁决是否真冲突 | 只能裁决已存在的事实，`fact_id` 不存在直接拒 |
| `write_review` | 生成综述正文 | 没有事实时拒绝执行并指明下一步该调什么 |

ReAct 循环每一步的工具名与参数都要过 schema 校验（`agent.agent_react._validate_args`），非法动作直接拒绝，不丢给模型自己纠。

### 4）模块职责

> 代码模块的路径都相对 `src/paper_agent/`，非代码目录（`evaluation/` 等）相对 `src/`。

| 分组 | 模块 | 职责 / 关键约束 |
|---|---|---|
| 入口与编排 | `api_server.py` | FastAPI 接口层：JWT、SSE 流式、限流、过载准入、WebUI 静态挂载 |
| | `config.py` / `state.py` | 配置常量；`AgentState` 与实体数据类（PaperMeta / PaperCard / FactItem / Conflict / ReviewReport） |
| | `orchestration/pipeline.py` | simple 短路与 survey 全链路的编排入口 |
| | `orchestration/task_router.py` | 关键词路由 + 进程内令牌桶限流，无外部依赖 |
| | `orchestration/run_context.py` | 单次调用的上下文（请求 / 任务 / 工具调用追溯） |
| 检索 | `retrieval/retrieval_service.py` | **唯一检索入口**：查询计划、召回、重排、去重、`chunk_id`、缓存、预算边界 |
| | `retrieval/rerank.py` / `retrieval/rerank_policy.py` | Cross-Encoder 重排，以及"是否需要重排"的判定与进程内 LRU |
| | `retrieval/web_search.py` | 外网检索（Wikipedia / arXiv）并发、超时与降级；simple 路径的混合检索入口 |
| | `retrieval/paper_entities.py` | 论文别名与标题识别（多轮追问补实体用） |
| | `retrieval/build_merged_faiss.py` / `retrieval/pdf_preprocess.py` | 离线构建索引；PDF 预处理 |
| Agent | `agent/agent_react.py` | 手写 ReAct 循环（Supervisor + Worker）、参数 schema 校验、重复动作上限 |
| | `agent/survey_agent.py` | 单 ReAct Agent 综述编排：工具前置/后置校验、`pending_conflicts` 聚合、预算降级、进度文案 |
| | `agent/tools.py` | 6 个工具的落地实现，引用在这里确定性生成 |
| | `agent/agent2_parse.py` | **在用**：建卡与论文对比，`tools.py` 从这里导入 |
| 基础设施 | `infra/llm_api.py` | DeepSeek 调用封装：重试、每日预算硬闸、成本与 token 记账 |
| | `infra/store.py` | SQLite 持久化：会话、消息、任务、成本、工具调用；导入即迁移 |
| | `infra/cost_report.py` | 按日汇总成本的报表脚本 |
| 文档接入 | `ingest/doc_ingest.py` | 统一文档解析入口（PDF / Word / Excel / CSV / 图片），章节加载与标题映射的唯一实现 |
| 持久化 | `migrations/` | SQLite schema 迁移（3 个），路径在 `src/` 下 |
| 设计与契约 | `DESIGN.md` | 检索、缓存、过载、接口、成本、上下文与评测口径的设计说明 |
| | `CONTRACT.md` | 模块 2/3 接口契约（State 结构、工具接口、旧代码映射、冻结门槛） |
| 评测与实验 | `evaluation/` | 语料冻结、题库、指标、消融与复现入口（结论的证据都在这里） |
| | `experiments/` | 一次性对照实验，不在线上链路，清单见 `experiments/README.md` |
| | `scripts/` | 验收、探针、批量对照脚本（如 `run_acceptance.py`、`dialog_rewrite_eval.py`） |
| | `tests/` | 本地全量 127 条；CI 只跑其中 6 个纯逻辑文件 |
| 已弃用 | `legacy/rag_tool.py`、`legacy/agent1_retrieve.py`、`legacy/agent3_review.py` | **已退役，不在生产链路**。保留只为让 `experiments/` 里的 RRF、重排与单/多 Agent 对照能复现旧口径 |

### 5）四条不能越过的线

1. **引用不交给模型**：综述的 references 由 facts 确定性生成，保证每条都能落到真实 `chunk_id`。
2. **检索只有一个入口**：召回与重排只在 `retrieval/retrieval_service.py` 里发生，缓存与预算边界也跟着只做一份。
3. **模型动作必须过 schema**：工具名、参数类型、重复动作次数都在执行前校验，非法动作直接拒绝。
4. **预算与步数是硬限制**：`DAILY_COST_BUDGET` 是每日硬闸，`MAX_AGENT_STEP=15`、`MAX_SEARCH_PER_SESSION=3` 防死循环。

## 目录结构

```
paper_agent/                      # 仓库根：只放仓库级元数据
├── .github/workflows/ci.yml      # 轻量 CI：只跑 6 个纯逻辑测试文件（30 条）
├── LICENSE  README.md
└── src/                          # 代码根，所有命令都在 src/ 下执行
    ├── paper_agent/              # 代码包（uvicorn paper_agent.api_server:app）
    │   ├── api_server.py         # 服务入口：FastAPI app + SSE + WebUI 挂载
    │   ├── config.py  state.py   # 配置常量；共享数据结构
    │   ├── orchestration/        # pipeline.py  task_router.py  run_context.py
    │   ├── retrieval/            # retrieval_service.py  rerank.py  rerank_policy.py
    │   │                         # web_search.py  paper_entities.py  build_merged_faiss.py  pdf_preprocess.py
    │   ├── agent/                # agent_react.py  survey_agent.py  tools.py  agent2_parse.py
    │   ├── infra/                # llm_api.py  store.py  cost_report.py
    │   ├── ingest/               # doc_ingest.py
    │   └── legacy/               # 已退役的 rag_tool / agent1_retrieve / agent3_review
    ├── web/                      # 前端静态资源（index.html / app.js / style.css / KaTeX）
    ├── tests/                    # 本地全量 127 条测试
    ├── scripts/                  # 验收、探针与批量对照脚本
    ├── evaluation/               # 语料冻结、题库、指标与消融（证据都在这里）
    ├── experiments/              # 一次性对照实验，见其中的 README.md
    ├── migrations/               # SQLite schema 迁移
    ├── multi_format_samples/     # 多格式解析样例（PDF / Word / Excel / CSV / 图片）
    ├── DESIGN.md  CONTRACT.md    # 设计与接口契约
    └── requirements.txt  .env.example  qasper_titles.json

不进仓库（由 .gitignore 挡住）：models/（约 16GB 权重）、docs/（论文 PDF 与个人资料）、
src/datasets/、src/data/、本地缓存（query_cache.json、llm_usage.jsonl）与 .env
```

## 接口

**HTTP**（FastAPI，13 个端点）：

| 端点 | 作用 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /` | WebUI 页面 |
| `POST /login` | 换取 JWT |
| `POST /ask` | 非流式问答，`use_survey` 可强制走某条链路 |
| `GET /ask/stream` | SSE 流式问答（`question` 走查询参数，需要 `Authorization` 头） |
| `GET /threads`、`PATCH /threads/{id}`、`DELETE /threads/{id}` | 会话列表、重命名、删除 |
| `GET /threads/{id}/messages`、`GET /threads/{id}/tasks` | 会话历史与任务记录 |
| `GET /tasks/{task_id}` | 单次任务状态 |
| `GET /metrics`、`GET /metrics/semantic_hits` | 运行指标与语义缓存命中追溯 |

SSE 事件：`thread`（会话 ID）→ `route`（走了哪条链路）→ `sources`（检索一完成就推送，用户先看到来源）
→ `token`（正文分片）→ `progress` / `ping`（综述链路的长任务进度与心跳）→ `done`；
出错时推 `error` 再推 `done`，连接不会无提示中断。

**CLI**：`python -m paper_agent.orchestration.pipeline` 是交互式入口，按路由分别走短路与综述链路。

## WebUI

浏览器打开 `http://127.0.0.1:8000`，用 `.env` 里的 `API_USERNAME` / `API_PASSWORD` 登录（默认 `admin` / `admin123`）。
界面支持单篇问答与文献综述两种路由：问答流式输出并展示引用来源，综述先提示预计时长再输出完整正文。
前端通过 fetch 读 SSE（不用浏览器原生 EventSource），Token 始终放在请求头里。

会话侧栏支持新建 / 重命名 / 删除会话、按最后活动时间分组、刷新后还原上次会话、历史消息与引用卡片。
会话管理与多轮追问的实现细节见 [`src/DESIGN.md`](src/DESIGN.md)。

## 测试

```bash
cd src && python -m pytest tests -q
```

纯逻辑测试（不加载模型，CI 跑这些）：`test_task_router.py`、`test_agent_react.py`、`test_llm_cost.py`、
`test_web_search.py`、`test_evaluation_metrics.py`、`test_dialog_anchor.py`。

其中 `test_dialog_anchor.py` 覆盖多轮追问的锚点逻辑；`test_dialog_e2e.py` 是端到端多轮回归，
需要本地完整服务已启动，未启动时自动跳过，因此不接入轻量 CI。其余测试（`test_api.py` 等）会加载模型。

**本地全量收集为 127 条**（2026-09-28 复测）；未启动完整服务时实测为 **123 passed、4 skipped**。
接入 CI 的是上面 6 个文件 **30 条**。

## 已知边界

- **外网检索当前不可用**：arXiv 与维基百科两个源都取不到数据，因此 `.env` 里用 `WEB_SEARCH_ENABLED=0`
  直接跳过；降级逻辑保证主链路不受影响。
- **综述不是真流式**：正文一整段生成后才分片推送，抽取事实到正文之间有约 25 秒只有心跳；一次综述约消耗 4-6 倍 token。
- **首 token 延迟约 8-12 秒**（检索 3.5 秒 + 模型读资料后开口 4.8 秒），这是当前方案的下限。

完整清单（限流挡不住什么、输入长度、公式渲染、多轮改写依赖模型等）见 [`src/DESIGN.md`](src/DESIGN.md) 的「已知限制与边界」。

## 文档入口

- 设计细节（缓存与阈值、过载准入、外网检索、重排、WebUI 会话管理、多轮追问、接口、成本、上下文、评测指标、已知限制）：[`src/DESIGN.md`](src/DESIGN.md)
- 接口契约与冻结门槛：[`src/CONTRACT.md`](src/CONTRACT.md)
- 对照实验索引：[`src/experiments/README.md`](src/experiments/README.md)
- 迭代流水、结论沉淀与方向调研：在本地 `docs/`（不进公开仓库）

## 协作与反馈

个人项目，没有开放的贡献流程；发现问题、有建议或想指出口径错误，请开
[Issue](https://github.com/X-0240/paper_agent/issues)。

## 许可与第三方素材

本仓库代码与文档采用 [MIT 许可](LICENSE)。以下内容**不在仓库内**，因此也不在本许可的授权范围内：

- **第三方论文原文与 PDF**：版权归原作者与出版方。仓库只保存 arXiv id、哈希与评测题（`src/evaluation/v300/`），
  论文正文需自行按 manifest 下载。
- **模型权重**：bge-m3、bge-reranker-v2-m3 等由各自原仓库发布并遵循其自身许可，需自行下载。
- **本地评测试题与人工复核表**：属于个人评测材料，未随仓库分发。

README 与文档里的性能/指标数字都是本机实测值，随硬件、模型与外部 API 状态变化，不承诺复现。
