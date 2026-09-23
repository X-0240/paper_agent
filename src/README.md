# 论文调研Agent系统

论文知识库问答 + 手写 ReAct Agent 调研系统。用户输入问题，系统按关键词路由：普通问答走混合检索短路路径；对比/综述类问题由单个 ReAct Agent 通过工具集完成检索、读章节、建卡、关系分析、冲突验证与综述生成。

## 快速启动

```bash
conda activate GPU
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

## 启动预热

服务通过 lifespan 钩子在启动阶段就加载 Embedding 模型、FAISS 索引、BM25 语料和重排模型，启动约 45 秒，之后每次请求都不再等模型。

改成启动预热之前的做法是懒加载：模型和索引在第一次请求时才加载，那次请求要多等 20 多秒（实测进程导入依赖 18.4 秒 + 加载模型 5.4 秒）。预热把这份等待从"用户第一次提问"挪到了"服务启动"。预热失败只记 warning，不会阻断启动。

## WebUI

服务启动后浏览器打开 `http://127.0.0.1:8000`，用 `.env` 里的 `API_USERNAME` / `API_PASSWORD` 登录（默认 `admin` / `admin123`）。

界面支持单篇问答和文献综述两种路由：单篇问答流式输出并展示引用来源；文献综述提示处理时长后输出完整综述。前端通过 fetch 读取 SSE，不依赖浏览器原生 EventSource，因此 Token 始终放在请求头里。

左侧会话栏支持完整的对话管理：

- 一个会话里可以连续问多个问题，只有点「新对话」才会新建会话
- 会话标题默认取该会话第一条提问，双击标题可以改成自己的名字（改名不影响它在列表里的排序位置）
- 按最后活动时间分组：今天 / 昨天 / 近 7 天 / 更早
- 悬停出现删除按钮；会话栏右上角可收起，收起后对话区左上角会出现展开入口
- 刷新页面会自动回到上次的会话并还原历史消息与引用卡片（会话 ID 记在 localStorage）
- 每条提问可复制、可修改后重新提问；每条回答可复制、可重新生成，操作按钮为图标，悬停才显示

多轮追问分三层，三层各有各的职责：

- **生成侧**（`DIALOG_HISTORY_ENABLED=1`，默认开）：把最近 3 轮问答拼进提示词，解决「它/那这个」指代的语义理解。拼在检索资料之前，并附一句提示要求模型结合上文判断
- **检索侧实体注入**（默认开）：追问里没有论文名时，从上一轮对话用别名识别提取论文名，注入检索查询。实测这一层是提升的关键——7 个代词型追问的命中从 2/7 提到 5/7
- **多轮改写**（`MULTI_TURN_QUERY_REWRITE=0`，默认关）：检索前用一次 LLM 把追问补全成自足问题。实测开启与关闭的追问命中都是 2/7，零收益还多一次调用，因此默认关闭

三层顺序是：生成侧拿到历史 → 检索侧补实体 → 改写（若开启）。批量对照数据见 `scripts/dialog_rewrite_eval.py`。

输入校验：空白、纯标点等内容在入口被拒（HTTP 400），不会消耗模型调用。前端的发送按钮做同样的判断，避免白跑一次请求。

## 外网检索

- simple 路径并行跑本地 FAISS 检索 + 外网检索（Wikipedia REST + arXiv API），来源统一展示
- 外网请求默认 8 秒超时、并发上限 5，失败自动降级为本地-only，不阻塞主链路
- 配置：`.env` 的 `WEB_SEARCH_TIMEOUT` / `WEB_SEARCH_MAX_RESULTS` / `WEB_SEARCH_CONCURRENCY`
- `WEB_SEARCH_ENABLED=0` 会跳过外网检索直接走本地。外网两个源都不可用时建议关闭：降级逻辑只保证不报错，但仍要白等一个超时周期（实测约 4.5 秒）

## 重排（Cross-Encoder）

- 生产链路：BM25+向量粗召回 Top-25 → 条件触发 Cross-Encoder 精排 → Top-5
- 重排模型：本地 `bge-reranker-v2-m3`（多语言，无 API 成本）；粗排 Top-1 与 Top-K 分差足够大时跳过重排，控制延迟
- 缓存：按 query + 索引快照做进程内 LRU，索引更新后自动失效
- 配置：`USE_RERANK` / `RERANK_MODEL_NAME` / `RERANK_MAX_CHARS` / `RERANK_TRIGGER_MARGIN` / `RERANK_CACHE_SIZE`
- 实测：88 题 64.8%→72.7%（+8.0pp），正向 42 题 64.3%→83.3%（+19.0pp）；候选 25 约 1.0s/题，候选 10 约 0.44s/题

当前统一 `RetrievalService` 已接入，但 Top-50 和检索专用英文 query 尚未达到可部署门槛，默认关闭；生产参数仍为候选 25、条件重排、原问题查询。

## 接口

### POST /login

请求体：
```json
{"username":"admin","password":"admin123"}
```

返回 `access_token`，后续请求放在 `Authorization: Bearer <token>`。

### POST /ask

请求体：
```json
{"question":"什么是Transformer？","use_survey":null}
```

`use_survey` 为空时按关键词自动路由；为 `true` 强制走综述全链路，`false` 强制短路。

每个账号默认每分钟最多 10 次请求（`.env` 的 `RATE_LIMIT_PER_MINUTE` 可调），超限返回 `429`。

### GET /ask/stream

SSE 流式接口，`question` 作为查询参数，需要 `Authorization` 头。

事件格式：
```json
{"type":"thread","thread_id":"th_xxx","task_id":"tk_xxx"}
{"type":"route","route":"simple"}
{"type":"sources","sources":[{"source":"论文名","section":"章节","text":"片段"}]}
{"type":"token","content":"..."}
{"type":"progress","content":"抽取事实 1/2：Attention_Is_All_You_Need"}
{"type":"ping"}
{"type":"done"}
```

可选参数 `thread_id`：带上就追加到该会话，不带就新建一个会话。`thread` 事件会把最终使用的会话 ID 返回给前端。

综述链路会先推送 `{"type":"status","content":"正在生成综述（约 1-3 分钟）"}`，之后推送 `progress` 分步进度，长时间没有新步骤时补一条带等待时长的 `progress`，心跳用 `ping`。发生异常时推送 `{"type":"error","message":"..."}`，然后推送 `done`，连接不会无提示中断。

### GET /threads

返回当前用户的会话列表，按最后活动时间倒序。`title` 是自定义标题，没改过名时回退为该会话的第一条提问；`title_source` 标明是 `custom` 还是 `auto`。

### PATCH /threads/{thread_id}

请求体 `{"title":"新名字"}`。传空字符串表示恢复为自动命名。改名不更新 `updated_at`，所以会话不会因此在列表里跳到最前。

### DELETE /threads/{thread_id}

删除会话及其消息与任务记录。不存在的会话返回 404，不是本人的会话同样按 404 处理，避免泄露存在性。

## 测试

```bash
python -m pytest tests -q
```

`tests/test_task_router.py`、`tests/test_agent_react.py`、`tests/test_llm_cost.py` 是纯逻辑测试，不加载模型，CI 只跑这三个文件；`tests/test_api.py` 会加载模型，用于本地全量回归。

## 成本控制

- 每次 LLM 调用会把 token 用量和估算成本写入 `llm_usage.jsonl`（已被 git 忽略）
- 估算价按 DeepSeek V4-Flash 峰谷价计算：高峰 9-12 点、14-18 点输出 9 元/百万，闲时 4.5 元/百万
- `.env` 里 `DAILY_COST_BUDGET` 是每日预算硬闸，超了会返回 `[预算保护]` 提示，不再调用 API
- 查看每日成本：`python cost_report.py`
- 省钱建议：重活（综述全链路、批量评测）放到闲时跑；Agent 循环已加输出上限和历史裁剪，防止多轮调用上下文无限膨胀

## 上下文管理

- ReAct 历史裁剪：只保留系统指令 + 初始问题 + 最近 3 轮
- Observation 超长截断（1500 字截到 1000 字）
- 会话缓存：章节缓存 `section_cache`、卡片缓存 `card_cache`，避免重复读/重复调 LLM；卡片缓存带提示词版本号，`CARD_PROMPT` 改动自动失效
- 输入裁剪：卡片字段裁剪、逐篇事实抽取、facts/conflicts/pending 数量上限
- 硬限制：`MAX_AGENT_STEP=15`、`MAX_SEARCH_PER_SESSION=3`，防止死循环和 token 失控

## 评测指标

| 维度 | 指标 | 当前结果 |
|---|---|---|
| v300 检索 | 证据级 HitRate@5 / @10 | 候选 81.7% / 88.6%（202/202 test，模型复核），基线 72.8% / 82.2% |
| v300 语料 | 论文/切片/split | 300 篇 / 16121 切片 / 100 dev + 200 test |
| 检索 | QASPER 证据级 HitRate@5 | 33.1% 基线 / 39.2%（Rerank+邻接），2026-08-23 重测 |
| 检索 | QASPER 章节级 / 论文级 HitRate@5 | 39.9% / 56.3%（2026-08-23） |
| 中文题（88题，翻译成英文） | 证据级 / 章节级 HitRate@5 | 61.4% / 67.0%（2026-08-25，主指标证据级） |
| 中文题（88题，原生英文改写） | 证据级 HitRate@5 / @10 / @20 | 64.8% / 75.0% / 85.2%（双AI交叉，章节级@5=72.7%） |
| 中文题可靠性 | 99题双AI交叉校验+模型复核 | 31题修正章节、11题判无效/无答案，有效88题 |
| 多格式 | multi_format_samples 解析通过率 | pytest 覆盖 6 类样例 |
| 冲突粗筛 | 合成小集 P/R/F1 | 1.00 / 1.00 / 1.00 |
| 综述质量 | LLM 评测（完整性/准确性/清晰度） | 7 / 9 / 8，avg 8.0（合成样例） |

## Docker

```bash
docker build -t paper-agent .
```

运行容器时需要挂载 FAISS 索引和 Embedding 模型目录，并通过环境变量覆盖 `FAISS_PATH`、`MODEL_PATH`、`DEEPSEEK_API_KEY` 等配置，见 `.env.example` 模板（本地敏感配置在 `.env`，已被 git 忽略）。

## 目录职责

- `task_router.py`：纯规则路由 + 进程内令牌桶限流，无外部依赖
- `CONTRACT.md`：模块2/3 接口契约（State 结构、工具接口、旧代码映射）
- `config.py`：统一配置常量（检索/Agent/成本上限）
- `state.py`：AgentState 与实体数据类（PaperMeta/PaperCard/FactItem/Conflict/ReviewReport）
- `doc_ingest.py`：统一文档解析入口（PDF/Word/Excel/CSV/图片），章节加载/标题映射的唯一实现
- `tools.py`：执行层工具入口，6 个工具已全部落地（search_papers / read_section / build_paper_card / analyze_paper_relations / verify_claim / write_review）；引用由 facts 确定性生成，LLM 不编引用
- `survey_agent.py`：单 ReAct Agent 综述编排，复用 agent_react 循环，工具前置/后置校验、pending_conflicts 聚合、预算降级
- `retrieval_service.py`：唯一检索服务入口，统一查询计划、召回、重排、去重、稳定 chunk_id、缓存和预算边界
- `evaluation/v150_pipeline.py`：v150语料的manifest、arXiv版本固定、PDF下载、分层split、候选出题审计和快照冻结入口
- `evaluation/v300/manifest.json`、`evaluation/questions/v300_*`：v300 300篇论文和冻结评测题
- `scripts/run_acceptance.py`：6 条固定 query 的验收脚本（简单/综述/冲突/超预算/无结果 + Transformer对比）
- `rag_tool.py`：**旧检索入口，已弃用**。生产链路走 `retrieval_service.py`；保留是因为 `experiments/` 下的 RRF 与重排对照脚本依赖它复现旧口径
- `web_search.py`：外网检索并发（Wikipedia/arXiv），统一来源结构与降级
- `agent_react.py`：手写 ReAct 循环，Supervisor + Worker
- `agent1_retrieve.py` / `agent3_review.py`：**旧 3-Agent 链路，已弃用**。生产综述走 `survey_agent.py` 的单 Agent 工具集；`pipeline.py` 里那段回退分支已于 2026-09-23 删除。保留它们只为让 `experiments/experiment_single_vs_multi.py` 能复现「单 Agent 与 3-Agent 对比」那组对比数据
- `agent2_parse.py`：**在用**。`tools.py` 仍从它导入 `build_paper_card` 与 `compare_papers`
- `pipeline.py`：simple 短路与 survey 全链路编排
- `api_server.py`：FastAPI 接口层，JWT 鉴权 + SSE 流式 + 限流
- `cost_report.py`：每日 LLM 成本报表
- `experiments/`：检索、Rerank、单/多 Agent 对比等评测脚本，不属于线上链路
- `multi_format_samples/`：多格式解析样例（PDF/Word/Excel/CSV/图片/表格），`scripts/make_samples.py` 可重新生成

## 已知边界

- 综述链路一次请求约消耗 4-6 倍 Token，且耗时较长，SSE 目前先完整跑完再分片推送，未做真正的中途流式
- **外网检索当前不可用**：arXiv 的 `export.arxiv.org/api/query` 对所有参数组合返回 406（主站正常），维基百科请求超时。代码里的降级逻辑保证主链路不受影响，但两个源都拿不到数据，因此 `.env` 里用 `WEB_SEARCH_ENABLED=0` 直接跳过，省掉一个超时周期的等待
- **公式渲染依赖模型输出格式**：前端能渲染 `\(...\)`、`\[...\]`、`$...$` 三种定界符，但模型有时把公式写成裸文本（如 `Attention(Q,K,V)=Softmax(QK^T/√d_k)V`），这种渲染不了。已在系统提示词里要求公式必须用 LaTeX，但提示词不能百分百约束住
- 首 token 延迟约 8-12 秒（检索 3.5 秒 + 模型读资料后开口 4.8 秒），这是当前方案的下限；`sources` 事件在检索完成时就会推送，可用来给用户可见反馈
- **限流防不住「慢慢刷」**：进程内令牌桶按每分钟 10 次补充，而一次问答要十几秒，等于两次请求之间自动回满 2 个以上令牌。实测连发 14 次全部放行；它能挡的是秒级并发。对 LLM 成本的真正兜底是每日预算硬闸与 `MAX_AGENT_STEP` 上限，要限制长窗口用量得另加按小时/按天的独立计数
- **单次请求没有输入长度上限**：超长输入（实测 8000 字）不是被应用层拒绝，而是卡在 HTTP 客户端的 URL 长度限制上（`InvalidURL: query too long`）。`/ask/stream` 用 GET 传问题，长文本场景需要改成 POST
- **多轮改写的指代解析依赖模型**：改写本身是一次 LLM 调用，模型没改对的代词它也解析不了。当前只在少数样例上人工验证，没有批量评测
- 综述正文是一整段生成的，只有在管线结束后才拿到，因此正文本身没有真流式；抽取事实与生成正文之间有约 25 秒只有心跳和等待时长提示
- `GET /ask/stream` 依赖 Authorization 头，浏览器原生 EventSource 无法直接携带，前端需要走 fetch 流式或查询参数方案
- 服务依赖本地模型和索引文件，Docker 内需要显式挂载并覆盖路径
- 论文场景以原生文本 PDF 为主，不处理扫描件；图片 OCR 为可选 P1，当前未安装引擎时接口明确返回 OCR 不可用
- Token 计数使用本地 bge-m3 子词 tokenizer（无网络依赖），只作为 DeepSeek 真实 token 数的近似；Chunk 带 `parent_id` 指向完整章节，供后续父文档检索
- Word 来源定位用“文件 + 章节 + 段落区间 + chunk_id”，不依赖页码；Chunk 的 `position` 字段记录段落区间
