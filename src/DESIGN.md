# 设计说明（检索、缓存、过载、接口、成本与评测）

本文件是从仓库根 README 拆出的设计细节。项目介绍与运行方式见 [`../README.md`](../README.md)，
接口契约与冻结门槛见 [`CONTRACT.md`](CONTRACT.md)。

## 语义召回缓存

同义问法复用上一次的召回结果，省掉一次编码与两路召回（实测 3.69s → 0.02s，检索耗时降到约 1/180）。**只缓存 chunk_id，不缓存答案**——实测同义问法 Top5 重合 4-5/5，检索比答案稳定，缓存检索风险低。

- 命中判断：问题编码后与缓存条目比余弦相似度，阈值 `SEMANTIC_CACHE_THRESHOLD`（默认 0.85）
- 阈值依据：实测同义问法最低 0.88（去掉一个术语缩写反例后）、同实体不同问题最高 0.78
- 索引快照变化时整体失效，避免复用旧切片
- `SEMANTIC_CACHE=0` 关闭做对照；缓存统计在 `GET /metrics` 的 `semantic_cache` 字段
- 已知局限：「RAG 是什么？」与「什么是检索增强生成」相似度只有 0.3962，**纯向量抓不住术语缩写与全称的同义**，需要别名映射补（`paper_entities.py` 里已有别名字典可复用）

### 阈值为什么定 0.85 而不是更保守的 0.92

用 16 对同义问法 + 19 对同实体不同问题扫过阈值，数据不支持收紧：

| 阈值 | 同义命中 | 异问误命中 | 11 次真实重复提问的命中率 |
|---|---|---|---|
| **0.85（采用）** | 12/16 | **0/19** | **7/11 = 64%** |
| 0.90 | 7/16 | 0/19 | 1/11 = 9% |
| 0.92 | 2/16 | 0/19 | 1/11 = 9% |

两条依据：**一是 0.85 已经是零误命中**（硬负例最高相似度 0.7844），收到 0.92 不会多挡住任何危险命中；**二是同义问法大量落在 0.85-0.92 之间**，收紧只会砍掉合法命中，命中率从 64% 掉到 9%，缓存等于没做。

安全边界在 0.80-0.82（硬负例上限 0.7844、真实同义下限 0.8170），0.85 取在偏保守一侧。注意这个余量只有 0.03，**「零误命中」只在当前样例集上成立，样本不大**。

### 命中追溯

`GET /metrics/semantic_hits?limit=20` 返回三部分：统计、最近命中（`asked` 命中了 `matched`、相似度、是精确还是语义命中、时间）、以及 `near_misses`（未达阈值但最接近的记录）。

用途是**事后核查误命中**：缓存误命中的表现是「用户问 A 却看到 B 的答案」，不报错，比崩溃更难发现。有了这条记录，出问题时能直接看到是谁的答案被复用；`near_misses` 还能用来判断阈值是不是卡太紧（如果大量接近阈值的记录其实是同义问法，说明该放宽）。

## 过载准入（队列与背压）

超过处理能力时**快速明确拒绝**，而不是让请求挂着等超时。实测拒绝在 0.06-0.23 秒返回，之前是干等十几分钟。

- `MAX_INFLIGHT`（默认 8）并发上限；`ADMISSION_WAIT_BUDGET`（默认 60 秒）用户能接受的最长排队时长；`ADMISSION_AVG_SECONDS`（默认 17 秒）单次问答耗时估算
- 预计等待超过预算就返回 **503** 并带原因，客户端能立刻知道该重试
- **等待预算的含义是「用户能接受的最长排队时长」，不是「预计等待」**。写成后者会让 inflight≥2 就拒绝、8 个名额只用上 2 个（实测踩过）
- 名额在流结束时归还；响应建立之前就失败（如会话不存在）也在异常分支归还
- 统计在 `GET /metrics` 的 `admission` 字段（当前并发、峰值、两类拒绝计数）

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

**生产配置（以本机 `.env` 为准）**：`RETRIEVAL_CANDIDATES=50`、`RERANK_MODE=conditional`、`USE_RERANK=1`、`QUERY_GENERATION_ENABLED=1`、`ENTITY_BOOST=1`、`MULTI_TURN_QUERY_REWRITE=0`。

代码里的兜底默认更保守（候选 25、英文 query 关闭），`.env.example` 模板给的是最小可跑值（候选 25、关闭）——**模板值≠生产口径**，看单变量消融数字时按生产配置读（候选 50 + 条件重排 + 运行时英文 query 生成）。

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
| v300 检索（生产口径：候选 50 + 条件重排 + 运行时英文 query 生成） | 证据级 HitRate@5 | **85.1%（172/202）**；关重排 81.2%、关英文 query 生成 72.3%（146/202）。单变量消融：英文 query +12.9pp（p<0.0001）、条件重排 +4.0pp（p=0.0215）、候选 25→50 无收益 |
| v300 语料 | 论文/切片/split | 300 篇 / 16121 切片 / 100 dev + 200 test |
| 检索 | QASPER 证据级 HitRate@5 | 33.1% 基线 / 39.2%（Rerank+邻接），2026-08-23 重测 |
| 检索 | QASPER 章节级 / 论文级 HitRate@5 | 39.9% / 56.3%（2026-08-23） |
| 中文题（88题，翻译成英文） | 证据级 / 章节级 HitRate@5 | 61.4% / 67.0%（2026-08-25，主指标证据级） |
| 中文题（88题，原生英文改写） | 证据级 HitRate@5 / @10 / @20 | 64.8% / 75.0% / 85.2%（双AI交叉，章节级@5=72.7%） |
| 中文题可靠性 | 99题双AI交叉校验+模型复核 | 31题修正章节、11题判无效/无答案，有效88题 |
| 多格式 | multi_format_samples 解析通过率 | pytest 覆盖 6 类样例 |
| 冲突粗筛 | 合成小集 P/R/F1 | 1.00 / 1.00 / 1.00 |
| 综述质量 | LLM 评测（完整性/准确性/清晰度） | 7 / 9 / 8，avg 8.0（合成样例） |
