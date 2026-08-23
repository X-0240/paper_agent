# 论文调研Agent系统

论文知识库问答 + 手写 ReAct Agent 调研系统。用户输入问题，系统按关键词路由：普通问答走混合检索短路路径；对比/综述类问题由单个 ReAct Agent 通过工具集完成检索、读章节、建卡、关系分析、冲突验证与综述生成。

## 快速启动

```bash
conda activate GPU
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

首次启动会加载 Embedding 模型和 FAISS 索引，耗时约 10-30 秒。

## WebUI

服务启动后浏览器打开 `http://127.0.0.1:8001`，用 `.env` 里的 `API_USERNAME` / `API_PASSWORD` 登录（默认 `admin` / `admin123`）。

界面支持单篇问答和文献综述两种路由：单篇问答流式输出并展示引用来源；文献综述提示处理时长后输出完整综述。前端通过 fetch 读取 SSE，不依赖浏览器原生 EventSource，因此 Token 始终放在请求头里。

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

### GET /ask/stream

SSE 流式接口，`question` 作为查询参数，需要 `Authorization` 头。

事件格式：
```json
{"type":"route","route":"simple"}
{"type":"sources","sources":[{"source":"论文名","section":"章节","text":"片段"}]}
{"type":"token","content":"..."}
{"type":"done"}
```

综述链路会先推送 `{"type":"status","content":"正在生成综述（约 1-3 分钟）"}`。发生异常时推送 `{"type":"error","message":"..."}`，然后推送 `done`，连接不会无提示中断。

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
- 会话缓存：章节缓存 `section_cache`、卡片缓存 `card_cache`，避免重复读/重复调 LLM
- 输入裁剪：卡片字段裁剪、逐篇事实抽取、facts/conflicts/pending 数量上限
- 硬限制：`MAX_AGENT_STEP=15`、`MAX_SEARCH_PER_SESSION=3`，防止死循环和 token 失控

## 评测指标

| 维度 | 指标 | 当前结果 |
|---|---|---|
| 检索 | QASPER 证据级 HitRate@5 | 33.1% 基线 / 39.2%（Rerank+邻接），2026-08-23 重测 |
| 检索 | QASPER 章节级 / 论文级 HitRate@5 | 39.9% / 56.3%（2026-08-23） |
| 检索 | 当前索引（2473 切片）章节级 HitRate@5 | 33.1%（+Rerank），2026-08-23 重测 |
| 多格式 | multi_format_samples 解析通过率 | pytest 覆盖 6 类样例 |
| 冲突粗筛 | 合成小集 P/R/F1 | 1.00 / 1.00 / 1.00 |
| 综述质量 | LLM 评测（完整性/准确性/清晰度） | 7 / 9 / 8，avg 8.0（合成样例） |

## Docker

```bash
docker build -t paper-agent .
```

运行容器时需要挂载 FAISS 索引和 Embedding 模型目录，并通过环境变量覆盖 `FAISS_PATH`、`MODEL_PATH`、`DEEPSEEK_API_KEY` 等配置，见 `.env.example` 模板（本地敏感配置在 `.env`，已被 git 忽略）。

## 目录职责

- `task_router.py`：纯规则路由，无外部依赖
- `CONTRACT.md`：模块2/3 接口契约（State 结构、工具接口、旧代码映射）
- `config.py`：统一配置常量（检索/Agent/成本上限）
- `state.py`：AgentState 与实体数据类（PaperMeta/PaperCard/FactItem/Conflict/ReviewReport）
- `doc_ingest.py`：统一文档解析入口（PDF/Word/Excel/CSV/图片），章节加载/标题映射的唯一实现
- `tools.py`：执行层工具入口，6 个工具已全部落地（search_papers / read_section / build_paper_card / analyze_paper_relations / verify_claim / write_review）；引用由 facts 确定性生成，LLM 不编引用
- `survey_agent.py`：单 ReAct Agent 综述编排，复用 agent_react 循环，工具前置/后置校验、pending_conflicts 聚合、预算降级
- `scripts/run_acceptance.py`：6 条固定 query 的验收脚本（简单/综述/冲突/超预算/无结果 + Transformer对比）
- `rag_tool.py`：混合检索（BM25 + FAISS + 可选 Rerank）
- `agent_react.py`：手写 ReAct 循环，Supervisor + Worker
- `agent1_retrieve.py` / `agent2_parse.py` / `agent3_review.py`：旧流水线模块，正在向 `tools.py` 单 Agent 工具集收敛
- `pipeline.py`：simple 短路与 survey 全链路编排
- `api_server.py`：FastAPI 接口层，JWT 鉴权 + SSE 流式
- `cost_report.py`：每日 LLM 成本报表
- `experiments/`：检索、Rerank、单/多 Agent 对比等评测脚本，不属于线上链路
- `multi_format_samples/`：多格式解析样例（PDF/Word/Excel/CSV/图片/表格），`scripts/make_samples.py` 可重新生成

## 已知边界

- 综述链路一次请求约消耗 4-6 倍 Token，且耗时较长，SSE 目前先完整跑完再分片推送，未做真正的中途流式
- `GET /ask/stream` 依赖 Authorization 头，浏览器原生 EventSource 无法直接携带，前端需要走 fetch 流式或查询参数方案
- 服务依赖本地模型和索引文件，Docker 内需要显式挂载并覆盖路径
- 论文场景以原生文本 PDF 为主，不处理扫描件；图片 OCR 为可选 P1，当前未安装引擎时接口明确返回 OCR 不可用
- Token 计数使用本地 bge-m3 子词 tokenizer（无网络依赖），只作为 DeepSeek 真实 token 数的近似；Chunk 带 `parent_id` 指向完整章节，供后续父文档检索
- Word 来源定位用“文件 + 章节 + 段落区间 + chunk_id”，不依赖页码；Chunk 的 `position` 字段记录段落区间
