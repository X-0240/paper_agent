# 论文调研多Agent系统

论文知识库问答 + 手写ReAct多Agent调研系统。用户输入问题，系统按关键词路由：普通问答走混合检索短路路径，对比/综述类问题走 3-Agent 全链路（检索 -> 解析对比 -> 事实校验与综述生成）。

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

## Docker

```bash
docker build -t paper-agent .
```

运行容器时需要挂载 FAISS 索引和 Embedding 模型目录，并通过环境变量覆盖 `FAISS_PATH`、`MODEL_PATH`、`DEEPSEEK_API_KEY` 等配置，见 `.env.example` 模板（本地敏感配置在 `.env`，已被 git 忽略）。

## 目录职责

- `task_router.py`：纯规则路由，无外部依赖
- `rag_tool.py`：混合检索（BM25 + FAISS + 可选 Rerank）
- `agent_react.py`：手写 ReAct 循环，Supervisor + Worker
- `agent1_retrieve.py` / `agent2_parse.py` / `agent3_review.py`：3-Agent 工具层
- `pipeline.py`：simple 短路与 survey 全链路编排
- `api_server.py`：FastAPI 接口层，JWT 鉴权 + SSE 流式
- `cost_report.py`：每日 LLM 成本报表
- `experiments/`：检索、Rerank、单/多 Agent 对比等评测脚本，不属于线上链路

## 已知边界

- 综述链路一次请求约消耗 4-6 倍 Token，且耗时较长，SSE 目前先完整跑完再分片推送，未做真正的中途流式
- `GET /ask/stream` 依赖 Authorization 头，浏览器原生 EventSource 无法直接携带，前端需要走 fetch 流式或查询参数方案
- 服务依赖本地模型和索引文件，Docker 内需要显式挂载并覆盖路径
