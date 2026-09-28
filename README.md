# 论文调研Agent系统

[![CI](https://github.com/X-0240/paper_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/X-0240/paper_agent/actions/workflows/ci.yml)

论文知识库问答 + 手写 ReAct Agent 调研系统。用户输入问题，系统按关键词路由：普通问答走混合检索短路路径；对比/综述类问题由单个 ReAct Agent 通过工具集完成检索、读章节、建卡、关系分析、冲突验证与综述生成。

一句话：**把单 Agent 架在 300 篇论文的知识库上，做「可溯源的单篇问答 + 文献综述」，主指标是证据级检索命中率（生产口径 202 题 85.1%，172/202）。**

## 技术栈

Python / FastAPI / FAISS / BM25（rank_bm25）/ bge-m3 嵌入 / bge-reranker-v2-m3 交叉编码器 / DeepSeek API / SSE 流式 / JWT / SQLite 持久化 / GitHub Actions

## 核心能力

- **混合检索 + 条件重排**：BM25 + 向量粗召回 Top-50 → 按 Top-1 与 Top-K 分差条件触发 Cross-Encoder 精排 → Top-5
- **跨语言检索**：中文问题运行时生成英文检索 query（生产默认开），证据级命中率 72.3%→85.1%（同批 202 题配对，McNemar p<0.0001）
- **单 ReAct Agent + 6 工具**：检索、读章节、建卡、关系分析、冲突验证、综述生成；引用由事实记录确定性生成，不由模型编造
- **多轮追问**：生成侧拼最近 3 轮历史；追问缺实体名时从上一轮提问提取论文名注入检索，代词型追问命中 2/7→7/7
- **可复现评测**：语料与题库按 SHA256 冻结，202 题逐题复现一致（202/202）；单变量消融给出每个开关的独立贡献
- **工程外壳**：JWT 多用户隔离、SSE 真流式、令牌桶限流、过载快速拒绝、语义召回缓存（阈值 0.85）、按任务成本归因

## 这个仓库能跑什么（克隆前必读）

**本仓库只含代码与配置模板。以下内容不在仓库里，所以克隆后无法直接跑通完整服务：**

| 不在仓库里 | 原因 | 怎么补 |
|---|---|---|
| 300 篇论文 PDF 与切片 | 论文正文有版权，且体积大 | 按 `evaluation/v300/manifest.json` 的 arXiv id 重新下载 |
| FAISS 索引 + BM25 语料 | 体积大（`*.faiss` 被忽略） | `build_merged_faiss.py` 重建；规模记录见 `evaluation/v300/index_counts.json`（300 篇 / 16121 切片） |
| Embedding 与 Cross-Encoder 权重 | 本地约 16GB | 自行下载 bge-m3 与 bge-reranker-v2-m3，路径写进 `.env` |
| `.env` 密钥 | 敏感信息不入库 | 复制 `.env.example` 再填自己的 DeepSeek key |

**克隆后能直接跑的**：CI 那批纯逻辑测试（6 个文件、30 条，不需要模型与索引）——

```bash
cd src && python -m pytest tests/test_task_router.py tests/test_agent_react.py tests/test_llm_cost.py \
  tests/test_web_search.py tests/test_evaluation_metrics.py tests/test_dialog_anchor.py -q
```

本地全量收集为 **127 条**，其中 4 条端到端用例需要完整服务已启动，见下面「测试」一节。

## 快速启动

```bash
cd src                      # 代码都在 src/ 下
conda activate GPU          # 换成你自己的环境名
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

前提是本机已经有索引、语料与两个模型权重（见上一节）。

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
- **检索侧实体注入**（默认开）：追问里没有论文名时，从上一轮**用户提问**（不扫助手回答）用别名识别提取论文名，注入检索查询，最多取最近 3 篇。实测这一层是提升的关键——7 个代词型追问的命中从 2/7 提到 7/7
  - 为什么只取提问：助手回答末尾带「来源类型：本地论文 [local:XXX]」这类引用标注，扫回答会把引用过的论文全当成用户提到的对象。实测注入 4 个锚点（其中 3 个是引用标注带来的）反而把正确论文挤出候选池
  - 为什么支持多篇：上一轮是综述时用户可能点到好几篇，只取第一篇会选错；注入多篇让检索自己分辨
- **多轮改写**（`MULTI_TURN_QUERY_REWRITE=0`，默认关）：检索前用一次 LLM 把追问补全成自足问题。实测开启与关闭的追问命中都是 2/7，零收益还多一次调用，因此默认关闭

三层顺序是：生成侧拿到历史 → 检索侧补实体 → 改写（若开启）。批量对照数据见 `scripts/dialog_rewrite_eval.py`。

输入校验：空白、纯标点等内容在入口被拒（HTTP 400），不会消耗模型调用。前端的发送按钮做同样的判断，避免白跑一次请求。

## 测试

```bash
python -m pytest tests -q
```

纯逻辑测试（不加载模型，CI 跑这些）：`test_task_router.py`、`test_agent_react.py`、`test_llm_cost.py`、`test_web_search.py`、`test_evaluation_metrics.py`、`test_dialog_anchor.py`。

其中 `test_dialog_anchor.py` 覆盖多轮追问的锚点逻辑（只取用户提问、支持多篇、数量上限）；`test_dialog_e2e.py` 是端到端多轮回归，需要本地完整服务已启动，未启动时自动跳过，因此不接入轻量 CI。

其余测试（`test_api.py` 等）会加载模型，用于本地全量回归。**本地全量收集为 127 条**（`python -m pytest tests -q --collect-only` = 127 collected，2026-09-28 复测）；本次在未启动完整服务时实测为 **123 passed、4 skipped**。接入 CI 的是上面 6 个文件 **30 条**。

多轮改写的批量对照用 `scripts/dialog_rewrite_eval.py`（7 个代词型追问，跑一遍约 2 分钟）；锚点追踪用 `scripts/trace_anchor.py`。

## Docker

```bash
# Dockerfile 在 src/ 下，构建上下文是 src/
docker build -t paper-agent src
```

运行容器时需要挂载 FAISS 索引和 Embedding 模型目录，并通过环境变量覆盖 `FAISS_PATH`、`MODEL_PATH`、`DEEPSEEK_API_KEY` 等配置，见 `.env.example` 模板（本地敏感配置在 `.env`，已被 git 忽略）。

## 目录职责

> 以下路径都相对 `src/`（本 README 在仓库根，代码在 `src/`）。

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

## 许可与第三方素材

本仓库代码与文档采用 [MIT 许可](LICENSE)。以下内容**不在仓库内**，因此也不在本许可的授权范围内：

- **第三方论文原文与 PDF**：版权归原作者与出版方。仓库只保存 arXiv id、哈希与评测题（`src/evaluation/v300/`），论文正文需自行按 manifest 下载。
- **模型权重**：bge-m3、bge-reranker-v2-m3 等由各自原仓库发布并遵循其自身许可，需自行下载。
- **本地评测试题与人工复核表**：属于个人评测材料，未随仓库分发。

README 与文档里的性能/指标数字都是本机实测值（口径见「评测指标」一节），随硬件、模型与外部 API 状态变化，不承诺复现。

## 文档入口

- 设计细节（缓存与阈值、过载准入、外网检索、重排、接口定义、成本、上下文、评测指标）：[`src/DESIGN.md`](src/DESIGN.md)
- 接口契约与冻结门槛：[`src/CONTRACT.md`](src/CONTRACT.md)
- 迭代流水、结论沉淀与方向调研：在本地 `docs/`（不进公开仓库）
