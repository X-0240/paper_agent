# 模块2/3 接口契约（修订版 v2）

本契约约束执行层（模块2）与输出层（模块3）的交互，冻结接口，不锁内部实现。

## 一、设计基线

- 工具函数当前为同步函数，返回结果后由 Agent 循环统一写入 State；异步包装留到 P1 并发阶段
- 工具不做“副作用写 State”，保证可单测、可复用
- 成本单位统一为人民币（元），对齐现有 `DAILY_COST_BUDGET` 和 `llm_usage.jsonl`
- 所有引用必须能溯源：PDF 用页码，Word 用段落区间，兜底是 `doc_id + section + chunk_id`

## 二、State 数据结构

```python
class PaperMeta(TypedDict):
    paper_id: str          # 如 arXiv:2301.12345 或本地文件名
    title: str
    authors: List[str]
    year: Optional[int]    # 元数据缺失时为None，不编造
    source_type: Literal["local","external"]
    pdf_path: Optional[str]

class SectionRef(TypedDict):
    name: str              # 章节名，如 "3.1 Model Architecture"
    page: Optional[int]    # 有可靠页码才填；Word/跨页章节为None
    chunk_ids: List[str]   # 该章节下所有Chunk的ID

class PaperCard(TypedDict):
    paper_id: str
    title: str
    abstract: str
    key_findings: List[str]
    methodology: str
    limitations: List[str]
    sections_ref: List[SectionRef]  # 索引，不存全文

class FactItem(TypedDict):
    fact_id: str
    paper_id: str
    year: Optional[int]       # 论文发布年份，用于时间演进判定
    entity: str
    attribute: str
    value: str
    content: str
    source_chunk_id: str      # 禁止为空
    section_name: str
    raw_quote: str            # 原文摘录，不超过100字
    confidence: Literal["high","low"]

class Conflict(TypedDict):
    conflict_id: str
    fact_ids: List[str]
    category: Literal["true_conflict","superseded","misunderstanding","card_error","insufficient_evidence"]
    description: str
    verdict: str
    confidence: Literal["high","low"]
    evidence_supplementary: str

class ReviewReport(TypedDict):
    title: str
    consensus: List[str]
    disagreements: List[Conflict]
    superseded_conclusions: List[str]   # 被新结论显式推翻的历史结论
    open_questions: List[str]
    references: List[str]     # 格式：[Author, Year, §chunk_id]
    conflict_mark_list: List[str]

class AgentState(TypedDict):
    query: str
    complexity: Literal["SIMPLE","COMPLEX"]
    papers: List[PaperMeta]
    candidate_chunks: List[Chunk]
    cards: List[PaperCard]
    facts: List[FactItem]
    conflicts: List[Conflict]
    review: Optional[ReviewReport]
    simple_answer: Optional[str]
    step_count: int
    cost_consumed: float      # 人民币
    trace_log: List[Dict]
    section_cache: Dict[str,str]  # paper_id||section_name -> 章节全文，会话内缓存
    card_cache: Dict[str,PaperCard]
    pending_conflicts: List[Dict]  # 未裁决冲突候选，聚合自facts
    search_count: int
```

强制约束：
- `FactItem.source_chunk_id` 为空则丢弃，不写 State
- `Conflict` 只能由 `verify_claim` 产出，Agent 不能自行构造
- `trace_log` 只追加不覆盖
- `trace_log` 每条必须含 `step / action / input / observation / search_count / step_count / facts_count / conflicts_count`

## 三、6 个工具接口

统一约定：输入输出都是 JSON 可序列化对象；超时在 `config.py` 可配；工具本身不写 State。

### 1. search_papers

- 输入：`query: str`, `limit: int=MAX_PAPER_PER_QUERY`
- 输出：`{"papers": List[PaperMeta], "chunks": List[Chunk]}`
- 实现：复用 `rag_tool.search_papers_structured / search_papers_rerank`；Rerank 内部封装
- 超时：10s

### 2. read_section

- 输入：`paper_id: str`, `section_name: str`
- 输出：`str`
- 实现：优先复用 `rag_tool.read_section`；无缓存时用 `doc_ingest` 解析
- 限制：单次返回不超过 `MAX_CHUNKS_PER_PAPER` 对应文本量
- 超时：10s
- 缓存：命中 `State.section_cache` 直接返回；未命中解析后写入缓存
- 兜底：精确匹配 sections_ref → 模糊匹配（difflib，阈值0.6）→ 仍失败返回 None，Agent 记录 trace_log 后继续

### 3. build_paper_card

- 输入：`paper_id: str`
- 输出：`PaperCard`
- 实现：复用 `agent2_parse.build_paper_card`，输出字段对齐新 `PaperCard`
- 缓存：已存在直接返回，不重复调 LLM
- 超时：60s

### 4. analyze_paper_relations

- 输入：`paper_ids: List[str]`
- 输出：`{"comparison": str, "facts": List[FactItem]}`
- 实现：横向对比复用 `agent2_parse.compare_papers`；事实抽取为新增逻辑
- 质量约束：每个 FactItem 必须带 `source_chunk_id`
- 后置：返回 facts 必须带唯一 fact_id，编排层追加写入 State.facts
- 超时：60s

### 5. verify_claim

- 输入：`fact_a: FactItem`, `fact_b: FactItem`
- 输出：`Conflict`
- 实现：复用 `agent3_review.verify_with_retrieval` 的二次检索与裁决思路；先查本地，外网 P1
- 约束：category 只能是五种枚举之一
- 时间演进：Prompt 必须区分“实质矛盾”和“结论被推翻”；新论文须显式引用并否定同范围旧结论才判 `superseded`，不能只因为发布时间晚就赢
- 后置：裁决完成后编排层更新 pending_conflicts（解决/保留）
- 超时：60s

### 6. write_review

- 输入：`query: str`, `facts: List[FactItem]`, `conflicts: List[Conflict]`, `pending_conflicts: List[Dict]=[]`
- 输出：`ReviewReport`
- 实现：复用 `agent3_review.generate_review` 的综述生成思路，输出字段对齐新结构
- 强制：引用由 facts 确定性生成（`作者/论文, 年份, §chunk_id`），LLM 不编引用
- 降级：pending_conflicts 最多传 `MAX_PENDING_IN_REVIEW` 条，综述中标注“待核实”
- 超时：60s

## 四、旧代码映射（按真实函数名核对）

| 旧代码 | 新契约 | 操作 |
|---|---|---|
| `rag_tool.search_papers_structured` | `search_papers` | 复用适配 |
| `rag_tool.search_papers_rerank` | `search_papers` 内部 | 封装 |
| `rag_tool.read_section` | `read_section` | 复用 |
| `agent2_parse.build_paper_card` | `build_paper_card` | 适配字段 |
| `agent2_parse.compare_papers` | `analyze_paper_relations` | 复用 |
| `agent3_review.verify_with_retrieval` | `verify_claim` | 复用思路 |
| `agent3_review.generate_review` | `write_review` | 适配字段 |
| `agent_react` 的 Thought/Action/Observation | 全部保留 | 不改协议 |

过渡规则：旧文件在重构完成前保留并加 `deprecated` 标记；全量测试通过、新链路可用后再删除，不并存两份线上逻辑。

## 五、Agent 循环衔接

- 沿用 `agent_react` 协议：Thought / Action / Action Input / Observation
- 工具超时抛出 `TimeoutError`，Agent 捕获后写入 trace_log 并优雅降级
- `step_count>=MAX_AGENT_STEP` 强制调用 `write_review`
- `len(facts)>=MAX_FACTS_PER_SESSION` 或 `len(conflicts)>=MAX_CONFLICT_PER_SESSION` 提前进入 `write_review`
- 编排层自动兜底：循环结束后若有卡片但无 facts，自动补一次 `analyze_paper_relations` 再写综述
- 每次 LLM 调用前执行预算检查：
  - 已有 facts → 生成不完整综述，标注“因预算限制未完全展开”
  - 无 facts → 返回“预算不足，请明日再试”

## 六、评测接口

| 维度 | 指标 | 数据 | 输出 | 频次 |
|---|---|---|---|---|
| 检索召回 | QASPER 证据级 Hit Rate / MRR | QASPER 子集 | `{"hit_rate": 待实测, "mrr": 待实测}` | 每次检索改动 |
| 多格式通过率 | 解析成功 / 总样例 | multi_format_samples | `{"pass_rate": 待实测, "failures": []}` | 每次解析改动 |
| 冲突检测 | P / R / F1 | 自标 20-40 对 | `{"p": 待实测, "r": 待实测, "f1": 待实测}` | 手动 |
| 综述质量 | LLM-as-Judge | 自标 20 条 query | `{"avg_score": 待实测, "dimensions": {}}` | 手动 |

评测结果写 `eval_results/`，JSON + 时间戳。契约里不出现编造的指标数字。

## 七、常量（config.py）

```python
MAX_PAPER_PER_QUERY=10
MAX_CHUNKS_PER_PAPER=20
MAX_AGENT_STEP=15
MAX_FACTS_PER_SESSION=20
MAX_CONFLICT_PER_SESSION=5
MAX_EXTERNAL_SEARCH_NUM=3
MAX_SEARCH_PER_SESSION=3
MAX_PENDING_IN_REVIEW=3
MAX_CARDS_TEXT_CHARS=12000
DAILY_COST_BUDGET=5.0      # 元，可被.env覆盖
SESSION_COST_BUDGET=1.0    # 元
```

## 八、已确认决策

1. `FactItem.source_chunk_id` 强制非空：接受
2. 旧 agent2/agent3 不立即删除：先标记 deprecated，新链路测试通过后再删
3. 预算不足时生成不完整综述：接受
4. `MAX_AGENT_STEP=15`：接受（实测10步会被检索/建卡耗尽，无法进入事实抽取），成本护栏兜底
5. 冲突评测：自标 20-40 对，不用公开数据集，避免编造

## 九、外网检索并发（web_search，2026-08-26）

### 目标

simple 路径本地 + 外网双路并行检索，外网源为 Wikipedia REST API 与 arXiv API（免费无 key），失败降级为本地-only，整条链路不挂。

### 并发与生命周期（AI#3 审查后定稿）

- `/ask` 改为 `async def`，禁止在同步端点里用 `asyncio.run()`；SSE 生成器直接 `await` 同一套异步函数
- 本地检索用 `asyncio.to_thread()`；外网用 httpx AsyncClient，`async with` 局部创建，不跨事件循环复用
- `asyncio.Semaphore` 作为模块级懒加载单例，在事件循环内首次创建，默认并发 5（`WEB_SEARCH_CONCURRENCY`）
- 每个外网子任务独立 `asyncio.wait_for`，超时默认 8 秒（`WEB_SEARCH_TIMEOUT`）；子协程内部捕获所有异常并返回空列表，不向外抛
- arXiv 请求间隔 ≥3 秒，请求头带合规 User-Agent

### 统一来源字段（固定 key，空值填 None）

```python
{"source_type":"local|wikipedia|arxiv","title":str,"snippet":str,
 "paper_id":Optional[str],"section":Optional[str],
 "url":Optional[str],"arxiv_id":Optional[str]}
```

### 输入输出

- 输入：`query`、`top_k`
- 输出：`{"sources":[统一来源], "context":"本地在前、外网在后"}`
- 不去重；外网全失败时只含本地结果

### 环境变量

`WEB_SEARCH_TIMEOUT=8`、`WEB_SEARCH_MAX_RESULTS=3`、`WEB_SEARCH_CONCURRENCY=5`

### 验证

- 单测用 `unittest.mock.AsyncMock` patch `httpx.AsyncClient.get`：正常、超时、429、SSL、空结果、并发上限、中文查询
- 集成实测：Wikipedia/arXiv 各真实调用一次；断网或 API 失败时仍返回本地结果
- 全量 pytest 保持通过

## 十、数据与评测方案（Phase 0-4，2026-09-16）

### 目标与约束

- 主指标只有一个：证据级 HitRate@5，@10 作为同口径曲线；章节级/论文级只做内部诊断
- 证据级定义：top-k 片段中出现任一标注证据句（NFKC 归一化后匹配）
- 目标两档：可信目标 75%（点估计），挑战目标 80%（复合里程碑，不作为"与 75% 统计显著"的结论）
- 全程禁止在 test 上调参；dev/test 按论文分组，论文集尽量不相交
- 语料 Phase 1 冻结为 150 篇（50 基础经典 + 100 近期相关），Phase 4 再做每周自动更新

### 评测协议

- 出题：test 集必须读整篇论文正向出题，禁止从 chunk 反向生成；反向生成只允许用于 dev 调试
- 盲法：出题人不见切片结果和索引；证据规则预注册为 1-3 句，按"任一合理证据"判定
- 一致性：20% 题目双标注，报告 Cohen's kappa；kappa < 0.7 先统一标注标准再评测
- 题型：事实 25%、方法 25%、实验 15%、局限 10%、对比 15%、多跳 10%；对比+多跳合计不低于 30%
- 切分：Phase 0 用现有 88 题做 dev 同集 A/B；Phase 1 目标 50 dev / 150 test，单篇论文出题 ≤1-2 题
- 外部验证：QASPER 263 题为段落级证据，与句级指标口径不同，只做方向性 sanity check，不并入合并统计
- 报告：所有比例指标附 Wilson 95% 置信区间和样本量；预注册主指标与阈值，禁止事后改阈值

### Phase 0 门槛（约 2 周，冻结现有 50 篇 + 88 题）

- 主门槛：88 题 dev 上旧链路 vs 新链路同集 A/B，证据级@5 提升 Δ≥5-7pp（或配对检验显著）
- 辅门槛：40 题正向出题集绝对底线 ≥65%
- 泄露量化：同一 40 题、同一索引，反向出题版 vs 正向出题版各跑一次，gap >10pp 则先修出题流程
- Phase 0 三个杠杆：结构完整性切片（表格/图注/公式不切碎）、多通道召回与融合（向量/BM25/章节标题/表格）、跨语言改写+多 query

### Phase 1-4

- Phase 1：冻结 150 篇语料；建 50 dev / 150 test + 30-50 域外题；双模型校验 + 人工复核；冻结快照并跑基线
- Phase 2：统一重构 RetrievalService（查询理解 → 多通道召回 → 融合 → 章节聚合 → 证据选择），生产和评测共用
- Phase 3：在冻结 test + QASPER + 域外题上验收；75% 可信、80% 复合里程碑
- Phase 4：每周自动更新独立迭代，带灰度校验、快照指纹和一键回滚

### 快照指纹

`snapshot_fingerprint = hash(chunk_ids, embedding_model, chunking_version, pdf_parser_version, dedup_rule_version, metadata_schema_version, question_set_version, evidence_schema_version, retrieval_config, llm_judge_version)`

### 数据分层

- 公开仓库：论文 id 列表、元数据、抓取/解析/评测脚本
- 本地保存并 gitignore：PDF、切片、FAISS/BM25 索引、快照文件

### 灰度与回滚

- 固定回归集 ≥100 题；周更后点估计下降 ≥2pp，或表格/图注子组相对下降 ≥15% 即阻断
- 保留 Phase 1 冻结快照作为基线，支持一键回滚

### 已知限制

- 150 test 观测 80% 时单侧 95% 下界约 74%，80% 只能作为点估计/复合里程碑
- 区分 75% 与 80% 统计显著需要约 560 题，当前规模做不到，报告中必须写清
- 人工正向标注成本约 250-350 人时；Phase 0 先用 LLM 辅助 + 人工复核压缩成本

## 十二、统一检索服务（2026-09-17）

### 目标与冻结口径

- 唯一目标：把当前离线英文 query 的质量收益变成可部署能力，不换索引、embedding、reranker 或向量库
- 统一 Service 的原中文 query 基线：88 题 34/88；旧离线口径为 35/88，42 题为 27/42
- 普通直译 query 的离线上限：54/88
- 检索专用英文 query 的离线上限：55/88
- Top50 全量重排未通过：统一 Service 中文重排 51/88，英文重排 57/88，分数融合最好 59/88
- 0.1 分差条件重排：53/88；当前可见特征无法复现“只重排失败题”的离线 oracle 结果
- 离线英文 query 只根据问题生成，不看论文、证据句、章节或正确论文信息
- 离线预生成 query 仅用于评测上限对照，不进入生产 Service

### RetrievalService 接口

```python
search(question,candidate_k=25,rerank_mode="conditional",top_k=5,search_query=None) -> list[dict]
search_async(question,...) -> list[dict]
```

- 生产只暴露 `search/search_async` 两个入口
- `search_query` 仅供评测注入预生成 query；生产默认不传，由 Service 内部决定原问题或生成query
- 召回、重排、去重、Top-K 和返回字段在生产和评测中共用同一实现
- 不再为查询来源建立公开类型或四套模式

### 结果契约

每条结果至少包含：

```text
source
section
text
recall_score
rerank_score
chunk_id
final_rank
retrieval_query
rerank_triggered
fallback_reason
service_version
```

- `text` 返回完整原始切片，输出给 Agent 或 WebUI 时再由调用方截断
- 论文标题、章节名和查询前缀只允许进入 `retrieval_text`，不得拼接进证据 `text`
- `chunk_id` 从索引元数据透传；缺失时由 `source + section + index + text hash` 生成稳定 ID
- 重排后只去完全重复 chunk；不在 Top-K 内执行章节限流

### 查询生成与缓存

- 纯英文高质量问题直接使用原问题
- 含中文问题且显式开启生成时，使用检索专用英文 query
- 生成 prompt 保留实体、缩写、术语、数值、比较关系和任务目标，优先论文原词，不逐字直译
- 生成客户端关闭 thinking、单次请求、不重试、3 秒超时、`max_tokens=120`
- 缓存 key 包含规范化问题、规范化版本、供应商、模型、prompt 哈希、temperature、max_tokens、目标语言和校验规则版本
- 相同完整缓存 key 使用 single-flight，避免并发重复调用
- 含中文、多行、Markdown、解释文本、超长内容判为非法，不写正向缓存
- 生成失败、超时、预算不足或非法输出时回退原问题，并写短 TTL 负缓存

### 预算与并发

- 查询生成调用前通过统一预算账本原子预留，成功后按 usage 结算，失败释放
- 查询生成低层客户端不重复检查预算，也不重复记账
- LLM 生成与本地 CPU 推理使用独立 Semaphore
- API 异步入口使用 `asyncio.to_thread()` 调用同步 Service，事件循环不执行模型推理
- 同一完整缓存 key 的并发请求只允许一次生成

### 召回与重排

- BCEmbedding 与 BM25 只使用内部得到的实际检索 query
- Top50 是候选能力，不代表已经通过生产质量验收
- 重排 query 使用原始问题，语言差异作为独立 A/B，不混入 P0 切换
- P0 不执行旧 `filter_results` 的章节限流和重叠过滤；如后续启用，生产与评测必须同步并重新测量
- 全量重排、条件重排和融合重排均必须重新达到验收门槛后才允许启用
- 当前生产默认保持 `RETRIEVAL_CANDIDATES=25`、`RERANK_MODE=conditional`、`QUERY_GENERATION_ENABLED=0`

### 调用边界

- `rag_tool` 保留字符串、结构化列表和重排列表三种旧接口，内部全部委托 Service
- `tools.search_papers` 只做 State 结构转换，不再决定检索实现
- `web_search` 继续负责 Wikipedia/arXiv 并发，本地检索委托 `RetrievalService.search_async`
- `read_section` 保持章节读取职责，不经过查询生成
- `evaluation.Retriever` 改为 Service 适配器，不再维护平行 BM25、FAISS 和排序实现

### 上下文与成本边界

- Agent 工具 Observation 单条截断 300 字，回答 LLM 的本地来源预算不超过现有 1500 字
- 评测使用完整 `text` 做证据匹配
- query 生成冷缓存失败率必须低于 5%，不因生成失败中断 API

### 验收门槛

- 最终验收必须使用运行时生成 query，离线预生成 query 只能报告上限
- 88 题证据级@5 至少 66/88（75.0%）
- 42 题回归至少 36/42，同时报告相对离线 39/42 的逐题胜负
- 报告配对胜负矩阵、Wilson CI、query 失败率、缓存命中率、冷缓存生成延迟、重排延迟和本地总延迟
- query 生成新增 p95 不高于约 2 秒，本地检索总 p95 不高于约 4 秒
- 当前尚未达到门槛，禁止把离线 77.3% 或 92.9% 写成生产结果

### 当前实现状态

- 已完成统一 `RetrievalService`、旧 `rag_tool` 委托层、稳定 chunk_id、查询计划缓存、预算预留/结算和评测适配
- 新能力默认关闭，生产行为保持旧候选 25、条件重排和原问题查询
- 尚未完成：找到不依赖 ground truth 的可靠重排触发或融合策略，使统一 Service 达到 75%

### 回滚

- `QUERY_GENERATION_ENABLED=0`：回退原问题 query
- `RERANK_MODE=always/conditional`：切换重排模式
- `RETRIEVAL_CANDIDATES=25/50`：切换候选池
- 配置在进程启动时加载，切换需要重启，不实现热切换
- 回滚不修改索引，不需要重建 FAISS
