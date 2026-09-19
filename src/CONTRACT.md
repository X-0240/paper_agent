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
- 生产默认值的**唯一权威**是第十五节「当前默认配置」；本节不再复述具体数值，避免两处各自演化（2026-09-20 整合）

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

### 历史门槛（2026-09-17，已退役）

以下门槛是统一 Service 这一轮为**旧 10 篇语料**（88 题与 42 题）预注册的，v300 语料换代后不再作为当前验收条件，保留原文以便追溯：

- 最终验收必须使用运行时生成 query，离线预生成 query 只能报告上限
- 88 题证据级@5 至少 66/88（75.0%）
- 42 题回归至少 36/42，同时报告相对离线 39/42 的逐题胜负
- 报告配对胜负矩阵、Wilson CI、query 失败率、缓存命中率、冷缓存生成延迟、重排延迟和本地总延迟
- query 生成新增 p95 不高于约 2 秒，本地检索总 p95 不高于约 4 秒
- 当前尚未达到门槛，禁止把离线 77.3% 或 92.9% 写成生产结果

退役说明与补跑记录（2026-09-19）：该门槛指向旧题集，与 v300 语料的题目口径不同，不可混用。用当前配置补跑旧题集的结果是 88 题 45/88、42 题 32/42，两项均未达到原定值；延迟项中冷缓存 p95 3717ms、热缓存 162ms 通过，query 生成新增延迟约 3.5s 高于原定 2s。v300 体系当前没有预注册门槛，因此 v300 的结论只有相对证据（同批对照与单变量消融），不宣称通过验收。

### 当前默认配置（2026-09-19）

- 当前默认：`RETRIEVAL_CANDIDATES=50`、`QUERY_GENERATION_ENABLED=1`、`RERANK_MODE=conditional`，与 `.env` 实际值一致
- 依据：同一批 202 题（v300 test）配对对照与单变量消融——运行时英文 query 生成 146 → 172（+12.9pp，精确 McNemar p<0.0001）、条件重排 164 → 172（+4.0pp，p=0.0215）、候选池 25 与 50 在 top_k=10 口径下净 -2 题无收益
- 候选池保持 50 的理由：25 与 50 的对照只在 top_k=10 口径下测过，尚未在 top_k=5 生产口径复测，暂不调整默认值
- 降级路径：`QUERY_GENERATION_ENABLED=0` 回退到原问题检索（同批 202 题为 146/202），该路径不依赖外部 LLM，用于查询生成超时或失败时保证链路可用
- 尚未完成：不依赖 ground truth 的可靠重排触发策略；把结论从"相对证据"提升为"预注册验收"需要新封存测试集

### 查询计划三种形态（术语统一）

项目历史上"改写"一词指过三件事，文档与话术统一按下表表述：

- 中文改写策略（2026-08-19，已放弃）：保留中文并追加英文关键词等做法，在论文级命中率上负优化（93% → 92% / 91%），不再使用
- 离线英文改写（2026-08-26，仅作对照）：双 AI 预生成的英文 query，用于评测上限与对照，不进生产链路
- 运行时英文 query 生成（当前默认）：由 `RetrievalService` 在运行时把中文问题生成一行英文检索 query，带缓存、3 秒超时、预算预留与失败回退

### 当前实现状态

- 已完成统一 `RetrievalService`、旧 `rag_tool` 委托层、稳定 chunk_id、查询计划缓存、预算预留/结算和评测适配
- 查询生成已从"候选能力"转为当前默认，理由与数据见上一节；`QUERY_GENERATION_ENABLED=0` 保留为降级路径
- 尚未完成：找到不依赖 ground truth 的可靠重排触发或融合策略，使统一 Service 达到 75%

## 十三、v150 语料与评测集（2026-09-17）

### 目标与边界

- 合并现有 50 篇与新增 100 篇组成 150 篇冻结语料
- 新增论文按主题为主、年份配额分层；不为追求“最新”堆满 2024-2026
- 现有 88 题和 42 题只作为历史回归集，不参与新 test 口径
- 建立唯一版本化 v150 流水线，旧脚本只保留历史入口
- QASPER 只作 sanity/regression，不混入主指标

### Manifest 契约

每篇论文必须记录：

```text
paper_id
source_family
arxiv_id
arxiv_version
title
published_at
updated_at
primary_category
selection_stratum
series_id
license
pdf_sha256
downloaded_at
parse_status
```

- 现有 10 篇本地 PDF 和 40 篇 QASPER 必须回填同等级字段
- `series_id` 使用可确定规则，不靠人工模糊判断同系列
- arXiv 论文固定具体版本，不记录漂移的基础 id

### 分层与 split

- 主分层变量：主题
- 年份只做配额：约 70 篇 2024-2026，约 30 篇 2020-2023
- 来源、论文长度、解析器类型和 `series_id` 作为分布检查变量
- 合并 150 篇后切分 50 篇 dev、100 篇 test，论文不重叠
- 同论文不同版本及同 series 不得跨 split

### 出题与审核

- 出题只允许读取完整论文，禁止读取 chunk、检索排名、gold section 或 source query
- dev 约 50-60 题，test 约 140-150 题，总候选 250-300 后剔除至约 200
- 主 test 以单论文、单证据组问题为主
- 跨论文对比和多跳使用 required_papers/evidence_groups 表达，全证据覆盖单独报告
- 审核人不得查看检索排名或模型输出，争议题按预注册规则仲裁

### 实际审核状态

- v150 已完成的 dev/test 审核由豆包/AI完成，属于 `model_review`，不是人工复核
- 现有 v150 只能表述为自动证据审计 + 豆包/模型复核，不能写成人工标注或人工复核
- 后续 v300 计划增加人工抽检和核心题人工确认，完成前必须标为待审核，不能冒充已验证数据

### 版本与验收

- 索引快照使用独立路径，不覆盖现有 FAISS
- 快照指纹包含 parser、chunk、embedding、manifest 和 question 版本
- dev 调优完成后封存 test；启动一次密封验收任务
- 同批次计算旧基线与候选方案，结果封存后才允许查看
- test 运行后不得修改；修正必须提升版本并全量重跑
- 150 题上的 80% 只能作为点估计，不能宣称强统计结论

### 回滚

- `QUERY_GENERATION_ENABLED=0`：回退原问题 query
- `RERANK_MODE=always/conditional`：切换重排模式
- `RETRIEVAL_CANDIDATES=25/50`：切换候选池
- 配置在进程启动时加载，切换需要重启，不实现热切换
- 回滚不修改索引，不需要重建 FAISS

## 十四、v300 复现与归因（2026-09-19）

### 复现结论

- 产生 v300 密封结果的 runner 此前不在仓库内（只有读 sealed 文件的脚本，没有写它的脚本）。本轮补 `evaluation/repro_v300.py`，密封结果可用仓库命令复现
- 输入基底校验：`freeze.json` 记录的 index、index_meta、manifest、六个题库文件、sealed 结果 sha256 与当前文件全部一致（10/10）
- 反推出的密封配置：检索 query 使用 sealed 结果文件里落盘的英文 query，`RETRIEVAL_CANDIDATES=50`，`RERANK_MODE=conditional`，评测调用 `top_k=10`（一次调用同时算 hit5 与 hit10）
- 复现结果：202 题 hit5 与 hit10 与密封文件逐题完全一致（202/202）
- 密封 baseline 未完全对齐：最接近的配置是"原问题 query + 候选 50 + 条件重排"，202 题中 195 题布尔一致、命中数 146 对 147。baseline 的精确配置无法从仓库现有信息还原，对外只引用候选口径

### 调用口径（重要）

- 生产调用是 `top_k=5`，密封评测用 `top_k=10`；条件重排的触发判据是比较第 1 名与第 `top_k` 名的分数差，两个口径的触发率因此不同：`top_k=10` 为 3.5%，`top_k=5` 为 20.8%（与 2026-09-17 记录的 20.6% 一致）
- 结论：密封评测低估了生产配置的表现。同一批 202 题、生产口径下，基线（同配置关重排）164/202=81.2%，候选（条件重排）172/202=85.1%
- 对外报数必须写明调用口径（top_k 与候选数），否则不同来源的数字不可比

### 单变量消融（202 题，同一索引与题库）

- query 生成（原问题 → 运行时生成英文 query，重排均开启）：146 → 172，净增 26 题（+12.9pp），精确 McNemar p<0.0001
- 条件重排（同 query，开与关，top_k=5）：164 → 172，净增 8 题（+4.0pp），精确 McNemar p=0.0215
- 候选池 25 到 50（top_k=10 口径）：净 -2 题，无收益
- 失败面（生产口径候选失败 30 题）：19 题金标不在向量 Top100、11 题不在 BM25 Top100、8 题两路 Top100 都未召回；只有 1 题是重排把已召回的金标挤出 Top5

### 复核剔除题的偏置量化

- 202 = 211 经模型复核剔除 9 题后的保留集；9 条剔除理由逐条记录在 `review_reason`（数据卫生：剔除理由写在 `review_reason`，`reject_reason` 为空，过滤剔除题不能用 `reject_reason`）
- 9 题回测（生产口径）：候选 9/9、基线 9/9、原问题且关重排 8/9；放回 211 口径后命中率不降反升，候选 181/211=85.8%（202 口径为 85.1%），基线 173/211=82.0%
- 结论：复核剔除没有抬高命中率，"筛选测试集刷指标"的质疑在数据上不成立。9 题样本过小（95%CI 70-100%），只作方向性结论

### 门禁补检（当前配置、运行时生成 query）

- 88 题证据级@5 = 45/88，门槛 66/88，未通过（差 21 题）
- 42 题回归@5 = 32/42，门槛 36/42，未通过（差 4 题）
- 延迟：冷缓存 p95 3717ms（门槛不高于 4000ms，通过）、热缓存 p95 162ms
- query 生成新增延迟近似 3.5s（冷热缓存差），高于 2s 门槛；该值未单独埋点，属近似口径
- 结论：门禁未通过，与 2026-09-17 记录一致（当时 45/88、30/42；本次 42 题从 30 提升到 32）。v300 只能定位为本地演示/预发布，不得表述为通过生产验收
- 演示配置（`.env` 的 `RETRIEVAL_CANDIDATES=50`、`QUERY_GENERATION_ENABLED=1`）是在门禁未通过的条件下启用的，属于演示口径，文档与话术必须保持这一表述

### 证据充分性判读

- 判读对象是生产口径检索回来的 Top5 证据，rubric 事前冻结（sha 记录在结果文件），模型按 rubric 判读，人工抽判复用抽检表
- 结果：50 题抽样中 34 题证据充分；仅候选命中 16/18、两边都中 6/8、两边都没中 8/14、仅基线命中 4/10
- 该指标只叫"证据充分性"，不得表述为答案准确率

### 人工证据抽检（50 题分层，2026-09-19）

- 抽样：仅基线命中 10 题全抽、仅候选命中 18 题、两边都没中 14 题、两边都中 8 题；判定标准是"金标证据句能否在论文原文找到并支撑答案"
- 结果：47 题完全支撑，3 题判"原文支持弱"（3/50=6.0%，95%CI 2.1-16.2%）；未发现"金标句不在原文""章节错""论文错"
- 3 例的共同模式是枚举题漏条目，金标句本身都能在原文逐字定位，但问题要求枚举、金标只覆盖其中一部分：
  - 第 8 题（2609.18677，仅基线命中桶）：问"对哪些组件做了消融"，原文 Section 5 列出四个变体（no temperature correction 12.3%、all tokens 64.4%、text only 63.8%、group size 8 67.0%），金标只写温度校正
  - 第 22 题（Chain_of_Thought，仅候选命中桶）：原文 Discussion 局限段有 first/Second/Third/Finally 四条，金标只覆盖前三条，漏掉 Finally（只在模型规模足够大时才涌现、真实部署成本高、如何在小模型上诱导推理）
  - 第 34 题（2401.00112，两边都没中桶）：预处理四步里金标只写①重采样②删全空记录④归一化，漏③删除船只静止状态记录（电池电压为空、SoG 极小）
- 对主指标的影响方向：3 例中 1 例落在"仅候选命中"（可能抬高候选口径，上界 2.0%，95%CI 0.4-10.5%）、1 例落在"仅基线命中"（抬高的是基线口径）、1 例两边都没中（不影响任何一方）。也就是说，抽检未发现候选指标的显著虚高
- 暴露的真实问题：复核阶段对枚举题采用"关键句"级别，人工抽检采用"完整清单"级别，两次口径不一致。文档统一规范为"金标需覆盖问题的全部核心信息点；枚举题不能只给关键句"，后续版本按此标准重审
- 另发现章节归属问题：部分论文的解析章节标题不可靠（例：TUCANA 的预处理步骤被切进标题为 Abstract 的章节）。该问题只影响章节级诊断指标，不影响证据级主指标
- 结论适用范围仅限这 50 题，不外推为整体标注准确率

### 复现命令

- 复现与消融：`python evaluation/repro_v300.py --configs prod_baseline_k5,prod_candidate_k5,abl_querygen_off,sealed_candidate --extra-questions evaluation/questions/v300_test_valid.json`
- 抽检表：`python evaluation/build_label_audit.py`
- 证据充分性：`python evaluation/evidence_sufficiency.py --repro evaluation/results/v300_repro_prod.json --config prod_candidate_k5`
- 门禁补检：`python evaluation/gate_check.py`
- 运行前先停掉 API 服务：8G 显存同时加载演示服务、嵌入模型与重排模型会 OOM，重排模型不可用时链路会静默降级成"不重排"，导致结论失真

## 十五、持久化与查询接口（2026-09-19）

### 目标与边界

- 目标：多轮会话与任务状态落盘，使多轮对话可续接、任务状态可追溯
- 一期不做：断点续跑、工具调用与成本落库、审批队列、`/metrics`（均为二期）；不做多机部署、Redis/Kafka
- 不改检索与评测口径；多轮改写默认关闭，关闭时检索输入与改造前一致

### 表结构

- `threads`：thread_id 主键、user、created_at、updated_at
- `messages`：message_id 自增主键、thread_id 外键、role、message_type（user_query / assistant_answer / task_result / system_error）、content、sources_json、rewritten_query、created_at
- `tasks`：task_id 主键、thread_id 外键、user、route、status（running / succeeded / failed / cancelled）、created_at、started_at、finished_at、error_message、checkpoint
- `schema_version`：单表记录迁移版本，取值用 MAX(version)
- 消息优先级不单独存列，由 message_type 在一处映射派生；耗时由起止时间计算，不存冗余列

### 数据访问契约

- `store.py` 是唯一数据访问层，其他模块不得自行 `sqlite3.connect`
- 连接按线程缓存（thread-local）；连接参数：`busy_timeout=5000`、`journal_mode=WAL`、`synchronous=NORMAL`、`foreign_keys=ON`
- 数据库调用一律写成同步函数，**事务内不得出现 await**；异步路径统一用 `asyncio.to_thread` 调用
- 所有按 thread 或 task 的查询在数据层强制带 user 过滤，查不到或不属于该用户都返回 None，由接口层统一转 404（不返回 403，避免泄露资源存在性）
- 一个请求内的多次写应合并到同一事务，减少写锁获取次数

### 接口

- `POST /ask` 与 `GET /ask/stream` 增加可选 `thread_id`，缺省新建会话；响应返回 `thread_id` 与 `task_id`，流式接口通过首个 `thread` 事件返回
- `GET /tasks/{task_id}`：任务状态、起止时间、错误信息
- `GET /threads/{thread_id}/messages?limit=`：按会话回放消息，含来源摘要
- `GET /threads/{thread_id}/tasks`：该会话下的任务列表

### 流式事件契约

- 事件类型与顺序：`thread`（会话与任务 id）→ `route` → `sources` 或 `status` → 若干 `progress`/`ping` → 若干 `token` → `done`；异常时插入 `error` 且仍以 `done` 收尾
- 简单问答走真流式：模型每产生一个 token 就推一个 `token` 事件
- 综述链路不是 token 流：它是多步任务，中间产物是结构化数据（卡片/事实/冲突），因此以 `progress` 事件表达阶段进度，字段为工具中文名加当前状态计数（论文数、卡片数、事实数、冲突数）
- 心跳：`progress` 之间超过 `SSE_HEARTBEAT_SECONDS`（默认 10 秒）没有事件时推一条 `ping`，防止中间层或浏览器把静默判成超时断开
- 跨线程约束：进度回调在 `asyncio.to_thread` 的工作线程里触发，必须先用 `loop.call_soon_threadsafe` 唤醒事件循环；进度内容先追加到列表、再发唤醒信号，避免最后一条进度在任务结束瞬间丢失
- 进度文案只在编排层定义一处（工具名到中文标签的映射），未知工具名直接显示原始名，便于发现映射遗漏

### 多轮检索改写

- 开关 `MULTI_TURN_QUERY_REWRITE`，默认 0。关闭时即使传入 thread_id 也走单轮路径，检索输入与改造前完全一致
- 打开且该会话存在历史时，用最近 3 轮加当前问题补全成自足问题，只用于检索；回答仍使用原问题；实际使用的检索问题记入 `rewritten_query`
- 该能力尚未量化评测，默认关闭。开启前需按 10-20 条多轮样例覆盖三类场景（明确指代、无指代追问、话题切换），逐条对比单轮与多轮路径的检索结果后再决定

### 迁移规则

- 迁移文件按 `migrations/NNN_*.sql` 编号，一期只有 `001_initial.sql`
- 迁移与版本号在同一事务内提交（Python 的 `executescript` 会先隐式提交，因此把 `BEGIN`/`COMMIT` 写进同一个脚本）
- `MAX_KNOWN_SCHEMA_VERSION` 硬编码，并与迁移目录中的最高编号核对；库版本高于代码已知版本时拒绝启动，且该检查在迁移执行之前
- 不使用 `CREATE TABLE IF NOT EXISTS` 做版本管理，避免无法区分"未建表"和"表被删"

### 关键坑

- Python 的 sqlite3 连接不能跨线程共享；`check_same_thread=False` 只是关闭守卫，并不会让连接线程安全
- WAL 模式下读不阻塞写，但写与写仍然串行；长事务会放大锁竞争
- Python 的 `connect` 默认 `timeout=5.0`（实测 `PRAGMA busy_timeout` 返回 5000）。代码里显式声明是为了防止后续调整 connect 参数时静默改变锁等待行为，不是修默认值；C 语言版 SQLite API 的默认值才是 0
- 事件循环只在 `await` 处切换协程，所以"事务内不得 await"是 thread-local 连接成立的前提；一旦有人在事务里 await，同线程的另一个协程就可能插入并交叉事务
- 对话落库不等于注入上下文：二期若做历史注入，必须经过独立的纯函数裁剪，不得直接把 `messages` 查询结果拼进 prompt；一期答案生成不吃历史，只改检索输入

### 运行时指标接口（2026-09-19）

- `GET /metrics?since_hours=24`：按登录用户返回任务状态分布、按路由分布、成功率、平均与 p95 耗时、消息类型计数与会话数
- 指标只统计当前用户自己的数据，与其他查询接口共用同一套归属过滤
- 同时返回进程级信息：进程运行时长、schema 版本，以及当前检索配置（候选数、重排模式、查询生成、点名加权、多轮改写开关），用于确认演示跑的是哪套配置
- 耗时由 `tasks` 的 started_at 与 finished_at 计算，不存冗余列
- 已知口径：uptime 从模块导入时刻起算，不含导入耗时；由于 uvicorn 先绑定端口再导入应用，刚启动时该值偏小

### 逐篇事实抽取并发（2026-09-19）

- 背景：综述链路里逐篇事实抽取是最大耗时项，单次实测占整轮 62%（3 篇论文 83 秒），且原实现是串行
- 做法：`analyze_paper_relations` 对多篇卡片用线程池并发抽取，并发度由 `EXTRACT_CONCURRENCY` 控制（默认 3）；单篇或并发度设为 1 时退回串行路径
- 为什么用线程而不是协程：抽取是"发请求后等待网络"，属 IO 等待，线程可让等待重叠；协程对 CPU 密集型无效这一条在这里不适用，因为瓶颈不在本地计算
- 失败隔离：单篇失败只记 warning 并把该篇事实记为 0，不影响其他篇；进度在"完成时"上报（`抽取事实 i/n`），并发下顺序按完成先后
- 实测：2 篇论文抽取从串行约 60-76 秒降到 42 秒（两篇完成时间相差 8.4 秒，说明确实并行）；整轮从 90-105 秒降到 62 秒（8 篇论文、2 张卡片、15 条事实）；单篇场景无收益（并发无从叠加）
- 与既有并发设计的区别：外网检索用的是 asyncio（IO 等待 + Semaphore 限流 + 单源失败降级），这里用的是线程池（同步代码内并发），两者各自适配自己的调用形态

### 成本与工具调用归因（2026-09-19）

- 迁移 002 新增两张表：`costs`（每次 LLM 调用的模型、token、金额、task_id）与 `tool_calls`（工具名、参数摘要、成功与否、耗时、返回体量）
- 归因通过 `run_context.py` 的 contextvar 传递当前 task_id/thread_id/user；`asyncio.to_thread` 会复制上下文，因此工作线程内可见；同步路由里设置的上下文不会带进事件循环任务，流式接口在生成器内部设置
- `llm_usage.jsonl` 仍是成本与每日预算的权威（每日硬闸读它），`costs` 表只做按任务与按用户的归因查询；这样不动预算链路，代价是存在双写不一致的可能，需要时按 jsonl 对账
- 工具归因覆盖两类路径：ReAct 循环内经 `on_step` 记录；兜底路径（强制检索、自动抽取）单独记一笔，参数摘要用 `{"source":"fallback"}` 标注
- 归因写入全部包在 try 里：写库失败只打 warning，绝不影响主流程（实测踩过一次 `json` 未导入导致全部写入失败，主流程未受影响，这正是该设计的作用）
- 查询接口：`GET /tasks/{id}` 返回 `attribution`（金额、tokens、LLM 调用数、工具调用数与明细）；`GET /metrics` 返回窗口内成本与工具调用汇总
- 查询生成也计入账：`_generate_query` 走 `reserve_budget` 与 `settle_budget`，而 `settle_budget` 无条件调用 `log_usage`，因此即使调用侧传了 `record_usage=False`，这笔成本仍会写进 jsonl 与归因表（实测约 ¥0.0002-0.0003/次，比答案生成低一个数量级）
- 注意解释口径：查询结果有持久化缓存，同一个问题第二次提问不会产生生成调用，因此单次任务的归因里可能只看到 1 次 LLM 调用，这是缓存命中而不是漏记

### 排序与覆盖率指标（2026-09-19）

- `evaluation/metrics.py` 新增 `gold_rank`、`mrr`、`evidence_coverage`、`ndcg_at_k` 四个纯函数，供评测与报告共用
- 202 题封存集、生产口径（候选 50、条件重排、top_k=5）实测：MRR=0.6847、HitRate@5=0.8515、NDCG@5=0.7206、证据覆盖率 0.8292（单证据题 74 题为 0.8243、多证据题 128 题为 0.8320）
- HitRate@5 与既有的 85.1% 一致，作为指标实现的交叉校验
- 报告脚本：`python evaluation/rank_metrics.py`

### 论文级筛选实验（2026-09-19，结论：不启用）

- 动机：问题不点名论文时，检索层缺少"这题在问哪篇论文"的概念，来源可能跑偏
- 做法：新增 `PAPER_RERANK` 开关（默认 0），开启时用一次 LLM 调用从已召回论文里选出 1-3 篇相关论文，对它们的切片加权并跳过重排；只从已召回集合里选，不引入新候选
- 实测（202 题封存集，同口径）：证据级@5 85.1%→82.2%（少 6 题）、首位来源 93.1%→86.1%（少 14 题）、MRR 0.6833→0.6252、覆盖率 0.8292→0.7962；154/202 题能被选出相关论文，因此不是覆盖率不足，而是**模型判断的"相关论文"和数据集标注的目标论文经常不一致**
- 结论：默认关闭，保留为可切换能力与负面结论证据；来源归属问题不能靠"让模型猜哪篇论文"解决

### 点名论文加权（2026-09-19）

- 问题：问题点名某模型或论文时，相关但不直接的论文切片会排在前。实测"BERT 的输入最大长度"：向量路 BERT 自身切片第 1（归一 1.000），BM25 路 RoFormer 切片第 1（1.000）、BERT 只有 0.632，融合后 RoFormer 0.970 反超 BERT 0.816
- 开关与取值：`ENTITY_BOOST`（默认 1 启用）、`ENTITY_BOOST_VALUE`（默认 0.2，融合分增量）。关闭时检索行为与改造前逐字节一致
- 机制一：识别到的点名论文切片在融合分上加权。软加权而非硬过滤，避免"对比 A 和 B"这类多论文问题被砍掉一半
- 机制二：点名论文时跳过重排。实测重排会把已召回的正确切片压低（同一查询下 RoFormer 段落 CE 0.963、BERT 段落 0.645），点名场景下检索范围已被用户限定，重排的收益小于它带来的错位
- 点名识别收敛到 `paper_entities.py` 单一实现：别名字典（模型常用叫法）＋语料论文名直接匹配＋arXiv id 匹配；旧链路 `agent1_retrieve` 改为委托，不再自带一份别名表
- 别名匹配用连字符边界：`BERT-NER` 不再命中原版 BERT，`GPT-6` 不再命中 GPT2。这是实测发现的误报，修之前封存集上首位来源命中会掉 1 题
- 实测（202 题封存集，同口径）：开启与关闭结果完全一致（证据级@5 172/172、首位来源命中 188/188、目标论文在 Top5 192/192，变化 0 题）；识别到点名的题只有 14/202，而其中多数目标论文本来就排第一，所以封存集上测不出收益
- 目标场景验证：演示问题"BERT 的输入最大长度"开启后 Top5 前三名全是 BERT 自身切片（1.016/1.009/0.978），RoFormer 掉到第 4
- 结论与局限：收益只在"用户点名论文"的场景成立，封存集上只能证明无回归，不能证明提升；覆盖率受别名字典限制（中文泛指"注意力机制"这类匹配不到论文名）。撤销方式是把 `ENTITY_BOOST` 置 0
