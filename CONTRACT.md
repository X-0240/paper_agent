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
    category: Literal["true_conflict","misunderstanding","card_error","insufficient_evidence"]
    description: str
    verdict: str
    confidence: Literal["high","low"]
    evidence_supplementary: str

class ReviewReport(TypedDict):
    title: str
    consensus: List[str]
    disagreements: List[Conflict]
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
```

强制约束：
- `FactItem.source_chunk_id` 为空则丢弃，不写 State
- `Conflict` 只能由 `verify_claim` 产出，Agent 不能自行构造
- `trace_log` 只追加不覆盖

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
- 超时：60s

### 5. verify_claim

- 输入：`fact_a: FactItem`, `fact_b: FactItem`
- 输出：`Conflict`
- 实现：复用 `agent3_review.verify_with_retrieval` 的二次检索与裁决思路；先查本地，外网 P1
- 约束：category 只能是四种枚举之一
- 超时：60s

### 6. write_review

- 输入：`query: str`, `facts: List[FactItem]`, `conflicts: List[Conflict]`
- 输出：`ReviewReport`
- 实现：复用 `agent3_review.generate_review` 的综述生成思路，输出字段对齐新结构
- 强制：引用带 chunk_id
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
MAX_AGENT_STEP=10
MAX_FACTS_PER_SESSION=20
MAX_CONFLICT_PER_SESSION=5
MAX_EXTERNAL_SEARCH_NUM=3
DAILY_COST_BUDGET=5.0      # 元，可被.env覆盖
SESSION_COST_BUDGET=1.0    # 元
```

## 八、已确认决策

1. `FactItem.source_chunk_id` 强制非空：接受
2. 旧 agent2/agent3 不立即删除：先标记 deprecated，新链路测试通过后再删
3. 预算不足时生成不完整综述：接受
4. `MAX_AGENT_STEP=10`：接受，成本护栏兜底
5. 冲突评测：自标 20-40 对，不用公开数据集，避免编造
