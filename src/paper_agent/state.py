from dataclasses import dataclass, field


@dataclass
class SectionRef:
    name: str
    page: int | None = None
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class PaperMeta:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    source_type: str = "local"
    pdf_path: str | None = None


@dataclass
class PaperCard:
    paper_id: str
    title: str
    abstract: str = ""
    key_findings: list[str] = field(default_factory=list)
    methodology: str = ""
    limitations: list[str] = field(default_factory=list)
    sections_ref: list[SectionRef] = field(default_factory=list)


@dataclass
class FactItem:
    fact_id: str
    paper_id: str
    entity: str
    attribute: str
    value: str
    content: str
    source_chunk_id: str
    section_name: str = ""
    raw_quote: str = ""
    confidence: str = "low"
    year: int | None = None


@dataclass
class Conflict:
    conflict_id: str
    fact_ids: list[str] = field(default_factory=list)
    category: str = "insufficient_evidence"
    description: str = ""
    verdict: str = ""
    confidence: str = "low"
    evidence_supplementary: str = ""


@dataclass
class ReviewReport:
    title: str = ""
    consensus: list[str] = field(default_factory=list)
    disagreements: list[dict] = field(default_factory=list)
    superseded_conclusions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    conflict_mark_list: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    query: str = ""
    complexity: str = "SIMPLE"
    papers: list[PaperMeta] = field(default_factory=list)
    candidate_chunks: list[dict] = field(default_factory=list)
    cards: list[PaperCard] = field(default_factory=list)
    facts: list[FactItem] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    review: ReviewReport | None = None
    simple_answer: str | None = None
    step_count: int = 0
    cost_consumed: float = 0.0
    trace_log: list[dict] = field(default_factory=list)
    section_cache: dict = field(default_factory=dict)
    card_cache: dict = field(default_factory=dict)
    pending_conflicts: list[dict] = field(default_factory=list)
    search_count: int = 0
