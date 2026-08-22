from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class SectionRef:
    name: str
    page: Optional[int]=None
    chunk_ids: List[str]=field(default_factory=list)

@dataclass
class PaperMeta:
    paper_id: str
    title: str
    authors: List[str]=field(default_factory=list)
    year: Optional[int]=None
    source_type: str="local"
    pdf_path: Optional[str]=None

@dataclass
class PaperCard:
    paper_id: str
    title: str
    abstract: str=""
    key_findings: List[str]=field(default_factory=list)
    methodology: str=""
    limitations: List[str]=field(default_factory=list)
    sections_ref: List[SectionRef]=field(default_factory=list)

@dataclass
class FactItem:
    fact_id: str
    paper_id: str
    entity: str
    attribute: str
    value: str
    content: str
    source_chunk_id: str
    section_name: str=""
    raw_quote: str=""
    confidence: str="low"

@dataclass
class Conflict:
    conflict_id: str
    fact_ids: List[str]=field(default_factory=list)
    category: str="insufficient_evidence"
    description: str=""
    verdict: str=""
    confidence: str="low"
    evidence_supplementary: str=""

@dataclass
class ReviewReport:
    title: str=""
    consensus: List[str]=field(default_factory=list)
    disagreements: List[dict]=field(default_factory=list)
    open_questions: List[str]=field(default_factory=list)
    references: List[str]=field(default_factory=list)
    conflict_mark_list: List[str]=field(default_factory=list)

@dataclass
class AgentState:
    query: str=""
    complexity: str="SIMPLE"
    papers: List[PaperMeta]=field(default_factory=list)
    candidate_chunks: List[dict]=field(default_factory=list)
    cards: List[PaperCard]=field(default_factory=list)
    facts: List[FactItem]=field(default_factory=list)
    conflicts: List[Conflict]=field(default_factory=list)
    review: Optional[ReviewReport]=None
    simple_answer: Optional[str]=None
    step_count: int=0
    cost_consumed: float=0.0
    trace_log: List[dict]=field(default_factory=list)
    section_cache: dict=field(default_factory=dict)
