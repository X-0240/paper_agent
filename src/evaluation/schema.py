import json
from dataclasses import dataclass,asdict,field
from typing import List,Optional

@dataclass
class Question:
    question_id: str
    question: str
    paper_id: str
    evidence_sentences: List[str]
    section: Optional[str]=None
    question_type: str="unknown"
    split: str="dev"
    source: str="forward"
    query_en: Optional[str]=None
    metadata: dict=field(default_factory=dict)

def load_questions(path):
    #统一题目格式；旧数据缺 evidence 或 paper_id 直接拒绝，避免污染评测
    raw=json.load(open(path,encoding="utf-8"))
    questions=[]
    for i,item in enumerate(raw):
        q=Question(
            question_id=str(item.get("question_id",item.get("id",i+1))),
            question=item.get("question","").strip(),
            paper_id=item.get("paper_id",item.get("source","")).strip(),
            evidence_sentences=[s for s in (item.get("evidence_sentences") or []) if s and s.strip()],
            section=item.get("section") or item.get("mapped_section"),
            question_type=item.get("question_type","unknown"),
            split=item.get("split","dev"),
            source=item.get("source","forward"),
            query_en=item.get("query_en"),
            metadata=item.get("metadata",{})
        )
        if not q.question or not q.paper_id or not q.evidence_sentences:
            continue
        questions.append(q)
    return questions

def save_questions(questions,path):
    data=[asdict(q) for q in questions]
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
