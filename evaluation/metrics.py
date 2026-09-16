import math
import re
import unicodedata

def norm(s):
    #NFKC统一数学斜体/连字，再只保留字母数字，兼容论文PDF符号差异
    s=unicodedata.normalize("NFKC",s or "")
    return re.sub(r"[^a-z0-9]+","",s.lower())

def section_norm(s):
    s=unicodedata.normalize("NFKC",s or "")
    s=re.sub(r"^[\d.]+\s*","",s.lower())
    return re.sub(r"[^a-z ]","",s).strip()

def evidence_hit(retrieved,evidence_sentences):
    #任一证据句出现在任一命中片段内即算命中
    targets=[norm(x) for x in evidence_sentences if norm(x)]
    if not targets:
        return False
    for item in retrieved:
        text=norm(item.get("text",""))
        if any(t and t in text for t in targets):
            return True
    return False

def paper_hit(retrieved,paper_id):
    return any(item.get("source")==paper_id for item in retrieved)

def section_hit(retrieved,paper_id,section):
    if not section:
        return False
    target=section_norm(section)
    for item in retrieved:
        if item.get("source")!=paper_id:
            continue
        got=section_norm(item.get("section",""))
        if target and got and (target in got or got in target):
            return True
    return False

def wilson_ci(hits,n,z=1.96):
    #Wilson区间：小样本比例指标比正态近似更稳
    if n<=0:
        return 0.0,0.0
    p=hits/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return max(0.0,center-half),min(1.0,center+half)

def evaluate_questions(questions,retriever,k=5,query_field="question"):
    rows=[]
    for q in questions:
        query=getattr(q,query_field) if query_field!="question" else q.question
        query=query or q.question
        retrieved=retriever.search(query,k=k)
        ev=evidence_hit(retrieved,q.evidence_sentences)
        rows.append({
            "question_id":q.question_id,
            "question":q.question,
            "paper_id":q.paper_id,
            "question_type":q.question_type,
            "split":q.split,
            "source":q.source,
            "evidence_hit":ev,
            "paper_hit":paper_hit(retrieved,q.paper_id),
            "section_hit":section_hit(retrieved,q.paper_id,q.section),
            "top_sources":[{"source":r.get("source"),"section":r.get("section")} for r in retrieved]
        })
    n=len(rows)
    hits=sum(1 for r in rows if r["evidence_hit"])
    lo,hi=wilson_ci(hits,n)
    return {"n":n,"hits":hits,"rate":hits/n if n else 0.0,"ci95":[lo,hi],"rows":rows}
