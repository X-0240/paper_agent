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

def gold_rank(items,evidence_sentences):
    #金标句首次出现的名次；0 表示Top-k内未出现。多证据题按"任一命中"取最早名次
    targets=[norm(x) for x in evidence_sentences if norm(x)]
    if not targets:
        return 0
    for rank,item in enumerate(items,1):
        text=norm(item.get("text",""))
        if any(t in text for t in targets):
            return rank
    return 0


def mrr(ranks):
    #MRR：命中题取名次倒数，未命中记0，再对全题取平均
    if not ranks:
        return 0.0
    return sum((1.0/r) if r>0 else 0.0 for r in ranks)/len(ranks)


def evidence_coverage(items,evidence_sentences):
    #多证据覆盖率：本题的金标句有多少比例出现在Top-k里（单证据题退化为0或1）
    targets=[norm(x) for x in evidence_sentences if norm(x)]
    if not targets:
        return 0.0
    texts=[norm(item.get("text","")) for item in items]
    covered=sum(1 for t in targets if any(t in body for body in texts))
    return covered/len(targets)


def ndcg_at_k(items,evidence_sentences,k=None):
    #简化NDCG@k：相关单位置按名次折损求和后除以理想位置的折损和
    import math
    targets=[norm(x) for x in evidence_sentences if norm(x)]
    if not targets:
        return 0.0
    scope=items[:k] if k else items
    gains=[]
    for item in scope:
        text=norm(item.get("text",""))
        hits=sum(1 for t in targets if t in text)
        gains.append(min(hits,len(targets)))
    dcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gains))
    ideal=sorted(gains,reverse=True)
    idcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0.0


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
