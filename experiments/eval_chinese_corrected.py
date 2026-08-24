import json
import logging
import os
import re
import sys
import time
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from rerank import rerank

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

FAISS_PATH=os.getenv("FAISS_PATH")
MODEL_PATH=os.getenv("MODEL_PATH")
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
CASES_PATH=os.path.join(BASE_DIR,"corrected_chinese_cases.json")
CACHE_PATH=os.path.join(BASE_DIR,"eval_query_transform_cache.json")

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

def norm(s):
    s=re.sub(r"^[\d.]+\s*","",s.lower())
    return re.sub(r"[^a-z ]","",s).strip()

def section_match(section,expected):
    e=norm(expected); a=norm(section)
    if not e or not a:
        return False
    return e in a or a in e

def chapter_hit(idxs,truth_source,truth_section):
    for j in idxs:
        if sources[j]!=truth_source:
            continue
        if section_match(sections[j],truth_section):
            return True
    return False

def weighted_candidates(query,candidates=25,alpha=0.5):
    qv=model.encode([query]); qv=qv/np.linalg.norm(qv)
    vs,vi=index.search(qv.astype("float32"),candidates)
    bm=bm25.get_scores(tokenize(query))
    bm_idx=sorted(range(len(bm)),key=lambda i:bm[i],reverse=True)[:candidates]
    comb={}; vs=vs[0]; vi=vi[0]
    if max(vs)>0: vs=vs/max(vs)
    for s,j in zip(vs,vi): comb[j]=comb.get(j,0)+alpha*float(s)
    bm=np.array([bm[j] for j in bm_idx])
    if max(bm)>0: bm=bm/max(bm)
    for s,j in zip(bm,bm_idx): comb[j]=comb.get(j,0)+(1-alpha)*float(s)
    return sorted(comb.items(),key=lambda x:x[1],reverse=True)[:candidates]

def top_k(query,k):
    # 候选池取25，保证top20也有足够余量
    return [j for j,_ in weighted_candidates(query,25,0.5)[:k]]

meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
chunks=meta["documents"]; sources=meta["sources"]; sections=meta["sections"]
logging.info(f"索引切片：{len(chunks)}")
model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=True)
index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])

cases=[c for c in json.load(open(CASES_PATH,encoding="utf-8")) if c["mapped_section"]]
logging.info(f"修正后中文题：{len(cases)}")
cache=json.load(open(CACHE_PATH,encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}
variants={"原文":[c["question"] for c in cases],
          "翻译成英文":[cache.get(f"translate|{c['question']}",c["question"]) for c in cases]}

t0=time.time()
for name,qs in variants.items():
    for k in [5,10,15,20]:
        hits=0
        paper_hits=0
        paper_hit_chapter_miss=0
        for c,q in zip(cases,qs):
            idxs=top_k(q,k)
            chapter=chapter_hit(idxs,c["source"],c["mapped_section"])
            paper=any(sources[j]==c["source"] for j in idxs)
            hits+=chapter
            paper_hits+=paper
            paper_hit_chapter_miss+=paper and not chapter
        print(f"{name}（修正章节）：章节级@{k}={hits/len(cases):.1%}（{hits}/{len(cases)}），论文级@{k}={paper_hits/len(cases):.1%}（{paper_hits}/{len(cases)}），论文中章节未中={paper_hit_chapter_miss}")

# Rerank对照：翻译变体，Top-25粗召回后用Cross-Encoder精排
for k in [5,10]:
    hits=0
    paper_hits=0
    for c,q in zip(cases,variants["翻译成英文"]):
        top=weighted_candidates(q,25,0.5)
        items=[{"source":sources[j],"section":sections[j],"text":chunks[j][:300],"idx":j} for j,_ in top]
        ranked=rerank(q,items,top_n=k)
        idxs=[it["idx"] for it in ranked]
        hits+=chapter_hit(idxs,c["source"],c["mapped_section"])
        paper_hits+=any(sources[j]==c["source"] for j in idxs)
    print(f"Rerank翻译（修正章节）：章节级@{k}={hits/len(cases):.1%}（{hits}/{len(cases)}），论文级@{k}={paper_hits/len(cases):.1%}（{paper_hits}/{len(cases)}）")

# 诊断：翻译变体下正确章节chunk的最低排名分布
diag={"top20":0,"beyond20":0,"not_in_candidates":0,"section_missing":0}
samples=[]
missing_samples=[]
for c,q in zip(cases,variants["翻译成英文"]):
    correct=[i for i in range(len(chunks)) if sources[i]==c["source"] and section_match(sections[i],c["mapped_section"])]
    if not correct:
        diag["section_missing"]+=1
        if len(missing_samples)<15:
            idx_sections=sorted(set(sections[i] for i in range(len(chunks)) if sources[i]==c["source"]))
            similar=[x for x in idx_sections if section_match(x,c["mapped_section"])]
            missing_samples.append({"question":c["question"][:50],"source":c["source"],"mapped_section":c["mapped_section"],"index_similar":similar[:3]})
        continue
    full=weighted_candidates(q,100,0.5)
    ranks=[r+1 for r,(j,_) in enumerate(full) if j in correct]
    min_rank=min(ranks) if ranks else None
    if min_rank is None:
        diag["not_in_candidates"]+=1
    elif min_rank<=20:
        diag["top20"]+=1
    else:
        diag["beyond20"]+=1
    if len(samples)<10 and (min_rank is None or min_rank>20):
        samples.append({"question":c["question"][:60],"source":c["source"],"mapped_section":c["mapped_section"],"min_rank":min_rank})
print(f"章节排名诊断（翻译变体）：{diag}")
for s in samples:
    print(s)
print(f"章节缺失样例（{len(missing_samples)}）")
for s in missing_samples:
    print(s)
print(f"评测耗时：{time.time()-t0:.1f}s")
