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

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

FAISS_PATH=os.getenv("FAISS_PATH")
MODEL_PATH=os.getenv("MODEL_PATH")
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
CASES_PATH=os.path.join(BASE_DIR,"corrected_chinese_cases.json")
CACHE_PATH=os.path.join(BASE_DIR,"eval_query_transform_cache.json")
EVIDENCE_PATH=os.path.join(BASE_DIR,"evidence_batch1_merged.json")

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

def norm(s):
    s=re.sub(r"^[\d.]+\s*","",s.lower())
    return re.sub(r"[^a-z ]","",s).strip()

def ev_norm(s):
    #证据匹配：只保留字母数字，容忍换行/连字符/数学符号差异
    return re.sub(r"[^a-z0-9]+","",s.lower())

def section_match(section,expected):
    e=norm(expected); a=norm(section)
    if not e or not a:
        return False
    return e in a or a in e

def chapter_hit(idxs,truth_source,truth_sections):
    for j in idxs:
        if sources[j]!=truth_source:
            continue
        for sec in truth_sections:
            if section_match(sections[j],sec):
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
    return [j for j,_ in weighted_candidates(query,25,0.5)[:k]]

meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
chunks=meta["documents"]; sources=meta["sources"]; sections=meta["sections"]
logging.info(f"索引切片：{len(chunks)}")
model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=True)
index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])

all_cases=[c for c in json.load(open(CASES_PATH,encoding="utf-8")) if c["mapped_section"] and c.get("valid_sections")]
cases=all_cases[:32]
evs=json.load(open(EVIDENCE_PATH,encoding="utf-8"))
ev_map={e["id"]:e for e in evs}
cache=json.load(open(CACHE_PATH,encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}

#把证据句映射到索引chunk
not_mapped=0
evidence_chunks={}
for c in cases:
    e=ev_map.get(cases.index(c)+1)
    if not e or not e.get("evidence_sentences"):
        not_mapped+=1
        evidence_chunks[cases.index(c)]=[]
        continue
    ids=set()
    for s in e["evidence_sentences"]:
        ns=ev_norm(s)
        for i in range(len(chunks)):
            if sources[i]==c["source"] and ns and ns in ev_norm(chunks[i]):
                ids.add(i)
    if not ids:
        not_mapped+=1
    evidence_chunks[cases.index(c)]=ids
print(f"证据映射：32题中未映射证据到chunk={not_mapped}")

t0=time.time()
for k in [5,10,20]:
    ev_hits=0
    ch_hits=0
    for c in cases:
        idx=cases.index(c)
        q=cache.get(f"translate|{c['question']}",c["question"])
        idxs=top_k(q,k)
        ev_hits+=bool(evidence_chunks[idx] and (set(idxs)&evidence_chunks[idx]))
        ch_hits+=chapter_hit(idxs,c["source"],c["valid_sections"])
    print(f"批1翻译（32题）：证据级@{k}={ev_hits/32:.1%}（{ev_hits}/32），章节级@{k}={ch_hits/32:.1%}（{ch_hits}/32）")
print(f"评测耗时：{time.time()-t0:.1f}s")
