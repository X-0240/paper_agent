import json
import logging
import os
import re
import sys
import time
import numpy as np
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

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

def norm(s):
    s=re.sub(r"^[\d.]+\s*","",s.lower())
    s=re.sub(r"^(figure|table|abstract|appendix|proposition|theorem|section)\s*[\d.]*\s*","",s)
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

def top5(query):
    return [j for j,_ in weighted_candidates(query,10,0.5)[:5]]

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
          "翻译成英文":[cache.get(c["question"],c["question"]) for c in cases]}

t0=time.time()
for name,qs in variants.items():
    hits=0
    for c,q in zip(cases,qs):
        hits+=chapter_hit(top5(q),c["source"],c["mapped_section"])
    print(f"{name}（修正章节）：章节级@5={hits/len(cases):.1%}（{hits}/{len(cases)}）")
print(f"评测耗时：{time.time()-t0:.1f}s")
