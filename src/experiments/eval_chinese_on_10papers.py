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
logger=logging.getLogger(__name__)

FAISS_PATH=os.getenv("FAISS_PATH")
MODEL_PATH=os.getenv("MODEL_PATH")
PAPERS_DIR=os.getenv("PAPERS_DIR")
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
CASES_PATH=os.path.join(BASE_DIR,"corrected_chinese_cases.json")
CACHE_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_query_transform_cache.json")

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

def section_match(section,expected_section):
    def norm(s):
        s=re.sub(r"^[\d.]+\s*","",s.lower())
        return re.sub(r"[^a-z ]","",s).strip()
    e=norm(expected_section)
    a=norm(section)
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
    qv=model.encode([query])
    qv=qv/np.linalg.norm(qv)
    vs,vi=index.search(qv.astype("float32"),candidates)
    bm_scores=bm25.get_scores(tokenize(query))
    bm_idx=sorted(range(len(bm_scores)),key=lambda i:bm_scores[i],reverse=True)[:candidates]
    combined={}
    vs=vs[0]; vi=vi[0]
    if max(vs)>0:
        vs=vs/max(vs)
    for s,j in zip(vs,vi):
        combined[j]=combined.get(j,0)+alpha*float(s)
    bm_scores=np.array([bm_scores[j] for j in bm_idx])
    if max(bm_scores)>0:
        bm_scores=bm_scores/max(bm_scores)
    for s,j in zip(bm_scores,bm_idx):
        combined[j]=combined.get(j,0)+(1-alpha)*float(s)
    return sorted(combined.items(),key=lambda x:x[1],reverse=True)[:candidates]

def top_k(query,k):
    # 候选池取25，与50篇评测口径一致
    return [j for j,_ in weighted_candidates(query,25,0.5)[:k]]

#10篇主论文来源
main_sources=set(sorted(f[:-4] for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")))
meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
all_chunks=meta["documents"]; all_sources=meta["sources"]; all_sections=meta["sections"]
keep=[i for i,s in enumerate(all_sources) if s in main_sources]
chunks=[all_chunks[i] for i in keep]
sources=[all_sources[i] for i in keep]
sections=[all_sections[i] for i in keep]
logger.info(f"10篇专属切片：{len(chunks)}（总{len(all_chunks)}）")

model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=True)
index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])

cases=[c for c in json.load(open(CASES_PATH,encoding="utf-8")) if c["mapped_section"]]
logger.info(f"修正后中文题：{len(cases)}")

cache=json.load(open(CACHE_PATH,encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}
variants={"原文":[c["question"] for c in cases],
          "翻译成英文":[cache.get(c["question"],c["question"]) for c in cases]}

t0=time.time()
for name,qs in variants.items():
    for k in [5,10,20]:
        hits=0
        paper_hits=0
        for c,q in zip(cases,qs):
            idxs=top_k(q,k)
            hits+=chapter_hit(idxs,c["source"],[c["mapped_section"]])
            paper_hits+=any(sources[j]==c["source"] for j in idxs)
        print(f"{name}（10篇专属索引）：章节级@{k}={hits/len(cases):.1%}（{hits}/{len(cases)}），论文级@{k}={paper_hits/len(cases):.1%}（{paper_hits}/{len(cases)}）")
print(f"评测耗时：{time.time()-t0:.1f}s")
