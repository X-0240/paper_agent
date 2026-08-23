import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import json
import logging
import re
import sys
import numpy as np
from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

_DATA=os.path.join(os.path.dirname(os.path.abspath(__file__)),"datasets","qasper-test-v0.3.json")
DATA_PATH=_DATA if os.path.exists(_DATA) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"datasets","qasper-test-v0.3.json")
MODEL_PATH=os.getenv("MODEL_PATH")
CHUNK_SIZE=500
OVERLAP=100
TOP_N_PAPERS=40

TOPICS=[
    "translation","parsing","generation","summarization","question answering","embedding",
    "attention","dialogue","sentiment","relation","knowledge","multimodal","reinforcement",
    "semantic","coreference","machine reading","information extraction","zero-shot",
]

def select_papers(papers):
    cands=[p for p in papers if p.get("full_text") and p.get("qas")]
    chosen=[]; chosen_ids=set()
    for topic in TOPICS:
        best=None
        for p in cands:
            if p["arxiv_id"] in chosen_ids:
                continue
            head=" ".join(b.get("paragraphs",[""])[0] for b in p.get("full_text",[])[:2])
            text=(p.get("title","")+" "+head).lower()
            if topic in text and (best is None or len(p.get("qas",[]))>len(best.get("qas",[]))):
                best=p
        if best:
            chosen.append(best)
            chosen_ids.add(best["arxiv_id"])
        if len(chosen)>=TOP_N_PAPERS:
            break
    rest=sorted(
        [p for p in cands if p["arxiv_id"] not in chosen_ids],
        key=lambda p:len(p.get("qas",[])),
        reverse=True
    )
    for p in rest:
        if len(chosen)>=TOP_N_PAPERS:
            break
        chosen.append(p)
    return chosen

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

def split_fixed(text):
    chunks=[]
    start=0
    while start<len(text):
        chunks.append(text[start:start+CHUNK_SIZE])
        start+=CHUNK_SIZE-OVERLAP
    return chunks

data=json.load(open(DATA_PATH,encoding="utf-8"))
papers=[{**v,"arxiv_id":k} for k,v in data.items()]
chosen=select_papers(papers)

chunks=[]; sources=[]
for p in chosen:
    for block in p.get("full_text",[]):
        text="\n".join(block.get("paragraphs",[]))
        for part in split_fixed(text):
            if len(part.strip())<20:
                continue
            chunks.append(part)
            sources.append(p["arxiv_id"])

model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])

def top_sources(mode,query,k=5,alpha=0.5):
    if mode=="vector":
        qv=model.encode([query])
        qv=qv/np.linalg.norm(qv)
        _,vi=index.search(qv.astype("float32"),k)
        return [sources[j] for j in vi[0]]
    if mode=="bm25":
        scores=bm25.get_scores(tokenize(query))
        top=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
        return [sources[j] for j in top]
    #hybrid
    qv=model.encode([query])
    qv=qv/np.linalg.norm(qv)
    vs,vi=index.search(qv.astype("float32"),10)
    bm_scores=bm25.get_scores(tokenize(query))
    bm_idx=sorted(range(len(bm_scores)),key=lambda i:bm_scores[i],reverse=True)[:10]
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
    top=sorted(combined.items(),key=lambda x:x[1],reverse=True)[:k]
    return [sources[j] for j,_ in top]

stats={"vector":0,"bm25":0,"hybrid":0}
total=0
for p in chosen:
    aid=p["arxiv_id"]
    for qa in p.get("qas",[]):
        q=qa.get("question","")
        if not q:
            continue
        total+=1
        stats["vector"]+=aid in top_sources("vector",q)
        stats["bm25"]+=aid in top_sources("bm25",q)
        stats["hybrid"]+=aid in top_sources("hybrid",q)

lines=[]
lines.append("## QASPER 论文级召回拆解（40篇，263题，@5）")
lines.append(f"- 纯向量：{stats['vector']/total:.1%}")
lines.append(f"- 纯BM25：{stats['bm25']/total:.1%}")
lines.append(f"- 混合：{stats['hybrid']/total:.1%}")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"qasper_retrieval_diagnose.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/qasper_retrieval_diagnose.md")
