import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import json
import logging
import re
import sys
import time
import numpy as np
from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from rerank import get_reranker

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

def norm_text(text):
    return re.sub(r"\s+"," ",text).lower()

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

chunks=[]; sources=[]; sections=[]; paper_sections={}
for p in chosen:
    for block in p.get("full_text",[]):
        sec=block.get("section_name") or "Unknown"
        text="\n".join(block.get("paragraphs",[]))
        paper_sections.setdefault(p["arxiv_id"],{})[sec]=text
        for part in split_fixed(text):
            if len(part.strip())<20:
                continue
            chunks.append(part)
            sources.append(p["arxiv_id"])
            sections.append(sec)

model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])
reranker=get_reranker()

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

def rerank_top(query,top_n=5,candidates=25,alpha=0.5):
    cands=weighted_candidates(query,candidates,alpha)
    pairs=[[query,chunks[j]] for j,_ in cands]
    scores=reranker.predict(pairs,show_progress_bar=False)
    ranked=sorted(zip(cands,scores),key=lambda x:x[1],reverse=True)
    return [j for (j,_),s in ranked[:top_n]]

def neighbors(idxs,width=1):
    #同一论文同一章节内，命中片段的相邻切片
    out=set()
    for j in idxs:
        out.add(j)
        for delta in range(1,width+1):
            for nb in (j-delta,j+delta):
                if 0<=nb<len(chunks) and sources[nb]==sources[j] and sections[nb]==sections[j]:
                    out.add(nb)
    return out

def evidence_hit(idxs,evidence_texts):
    for j in idxs:
        cnorm=norm_text(chunks[j])
        for evn in evidence_texts:
            for off in range(0,max(1,len(evn)-39),40):
                if evn[off:off+40] in cnorm:
                    return True
    return False

stats={
    "rerank":[0,0],
    "neighbor1":[0,0],
    "neighbor2":[0,0],
}
total=0; t0=time.time()
for p in chosen:
    aid=p["arxiv_id"]
    sec_map=paper_sections.get(aid,{})
    for qa in p.get("qas",[]):
        q=qa.get("question","")
        if not q:
            continue
        total+=1
        evidence_texts=[]; truth_sections=set()
        for ans in qa.get("answers",[]):
            answer_obj=ans.get("answer") or {}
            for ev in answer_obj.get("evidence") or []:
                evn=norm_text(ev)
                if len(evn)<20:
                    continue
                evidence_texts.append(evn)
                probe=evn[:80]
                for sec,text in sec_map.items():
                    if probe in norm_text(text):
                        truth_sections.add(sec)
                        break
        if not evidence_texts:
            continue
        top5=rerank_top(q,5)
        top10=rerank_top(q,10)
        stats["rerank"][0]+=evidence_hit(top5,evidence_texts)
        stats["rerank"][1]+=evidence_hit(top10,evidence_texts)
        stats["neighbor1"][0]+=evidence_hit(neighbors(top5,1),evidence_texts)
        stats["neighbor1"][1]+=evidence_hit(neighbors(top10,1),evidence_texts)
        stats["neighbor2"][0]+=evidence_hit(neighbors(top5,2),evidence_texts)
        stats["neighbor2"][1]+=evidence_hit(neighbors(top10,2),evidence_texts)
elapsed=time.time()-t0

lines=[]
lines.append("## QASPER 邻接片段扩展对比（证据级 HitRate）")
lines.append(f"- 问题数：{total}")
lines.append("| 配置 | 证据级@5 | 证据级@10 |")
lines.append("|---|---|---|")
for name,s in (
    ("Rerank 候选25",stats["rerank"]),
    ("Rerank + 相邻±1",stats["neighbor1"]),
    ("Rerank + 相邻±2",stats["neighbor2"]),
):
    lines.append(f"| {name} | {s[0]/total:.1%} | {s[1]/total:.1%} |")
lines.append(f"- 平均单题耗时：{elapsed/total:.2f}s")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"qasper_neighbor_compare.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/qasper_neighbor_compare.md")
