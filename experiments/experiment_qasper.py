import os
#国内网络直连HuggingFace会超时，必须在import sentence_transformers前设置镜像
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
from rerank import rerank

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

DATA_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"datasets","qasper-test-v0.3.json")
MODEL_PATH=os.getenv("MODEL_PATH")
CHUNK_SIZE=500
OVERLAP=100
TOP_N_PAPERS=40

TOPICS=[
    "translation","parsing","generation","summarization","question answering","embedding",
    "attention","dialogue","sentiment","relation","knowledge","multimodal","reinforcement",
    "semantic","coreference","machine reading","information extraction","zero-shot",
]

def split_chunk(text):
    chunks=[]
    start=0
    while start<len(text):
        chunks.append(text[start:start+CHUNK_SIZE])
        start+=CHUNK_SIZE-OVERLAP
    return chunks

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

def evidence_hit(idxs,evidence_texts):
    for j in idxs:
        cnorm=norm_text(all_chunks[j])
        for evn in evidence_texts:
            for off in range(0,max(1,len(evn)-39),40):
                if evn[off:off+40] in cnorm:
                    return True
    return False

data=json.load(open(DATA_PATH,encoding="utf-8"))
papers=[{**v,"arxiv_id":k} for k,v in data.items()]
chosen=select_papers(papers)

all_chunks=[]; all_sources=[]; all_sections=[]; paper_sections={}
for p in chosen:
    for block in p.get("full_text",[]):
        sec=block.get("section_name") or "Unknown"
        text="\n".join(block.get("paragraphs",[]))
        paper_sections.setdefault(p["arxiv_id"],{})[sec]=text
        for part in split_chunk(text):
            if len(part.strip())<20:
                continue
            all_chunks.append(part)
            all_sources.append(p["arxiv_id"])
            all_sections.append(sec)

model=SentenceTransformer(MODEL_PATH)
embeddings=model.encode(all_chunks,batch_size=64,normalize_embeddings=True)
index=faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in all_chunks])

def weighted_top(query,k=5,alpha=0.5,candidates=10):
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
    top=sorted(combined.items(),key=lambda x:x[1],reverse=True)[:k]
    return [j for j,_ in top]

def rrf_top(query,k=60,top_n=5,candidates=25):
    qv=model.encode([query])
    qv=qv/np.linalg.norm(qv)
    _,vi=index.search(qv.astype("float32"),candidates)
    bm_scores=bm25.get_scores(tokenize(query))
    bm_idx=sorted(range(len(bm_scores)),key=lambda i:bm_scores[i],reverse=True)[:candidates]
    scores={}
    for idxs in (vi[0],bm_idx):
        for rank,j in enumerate(idxs):
            scores[j]=scores.get(j,0)+1/(k+rank+1)
    top=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_n]
    return [j for j,_ in top]

def weighted_candidates(query,candidates=25):
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
        combined[j]=combined.get(j,0)+0.5*float(s)
    bm_scores=np.array([bm_scores[j] for j in bm_idx])
    if max(bm_scores)>0:
        bm_scores=bm_scores/max(bm_scores)
    for s,j in zip(bm_scores,bm_idx):
        combined[j]=combined.get(j,0)+0.5*float(s)
    return sorted(combined.items(),key=lambda x:x[1],reverse=True)[:candidates]

#策略计数器：证据级@5/@10
stats={
    "baseline":[0,0],
    "rrf":[0,0],
    "rerank":[0,0],
    "rerank50":[0,0],
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
        top5_base=weighted_top(q,k=5)
        top10_base=weighted_top(q,k=10)
        stats["baseline"][0]+=evidence_hit(top5_base,evidence_texts)
        stats["baseline"][1]+=evidence_hit(top10_base,evidence_texts)
        top5_rrf=rrf_top(q,top_n=5)
        top10_rrf=rrf_top(q,top_n=10)
        stats["rrf"][0]+=evidence_hit(top5_rrf,evidence_texts)
        stats["rrf"][1]+=evidence_hit(top10_rrf,evidence_texts)
        cands=weighted_candidates(q,candidates=25)
        items=[{"idx":j,"text":all_chunks[j]} for j,_ in cands]
        rr5=rerank(q,items,top_n=5)
        rr10=rerank(q,items,top_n=10)
        stats["rerank"][0]+=evidence_hit([it["idx"] for it in rr5],evidence_texts)
        stats["rerank"][1]+=evidence_hit([it["idx"] for it in rr10],evidence_texts)
        cands50=weighted_candidates(q,candidates=50)
        items50=[{"idx":j,"text":all_chunks[j]} for j,_ in cands50]
        rr50_5=rerank(q,items50,top_n=5)
        rr50_10=rerank(q,items50,top_n=10)
        stats["rerank50"][0]+=evidence_hit([it["idx"] for it in rr50_5],evidence_texts)
        stats["rerank50"][1]+=evidence_hit([it["idx"] for it in rr50_10],evidence_texts)
elapsed=time.time()-t0

lines=[]
lines.append("## QASPER 检索策略对比（证据级 HitRate，问题数含证据=影响分母需注意）")
lines.append(f"- 论文数：{len(chosen)}，切片数：{len(all_chunks)}，问题数：{total}")
lines.append("| 策略 | 证据级@5 | 证据级@10 |")
lines.append("|---|---|---|")
for name in ("baseline","rrf","rerank","rerank50"):
    lines.append(f"| {name} | {stats[name][0]/total:.1%} | {stats[name][1]/total:.1%} |")
lines.append(f"- 平均单题耗时（含三次检索）：{elapsed/total:.2f}s")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"qasper_strategy_compare.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/qasper_strategy_compare.md")
