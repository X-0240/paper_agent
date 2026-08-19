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
from rerank import get_reranker

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

def split_sentences(text):
    parts=re.split(r"(?<=[.!?])\s+|\n+",text)
    return [p.strip() for p in parts if p.strip()]

def tail_sentences(text,size):
    parts=split_sentences(text)
    out=""
    for p in reversed(parts):
        if out and len(out)+len(p)+1>size:
            break
        out=(p+" "+out).strip()
    return out or text[-size:]

def split_semantic(text):
    chunks=[]; current=""
    for sentence in split_sentences(text):
        if len(current)+len(sentence)+1<=CHUNK_SIZE:
            current=(current+" "+sentence).strip()
            continue
        if current:
            chunks.append(current)
        if len(sentence)>CHUNK_SIZE:
            start=0
            while start<len(sentence):
                chunks.append(sentence[start:start+CHUNK_SIZE])
                start+=CHUNK_SIZE-OVERLAP
            current=""
        else:
            current=tail_sentences(current,OVERLAP)
            current=(current+" "+sentence).strip()
    if current:
        chunks.append(current)
    return chunks

data=json.load(open(DATA_PATH,encoding="utf-8"))
papers=[{**v,"arxiv_id":k} for k,v in data.items()]
chosen=select_papers(papers)

def build_corpus(chunker):
    chunks=[]; sources=[]; sections=[]; paper_sections={}
    for p in chosen:
        for block in p.get("full_text",[]):
            sec=block.get("section_name") or "Unknown"
            text="\n".join(block.get("paragraphs",[]))
            paper_sections.setdefault(p["arxiv_id"],{})[sec]=text
            for part in chunker(text):
                if len(part.strip())<20:
                    continue
                chunks.append(part)
                sources.append(p["arxiv_id"])
                sections.append(sec)
    return {"chunks":chunks,"sources":sources,"sections":sections,"paper_sections":paper_sections}

corpus_fixed=build_corpus(split_fixed)
corpus_sem=build_corpus(split_semantic)
logger.info(f"固定切片{len(corpus_fixed['chunks'])}，语义切片{len(corpus_sem['chunks'])}")

model=SentenceTransformer(MODEL_PATH)
def make_index(corpus):
    emb=model.encode(corpus["chunks"],batch_size=64,normalize_embeddings=True,show_progress_bar=False)
    idx=faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype("float32"))
    bm=BM25Okapi([tokenize(c) for c in corpus["chunks"]])
    return idx,bm

index_f,bm_f=make_index(corpus_fixed)
index_s,bm_s=make_index(corpus_sem)
reranker=get_reranker()

def weighted_candidates(query,index,bm,chunks,candidates=25,alpha=0.5):
    qv=model.encode([query])
    qv=qv/np.linalg.norm(qv)
    vs,vi=index.search(qv.astype("float32"),candidates)
    bm_scores=bm.get_scores(tokenize(query))
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

def rerank_top(query,index,bm,corpus,top_n=5,candidates=25,alpha=0.5):
    cands=weighted_candidates(query,index,bm,corpus["chunks"],candidates,alpha)
    pairs=[[query,corpus["chunks"][j]] for j,_ in cands]
    scores=reranker.predict(pairs,show_progress_bar=False)
    ranked=sorted(zip(cands,scores),key=lambda x:x[1],reverse=True)
    return [j for (j,_),s in ranked[:top_n]]

def evidence_hit(idxs,evidence_texts,corpus):
    for j in idxs:
        cnorm=norm_text(corpus["chunks"][j])
        for evn in evidence_texts:
            for off in range(0,max(1,len(evn)-39),40):
                if evn[off:off+40] in cnorm:
                    return True
    return False

def gather_truth(qa,paper_sections,aid):
    evidence_texts=[]; truth_sections=set()
    sec_map=paper_sections.get(aid,{})
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
    return evidence_texts,truth_sections

stats={
    "fixed_base":[0,0],"fixed_rr":[0,0],"fixed_rr03":[0,0],"fixed_rr07":[0,0],
    "sem_base":[0,0],"sem_rr":[0,0],
}
total=0; t0=time.time()
for p in chosen:
    aid=p["arxiv_id"]
    for qa in p.get("qas",[]):
        q=qa.get("question","")
        if not q:
            continue
        total+=1
        ev,truth=gather_truth(qa,corpus_fixed["paper_sections"],aid)
        if not ev:
            continue
        fb5=[j for j,_ in weighted_candidates(q,index_f,bm_f,corpus_fixed["chunks"],10,0.5)[:5]]
        fb10=[j for j,_ in weighted_candidates(q,index_f,bm_f,corpus_fixed["chunks"],10,0.5)[:10]]
        stats["fixed_base"][0]+=evidence_hit(fb5,ev,corpus_fixed)
        stats["fixed_base"][1]+=evidence_hit(fb10,ev,corpus_fixed)
        for key,alpha in (("fixed_rr",0.5),("fixed_rr03",0.3),("fixed_rr07",0.7)):
            r5=rerank_top(q,index_f,bm_f,corpus_fixed,5,25,alpha)
            r10=rerank_top(q,index_f,bm_f,corpus_fixed,10,25,alpha)
            stats[key][0]+=evidence_hit(r5,ev,corpus_fixed)
            stats[key][1]+=evidence_hit(r10,ev,corpus_fixed)
        sb5=[j for j,_ in weighted_candidates(q,index_s,bm_s,corpus_sem["chunks"],10,0.5)[:5]]
        sb10=[j for j,_ in weighted_candidates(q,index_s,bm_s,corpus_sem["chunks"],10,0.5)[:10]]
        stats["sem_base"][0]+=evidence_hit(sb5,ev,corpus_sem)
        stats["sem_base"][1]+=evidence_hit(sb10,ev,corpus_sem)
        sr5=rerank_top(q,index_s,bm_s,corpus_sem,5,25,0.5)
        sr10=rerank_top(q,index_s,bm_s,corpus_sem,10,25,0.5)
        stats["sem_rr"][0]+=evidence_hit(sr5,ev,corpus_sem)
        stats["sem_rr"][1]+=evidence_hit(sr10,ev,corpus_sem)
elapsed=time.time()-t0

lines=[]
lines.append("## QASPER 切片×Rerank×alpha 对比（证据级 HitRate）")
lines.append(f"- 问题数：{total}")
lines.append("| 配置 | 证据级@5 | 证据级@10 |")
lines.append("|---|---|---|")
names=[
    ("固定切片 baseline",stats["fixed_base"]),
    ("固定切片 Rerank a=0.5",stats["fixed_rr"]),
    ("固定切片 Rerank a=0.3",stats["fixed_rr03"]),
    ("固定切片 Rerank a=0.7",stats["fixed_rr07"]),
    ("语义切片 baseline",stats["sem_base"]),
    ("语义切片 Rerank a=0.5",stats["sem_rr"]),
]
for name,s in names:
    lines.append(f"| {name} | {s[0]/total:.1%} | {s[1]/total:.1%} |")
lines.append(f"- 平均单题耗时：{elapsed/total:.2f}s")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"qasper_semantic_compare.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/qasper_semantic_compare.md")
