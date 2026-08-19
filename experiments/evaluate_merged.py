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
from rerank import rerank

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

DATA_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"datasets","qasper-test-v0.3.json")
SECTIONS_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"papers_sections")
MODEL_PATH=os.getenv("MODEL_PATH")
CHUNK_SIZE=500
OVERLAP=100
TOP_N_PAPERS=40

TOPICS=[
    "translation","parsing","generation","summarization","question answering","embedding",
    "attention","dialogue","sentiment","relation","knowledge","multimodal","reinforcement",
    "semantic","coreference","machine reading","information extraction","zero-shot",
]

def split_fixed(text):
    chunks=[]
    start=0
    while start<len(text):
        chunks.append(text[start:start+CHUNK_SIZE])
        start+=CHUNK_SIZE-OVERLAP
    return chunks

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

def norm_text(text):
    return re.sub(r"\s+"," ",text).lower()

def select_qasper(papers):
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

def section_match(section,expected_section):
    def norm(s):
        s=re.sub(r"^[\d.]+\s*","",s.lower())
        s=re.sub(r"^(figure|table|abstract|appendix|proposition|theorem|section)\s*[\d.]*\s*","",s)
        return re.sub(r"[^a-z ]","",s).strip()
    e=norm(expected_section)
    a=norm(section)
    if not e or not a:
        return False
    return e in a or a in e

#主项目10篇：只读papers_pdf里的论文，避免QASPER缓存重复计入
chunks=[]; sources=[]; sections=[]
main_sources=sorted(f[:-4] for f in os.listdir(os.getenv("PAPERS_DIR")) if f.endswith(".pdf"))
for source in main_sources:
    cache_path=os.path.join(SECTIONS_DIR,f"{source}.json")
    if not os.path.exists(cache_path):
        continue
    sec_list=json.load(open(cache_path,encoding="utf-8"))
    for title,text in sec_list:
        for part in split_fixed(text):
            if len(part.strip())<20:
                continue
            chunks.append(part)
            sources.append(source)
            sections.append(title)

#QASPER 40篇：JSON full_text
data=json.load(open(DATA_PATH,encoding="utf-8"))
papers=[{**v,"arxiv_id":k} for k,v in data.items()]
qasper_papers=select_qasper(papers)
qasper_ids={p["arxiv_id"] for p in qasper_papers}
paper_sections={}
for p in qasper_papers:
    aid=p["arxiv_id"]
    paper_sections[aid]={}
    for block in p.get("full_text",[]):
        sec=block.get("section_name") or "Unknown"
        text="\n".join(block.get("paragraphs",[]))
        paper_sections[aid][sec]=text
        for part in split_fixed(text):
            if len(part.strip())<20:
                continue
            chunks.append(part)
            sources.append(aid)
            sections.append(sec)
logger.info(f"合并语料：{len(chunks)}切片，{len(sources)}条来源")

model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])

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
    items=[{"idx":j,"text":chunks[j]} for j,_ in cands]
    rr=rerank(query,items,top_n=top_n)
    return [it["idx"] for it in rr]

def neighbors(idxs,width=2):
    out=set()
    for j in idxs:
        out.add(j)
        for delta in range(1,width+1):
            for nb in (j-delta,j+delta):
                if 0<=nb<len(chunks) and sources[nb]==sources[j] and sections[nb]==sections[j]:
                    out.add(nb)
    return out

def base_top(query,top_n=5):
    cands=weighted_candidates(query,10,0.5)
    return [j for j,_ in cands[:top_n]]

def chapter_hit(idxs,truth_source,truth_sections,use_section_match=False):
    for j in idxs:
        if sources[j]!=truth_source:
            continue
        if use_section_match:
            for sec in truth_sections:
                if section_match(sections[j],sec):
                    return True
        elif sections[j] in truth_sections:
            return True
    return False

paper_map={
    "Attention Is All You Need":"Attention_Is_All_You_Need",
    "BERT":"BERT","Chain-of-Thought":"Chain_of_Thought","FlashAttention":"FlashAttention",
    "GraphRAG":"GraphRAG","LoRA":"LoRA","RAG Original":"RAG_Original_Paper",
    "ReAct":"ReAct","GPT2":"GPT2","GPT-2":"GPT2","RoFormer":"RoFormer_RoPE",
}

#主项目100题
q_file=os.getenv("TEST_QUESTIONS_FILE")
questions_main=re.findall(r"问题：(.+)",open(q_file,encoding="utf-8").read())
v_content=open(os.getenv("TEST_REFERENCES_FILE"),encoding="utf-8").read()
fixed_refs=[]
for block in re.split(r"(?=第\d+题)",v_content):
    if not block.strip():
        continue
    m=re.search(r"修正后引用：(.*)",block)
    if m and m.group(1).strip():
        fixed_refs.append(m.group(1).strip())
    else:
        m2=re.search(r"原始引用：(.*)",block)
        if m2:
            fixed_refs.append(m2.group(1).strip())

test_cases=[]
for q,ref in zip(questions_main,fixed_refs):
    parts=re.split(r"\s+-\s+",ref,maxsplit=1)
    if len(parts)<2:
        continue
    paper_part,section_part=parts
    source=None
    for key,src in paper_map.items():
        if key.lower() in paper_part.lower():
            source=src; break
    if source:
        test_cases.append((q,source,[section_part],True))

#QASPER 263题
for p in qasper_papers:
    aid=p["arxiv_id"]
    sec_map=paper_sections[aid]
    for qa in p.get("qas",[]):
        q=qa.get("question","")
        if not q:
            continue
        truth_sections=set()
        for ans in qa.get("answers",[]):
            answer_obj=ans.get("answer") or {}
            for ev in answer_obj.get("evidence") or []:
                evn=norm_text(ev)
                if len(evn)<20:
                    continue
                probe=evn[:80]
                for sec,text in sec_map.items():
                    if probe in norm_text(text):
                        truth_sections.add(sec)
                        break
        if truth_sections:
            test_cases.append((q,aid,list(truth_sections),False))

stats={"base":0,"rerank":0,"neighbor":0}
total=len(test_cases)
t0=time.time()
for q,truth_source,truth_sections,use_match in test_cases:
    use_match=bool(use_match)
    stats["base"]+=chapter_hit(base_top(q),truth_source,truth_sections,use_match)
    rr=rerank_top(q,5)
    stats["rerank"]+=chapter_hit(rr,truth_source,truth_sections,use_match)
    stats["neighbor"]+=chapter_hit(neighbors(rr,2),truth_source,truth_sections,use_match)
elapsed=time.time()-t0

lines=[]
lines.append("## 合并评测集（50篇，363题，章节级 HitRate@5，bge-m3）")
lines.append(f"- 切片数：{len(chunks)}，测试题数：{total}")
lines.append(f"- 混合检索 baseline：{stats['base']/total:.1%}")
lines.append(f"- +Rerank：{stats['rerank']/total:.1%}")
lines.append(f"- +Rerank+邻接±2：{stats['neighbor']/total:.1%}")
lines.append(f"- 平均单题耗时：{elapsed/total:.2f}s")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"merged_chapter.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/merged_chapter.md")
