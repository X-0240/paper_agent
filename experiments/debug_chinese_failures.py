import json
import logging
import os
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
FAISS_PATH=os.getenv("FAISS_PATH")
MODEL_PATH=os.getenv("MODEL_PATH")
PAPERS_DIR=os.getenv("PAPERS_DIR")

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

main_sources=set(sorted(f[:-4] for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")))
meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
all_chunks=meta["documents"]; all_sources=meta["sources"]; all_sections=meta["sections"]
keep=[i for i,s in enumerate(all_sources) if s in main_sources]
chunks=[all_chunks[i] for i in keep]
sources=[all_sources[i] for i in keep]
sections=[all_sections[i] for i in keep]

model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])

paper_map={
    "Attention Is All You Need":"Attention_Is_All_You_Need","BERT":"BERT",
    "Chain-of-Thought":"Chain_of_Thought","FlashAttention":"FlashAttention","GraphRAG":"GraphRAG",
    "LoRA":"LoRA","RAG Original":"RAG_Original_Paper","ReAct":"ReAct","GPT2":"GPT2",
    "GPT-2":"GPT2","RoFormer":"RoFormer_RoPE",
}
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
        test_cases.append((q,source,[section_part],ref))

def top5(query):
    qv=model.encode([query]); qv=qv/np.linalg.norm(qv)
    vs,vi=index.search(qv.astype("float32"),10)
    bm=bm25.get_scores(tokenize(query))
    bm_idx=sorted(range(len(bm)),key=lambda i:bm[i],reverse=True)[:10]
    comb={}
    if max(vs[0])>0: vs=vs/max(vs)
    for s,j in zip(vs[0],vi[0]): comb[j]=comb.get(j,0)+0.5*float(s)
    bm=np.array([bm[j] for j in bm_idx]); 
    if max(bm)>0: bm=bm/max(bm)
    for s,j in zip(bm,bm_idx): comb[j]=comb.get(j,0)+0.5*float(s)
    return [j for j,_ in sorted(comb.items(),key=lambda x:x[1],reverse=True)[:5]]

shown=0
fail_total=0
fail_no_section=0
for q,src,truth_sections,ref in test_cases:
    idxs=top5(q)
    hit=any(sources[j]==src and any(section_match(sections[j],t) for t in truth_sections) for j in idxs)
    if hit:
        continue
    fail_total+=1
    truth_exists=any(s==src and section_match(sec,truth_sections[0]) for s,sec in zip(sources,sections))
    if not truth_exists:
        fail_no_section+=1
    top1=sources[idxs[0]]+" :: "+sections[idxs[0]]
    shown+=1
    if shown<=8:
        print(f"Q：{q}")
        print(f"  真理：{src} :: {truth_sections[0]} | 章节存在={truth_exists}")
        print(f"  top1：{top1}")
        print(f"  ref：{ref[:120]}")
        print()
print(f"失败总数：{fail_total}，其中真理章节在我们的解析结构中不存在：{fail_no_section}（{fail_no_section/fail_total:.1%}）")
