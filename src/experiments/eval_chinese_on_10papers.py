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
CACHE_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_query_transform_cache.json")

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

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

def base_top(query,top_n=5):
    cands=weighted_candidates(query,10,0.5)
    return [j for j,_ in cands[:top_n]]

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
        test_cases.append((q,source,[section_part]))
logger.info(f"中文测试题：{len(test_cases)}")

cache=json.load(open(CACHE_PATH,encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}
queries={"原文":[q for q,_,_ in test_cases]}
queries["翻译成英文"]=[cache.get(q,q) for q in queries["原文"]]

t0=time.time()
for name,qs in queries.items():
    hits=0
    for (q,source,truth_sections),tq in zip(test_cases,qs):
        hits+=chapter_hit(base_top(tq),source,truth_sections)
    print(f"{name}（10篇专属索引）：章节级@5={hits/len(test_cases):.1%}（{hits}/{len(test_cases)}）")
print(f"评测耗时：{time.time()-t0:.1f}s")
