import logging
import os
import re
import sys
import numpy as np
from dotenv import load_dotenv
import jieba
from rag_tool import index, model, sources, bm25

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)
N=len(sources)
CANDIDATES=10

#读取100题
q_file=os.getenv("TEST_QUESTIONS_FILE")
questions=re.findall(r"问题：(.+)",open(q_file,encoding="utf-8").read())
v_file=os.getenv("TEST_REFERENCES_FILE")
v_content=open(v_file,encoding="utf-8").read()
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

paper_map={
    "Attention Is All You Need":"Attention_Is_All_You_Need",
    "BERT":"BERT","Chain-of-Thought":"Chain_of_Thought","FlashAttention":"FlashAttention",
    "GraphRAG":"GraphRAG","LoRA":"LoRA","RAG Original":"RAG_Original_Paper",
    "ReAct":"ReAct","GPT2":"GPT2","GPT-2":"GPT2","RoFormer":"RoFormer_RoPE",
}

def norm(scores):
    max_s=max(scores) if scores else 0
    return [s/max_s if max_s>0 else 0 for s in scores]

def weighted_top(pre,alpha=0.5,k=5):
    vec_scores,vec_idx=pre["vec"]
    bm_scores,bm_idx=pre["bm"]
    combined={}
    vs=norm(list(vec_scores))
    for s,j in zip(vs,vec_idx):
        combined[j]=combined.get(j,0)+alpha*s
    bs=norm([bm_scores[j] for j in bm_idx])
    for s,j in zip(bs,bm_idx):
        combined[j]=combined.get(j,0)+(1-alpha)*s
    top=sorted(combined.items(),key=lambda x:x[1],reverse=True)[:k]
    return [sources[j] for j,_ in top]

def rrf_top(pre,k=60,top_n=5):
    #RRF：只看排名不算分数，score=Σ 1/(k+rank)
    scores={}
    for idxs in (pre["vec"][1],pre["bm"][1]):
        for rank,j in enumerate(idxs):
            scores[j]=scores.get(j,0)+1/(k+rank+1)
    top=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_n]
    return [sources[j] for j,_ in top]

#预计算每题两路top-25
pre=[]
for q in questions:
    q_vec=model.encode([q])
    q_vec=q_vec/np.linalg.norm(q_vec)
    vec_scores,vec_idx=index.search(q_vec.astype("float32"),CANDIDATES)
    bm_scores=bm25.get_scores(list(jieba.cut(q)))
    bm_idx=sorted(range(N),key=lambda i:bm_scores[i],reverse=True)[:CANDIDATES]
    pre.append({"vec":(vec_scores[0],vec_idx[0]),"bm":(bm_scores,bm_idx)})

weighted_hits=0; rrf_hits=0; changed=0; total=0
for qp,ref in zip(pre,fixed_refs):
    parts=re.split(r"\s+-\s+",ref,maxsplit=1)
    if len(parts)<2:
        continue
    source=None
    for key,src in paper_map.items():
        if key.lower() in parts[0].lower():
            source=src; break
    if not source:
        continue
    total+=1
    w=weighted_top(qp)
    r=rrf_top(qp)
    weighted_hits+=source in w
    rrf_hits+=source in r
    changed+=set(w)!=set(r)

lines=[]
lines.append("## RRF vs 加权融合（100题，论文级 HitRate@5，候选10）")
lines.append(f"- 加权(alpha=0.5)：{weighted_hits/total:.1%}")
lines.append(f"- RRF(k=60)：{rrf_hits/total:.1%}")
lines.append(f"- Top-5集合变化：{changed}/{total}")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"rrf_compare.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/rrf_compare.md")
