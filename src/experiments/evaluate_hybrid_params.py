import json
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

#读取100题（与paper_rag评估脚本一致）
q_file=os.getenv("TEST_QUESTIONS_FILE")
content=open(q_file,encoding="utf-8").read()
questions=re.findall(r"问题：(.+)",content)
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

#论文名到库内source的映射
paper_map={
    "Attention Is All You Need":"Attention_Is_All_You_Need",
    "BERT":"BERT","Chain-of-Thought":"Chain_of_Thought","FlashAttention":"FlashAttention",
    "GraphRAG":"GraphRAG","LoRA":"LoRA","RAG Original":"RAG_Original_Paper",
    "ReAct":"ReAct","GPT2":"GPT2","GPT-2":"GPT2","RoFormer":"RoFormer_RoPE",
}

def norm_slice(scores):
    #和rag_tool一致：按片段内最大值归一化到0-1
    max_s=max(scores) if scores else 0
    return [s/max_s if max_s>0 else 0 for s in scores]

def top_k(query_pre,k,alpha):
    #忠实复刻hybrid_search：每路取top 2k，加权融合后取top k
    vec_scores,vec_idx=query_pre["vec"]
    bm_scores,bm_idx=query_pre["bm"]
    combined={}
    vs=norm_slice(list(vec_scores[:2*k]))
    for s,j in zip(vs,vec_idx[:2*k]):
        combined[j]=combined.get(j,0)+alpha*s
    bs=norm_slice([bm_scores[j] for j in bm_idx[:2*k]])
    for s,j in zip(bs,bm_idx[:2*k]):
        combined[j]=combined.get(j,0)+(1-alpha)*s
    top=sorted(combined.items(),key=lambda x:x[1],reverse=True)[:k]
    return [sources[j] for j,_ in top]

#预计算每题：向量top20和BM25 top20，不同k/alpha共用，避免重复编码
pre=[]
for q in questions:
    q_vec=model.encode([q])
    q_vec=q_vec/np.linalg.norm(q_vec)
    vec_scores,vec_idx=index.search(q_vec.astype("float32"),20)
    bm_scores=bm25.get_scores(list(jieba.cut(q)))
    bm_idx=sorted(range(N),key=lambda i:bm_scores[i],reverse=True)[:20]
    pre.append({"vec":(vec_scores[0],vec_idx[0]),"bm":(bm_scores,bm_idx)})

def eval_one(alpha,k):
    #论文级Hit Rate@k：top k里是否出现标准答案论文
    hits=0; total=0
    for qp,ref in zip(pre,fixed_refs):
        parts=re.split(r"\s+-\s+",ref,maxsplit=1)
        if len(parts)<2:
            continue
        paper_part,_=parts
        source=None
        for key,src in paper_map.items():
            if key.lower() in paper_part.lower():
                source=src; break
        if not source:
            continue
        total+=1
        if source in top_k(qp,k,alpha):
            hits+=1
    return hits/total if total else 0

#alpha对照：固定k=5，记录@3和@5
alpha_rows=[]
for alpha in [0.3,0.5,0.7]:
    hr3=eval_one(alpha,3)
    hr5=eval_one(alpha,5)
    alpha_rows.append((alpha,hr3,hr5))

#k对照：固定alpha=0.5
k_rows=[]
for k in [3,5,10]:
    k_rows.append((k,eval_one(0.5,k)))

#输出表格并保存
lines=[]
lines.append("## alpha 对照实验（固定 k=5）")
lines.append("| alpha | HitRate@3 | HitRate@5 |")
lines.append("|---|---|---|")
for alpha,hr3,hr5 in alpha_rows:
    lines.append(f"| {alpha} | {hr3:.1%} | {hr5:.1%} |")
lines.append("")
lines.append("## k 对照实验（固定 alpha=0.5）")
lines.append("| k | HitRate@k |")
lines.append("|---|---|")
for k,hr in k_rows:
    lines.append(f"| {k} | {hr:.1%} |")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
out_path=os.path.join(out_dir,"alpha_k_hitrate.md")
with open(out_path,"w",encoding="utf-8") as f:
    f.write(report+"\n")
print(f"\n已保存：{out_path}")
