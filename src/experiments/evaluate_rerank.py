import logging
import os
import re
import sys
import time
from dotenv import load_dotenv
from rag_tool import search_papers_structured, search_papers_rerank

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

#读取100题（与既有评估一致）
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

#对比：无Rerank（混合Top-5）vs 有Rerank（候选25精排Top-5）
base_hits=0; rerank_hits=0; changed=0; total=0; rerank_cost=0.0
for q,ref in zip(questions,fixed_refs):
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
    base=[r["source"] for r in search_papers_structured(q,k=5)]
    t0=time.time()
    rr=[r["source"] for r in search_papers_rerank(q,k=5,candidates=25)]
    rerank_cost+=time.time()-t0
    base_hits+=source in base
    rerank_hits+=source in rr
    changed+=set(base)!=set(rr)

lines=[]
lines.append("## Rerank 量化对比（100题，论文级 HitRate@5）")
lines.append(f"- 无Rerank：{base_hits/total:.1%}")
lines.append(f"- 有Rerank：{rerank_hits/total:.1%}")
lines.append(f"- 变化：{(rerank_hits-base_hits)/total:+.1%}")
lines.append(f"- Top-5集合发生变化：{changed}/{total}")
lines.append(f"- 平均Rerank耗时：{rerank_cost/total:.2f}s/题")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"rerank_compare.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/rerank_compare.md")
