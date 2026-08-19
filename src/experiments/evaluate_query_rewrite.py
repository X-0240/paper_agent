import logging
import os
import re
import sys
from dotenv import load_dotenv
from pipeline import rewrite_query
from rag_tool import search_papers_structured

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

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

#对比：原问题检索 vs 改写后检索（论文级HitRate@5）
base_hits=0; rw_hits=0; changed=0; same_query=0; total=0
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
    rw=rewrite_query(q)
    rw_src=[r["source"] for r in search_papers_structured(rw,k=5)]
    base_hits+=source in base
    rw_hits+=source in rw_src
    changed+=set(base)!=set(rw_src)
    same_query+=rw==q

lines=[]
lines.append("## Query 改写对比（100题，论文级 HitRate@5）")
lines.append(f"- 原问题：{base_hits/total:.1%}")
lines.append(f"- 改写后：{rw_hits/total:.1%}")
lines.append(f"- 变化：{(rw_hits-base_hits)/total:+.1%}")
lines.append(f"- Top-5集合变化：{changed}/{total}")
lines.append(f"- 改写后与原问题相同：{same_query}/{total}")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"query_rewrite_compare.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/query_rewrite_compare.md")
