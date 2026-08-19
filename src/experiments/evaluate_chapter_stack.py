import logging
import os
import re
import sys
from dotenv import load_dotenv
from rag_tool import hybrid_search, filter_results, sources, sections, documents
from rerank import rerank

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

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

def chapter_hit(idxs,source,section_part):
    return any(sources[j]==source and section_match(sections[j],section_part) for j in idxs)

def neighbors(idxs,width=2):
    out=set()
    for j in idxs:
        out.add(j)
        for delta in range(1,width+1):
            for nb in (j-delta,j+delta):
                if 0<=nb<len(sources) and sources[nb]==sources[j] and sections[nb]==sections[j]:
                    out.add(nb)
    return out

stats={"base":0,"rerank":0,"neighbor":0}
total=0
for q,ref in zip(questions,fixed_refs):
    parts=re.split(r"\s+-\s+",ref,maxsplit=1)
    if len(parts)<2:
        continue
    paper_part,section_part=parts
    source=None
    for key,src in paper_map.items():
        if key.lower() in paper_part.lower():
            source=src; break
    if not source:
        continue
    total+=1
    #baseline：混合检索top5
    top,status=hybrid_search(q,5,0.5)
    top=filter_results(top)
    stats["base"]+=chapter_hit([j for j,_ in top],source,section_part)
    #候选25 + Rerank
    cands,status2=hybrid_search(q,25,0.5)
    cands=filter_results(cands)
    items=[{"idx":j,"text":documents[j]} for j,_ in cands]
    rr=rerank(q,items,top_n=5)
    rr_idx=[it["idx"] for it in rr]
    stats["rerank"]+=chapter_hit(rr_idx,source,section_part)
    stats["neighbor"]+=chapter_hit(neighbors(rr_idx,2),source,section_part)

lines=[]
lines.append("## 主项目100题·章节级检索栈对比（bge-m3）")
lines.append(f"- 混合检索 baseline：{stats['base']/total:.1%}")
lines.append(f"- +Rerank：{stats['rerank']/total:.1%}")
lines.append(f"- +Rerank+邻接±2：{stats['neighbor']/total:.1%}")
lines.append(f"- 题数：{total}")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"chapter_stack.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/chapter_stack.md")
