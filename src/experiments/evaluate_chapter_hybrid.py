import logging
import os
import re
import sys
from dotenv import load_dotenv
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

#章节级HitRate@3/@5：top-k里出现正确论文的正确章节
hits3=0; hits5=0; total=0
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
    results=search_papers_structured(q,k=5,alpha=0.5)
    hit3=any(r["source"]==source and section_match(r["section"],section_part) for r in results[:3])
    hit5=any(r["source"]==source and section_match(r["section"],section_part) for r in results[:5])
    hits3+=hit3
    hits5+=hit5

lines=[]
lines.append("## 主项目100题·章节级 HitRate（混合检索 alpha=0.5）")
lines.append(f"- 章节级@3：{hits3/total:.1%}")
lines.append(f"- 章节级@5：{hits5/total:.1%}")
lines.append(f"- 题数：{total}")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"chapter_hybrid.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/chapter_hybrid.md")
