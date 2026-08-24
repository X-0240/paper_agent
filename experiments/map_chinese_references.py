import difflib
import json
import logging
import os
import re
import sys
from dotenv import load_dotenv
from doc_ingest import load_sections

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

OUT_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"corrected_chinese_cases.json")

paper_map={
    "Attention Is All You Need":"Attention_Is_All_You_Need","BERT":"BERT",
    "Chain-of-Thought":"Chain_of_Thought","FlashAttention":"FlashAttention","GraphRAG":"GraphRAG",
    "LoRA":"LoRA","RAG Original":"RAG_Original_Paper","ReAct":"ReAct","GPT2":"GPT2",
    "GPT-2":"GPT2","RoFormer":"RoFormer_RoPE",
}

def norm(s):
    s=re.sub(r"^[\d.]+\s*","",s.lower())
    s=re.sub(r"^(figure|table|abstract|appendix|proposition|theorem|section)\s*[\d.]*\s*","",s)
    return re.sub(r"[^a-z ]","",s).strip()

def section_match(title,expected):
    e=norm(expected); a=norm(title)
    if not e or not a:
        return False
    return e in a or a in e

def english_terms(truth_section):
    #从参考答案章节名提取英文术语，用于在英文正文里定位
    words=re.findall(r"[A-Za-z][A-Za-z\-]{2,}",truth_section)
    return [w.lower() for w in words if len(w)>=3]

def map_section(source,truth_section,question):
    #按优先级把参考答案章节映射到实际解析章节
    sections=load_sections(source)
    if not sections:
        return "", "no_sections"
    titles=[s.get("title","") for s in sections]
    for title in titles:
        if section_match(title,truth_section):
            return title,"exact"
    norms={norm(t):t for t in titles}
    match=difflib.get_close_matches(norm(truth_section),list(norms.keys()),n=1,cutoff=0.6)
    if match:
        return norms[match[0]],"fuzzy"
    label=re.search(r"(table|figure)\s*\d+",truth_section,re.I)
    if label:
        target=label.group(0).lower()
        for s in sections:
            if target in (s.get("text","") or "").lower():
                return s.get("title",""),"content_label"
    terms=english_terms(truth_section)
    best=None; best_count=0
    for s in sections:
        text=(s.get("text","") or "").lower()
        count=sum(1 for t in terms if t in text)
        if count>best_count:
            best=s.get("title",""); best_count=count
    if best_count>=1:
        return best,"english_terms"
    return "", "not_mapped"

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

cases=[]
mapped=0
not_mapped=0
for q,ref in zip(questions_main,fixed_refs):
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
    mapped_section,method=map_section(source,section_part.strip(),q)
    if mapped_section:
        mapped+=1
    else:
        not_mapped+=1
    cases.append({
        "question":q,
        "source":source,
        "original_section":section_part.strip(),
        "mapped_section":mapped_section,
        "method":method
    })

with open(OUT_PATH,"w",encoding="utf-8") as f:
    json.dump(cases,f,ensure_ascii=False,indent=2)
print(f"总数：{len(cases)}，映射成功：{mapped}（{mapped/len(cases):.1%}），未映射：{not_mapped}")
methods={}
for c in cases:
    methods[c["method"]]=methods.get(c["method"],0)+1
print(f"映射方式分布：{methods}")
