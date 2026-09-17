import argparse
import json
import os
import re
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from doc_ingest import load_sections
from llm_api import safe_call_deepseek

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTION_DIR=os.path.join(BASE,"evaluation","questions")

FORWARD_PROMPT="""你是学术论文调研系统的出题人。你会看到一篇论文的完整章节文本。
请提出 {n} 个中文问题，模拟真实用户做论文调研时会问的问题，覆盖事实、方法、实验、局限、对比等类型。
要求：
1. 问题必须能由论文原文回答，且不依赖图表编号之外的外部知识；
2. 每题给出 1-3 句英文证据原句，必须逐字来自论文原文；
3. 给出证据所在章节标题；
4. 只输出 JSON 数组，格式：
[{{"question":"中文问题","evidence_sentences":["原文句1"],"section":"章节标题","question_type":"fact|method|experiment|limitation|comparison|multi_hop"}}]"""

REVERSE_PROMPT="""下面是一篇论文的一个片段。请提出 1 个中文问题，这个问题正好能由该片段回答。
要求：
1. 问题面向论文调研，自然、具体；
2. 给出该片段中 1-3 句英文证据原句，必须逐字来自片段；
3. 只输出 JSON 数组，格式：
[{{"question":"中文问题","evidence_sentences":["原文句1"],"question_type":"fact|method|experiment|limitation"}}]"""

def extract_json_array(text):
    text=text.strip()
    if text.startswith("```"):
        text=text.strip("`")
        if text.startswith("json"):
            text=text[4:]
    #优先用raw_decode解析第一个完整数组，避免两个JSON拼在一起
    decoder=json.JSONDecoder()
    for idx,ch in enumerate(text):
        if ch!="[":
            continue
        try:
            obj,end=decoder.raw_decode(text[idx:])
            if isinstance(obj,list):
                return obj
        except json.JSONDecodeError:
            continue
    #兜底：去掉尾逗号后再试
    start=text.find("["); end=text.rfind("]")
    if start!=-1 and end!=-1:
        fixed=re.sub(r",\s*([\]}])",r"\1",text[start:end+1])
        return json.loads(fixed)
    raise ValueError("未输出JSON数组")

def call_json(messages,max_tokens=4000):
    last_error=None
    for attempt in range(3):
        result=safe_call_deepseek(messages,temperature=0.3,max_tokens=max_tokens)
        content=result["choices"][0]["message"]["content"] or ""
        try:
            return extract_json_array(content)
        except Exception as e:
            last_error=e
            #把失败输出回填，明确要求只输出合法JSON
            messages=messages+[
                {"role":"assistant","content":content[:2000]},
                {"role":"user","content":"上面的输出不是合法JSON数组，请只输出一个JSON数组，不要解释、不要代码块。"}
            ]
    raise ValueError(f"JSON解析失败：{last_error}")

def forward(paper_ids,per_paper,out_path,append=False):
    questions=[]
    if append and os.path.exists(out_path):
        questions=json.load(open(out_path,encoding="utf-8"))
    def save_progress():
        os.makedirs(os.path.dirname(out_path),exist_ok=True)
        with open(out_path,"w",encoding="utf-8") as f:
            json.dump(questions,f,ensure_ascii=False,indent=2)
    for paper_id in paper_ids:
        sections=load_sections(paper_id)
        if not sections:
            print(f"跳过 {paper_id}：无章节")
            continue
        parts=[]; total=0
        for sec in sections:
            title=sec.get("title","")
            text=sec.get("text","") or ""
            part=f"\n\n## {title}\n{text}"
            if total+len(part)>60000:
                break
            parts.append(part); total+=len(part)
        paper_text="".join(parts)
        try:
            items=call_json([
                {"role":"system","content":FORWARD_PROMPT.format(n=per_paper)},
                {"role":"user","content":f"论文：{paper_id}\n{paper_text}"}
            ])
        except Exception as e:
            print(f"正向出题失败 {paper_id}：{e}")
            continue
        for item in items:
            if not isinstance(item,dict):
                continue
            if not item.get("question") or not item.get("evidence_sentences"):
                continue
            item["paper_id"]=paper_id
            item["source"]="forward"
            questions.append(item)
        print(f"{paper_id}: 生成 {len(items)} 题")
        save_progress()
    save_progress()
    print(f"共 {len(questions)} 题 -> {out_path}")

def reverse(chunks,out_path):
    questions=[]
    for i,(paper_id,section,text) in enumerate(chunks):
        try:
            items=call_json([
                {"role":"system","content":REVERSE_PROMPT},
                {"role":"user","content":f"论文：{paper_id}\n章节：{section}\n片段：\n{text[:2000]}"}
            ])
        except Exception as e:
            print(f"反向出题失败 {i}：{e}")
            continue
        for item in items:
            item["paper_id"]=paper_id
            item["section"]=section
            item["source"]="reverse"
            questions.append(item)
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(questions,f,ensure_ascii=False,indent=2)
    print(f"共 {len(questions)} 题 -> {out_path}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",required=True,choices=["forward","reverse"])
    parser.add_argument("--papers",default="")
    parser.add_argument("--per-paper",type=int,default=3)
    parser.add_argument("--chunks",type=int,default=40)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--out",required=True)
    parser.add_argument("--append",action="store_true")
    args=parser.parse_args()
    load_dotenv()
    if args.mode=="forward":
        paper_ids=[x for x in args.papers.split(",") if x]
        if not paper_ids:
            raise SystemExit("forward模式需要--papers")
        forward(paper_ids,args.per_paper,args.out,append=args.append)
    else:
        faiss_path=os.getenv("FAISS_PATH")
        meta=json.load(open(faiss_path+".json",encoding="utf-8"))
        chunks=list(zip(meta["sources"],meta["sections"],meta["documents"]))
        paper_ids=[x for x in args.papers.split(",") if x]
        if paper_ids:
            allowed=set(paper_ids)
            chunks=[c for c in chunks if c[0] in allowed]
        import random
        random.Random(args.seed).shuffle(chunks)
        reverse(chunks[:args.chunks],args.out)

if __name__=="__main__":
    main()
