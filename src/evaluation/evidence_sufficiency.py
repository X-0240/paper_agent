import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from llm_api import call_deepseek_once

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#判读标准必须先冻结再跑：三条同时满足才算"充分"，禁止看到结果后修改
RUBRIC=(
    "你是检索证据充分性判读员。只看给定证据，不使用外部常识。\n"
    "判定为“充分”必须同时满足三条：\n"
    "1) 问题涉及的所有核心信息点都能在证据里找到明确原文支撑；\n"
    "2) 证据内部不存在互相矛盾的表述；\n"
    "3) 不需要额外常识或外部资料即可从证据直接得出答案。\n"
    "任一条不满足即判“不充分”，并指出缺少哪个信息点。\n"
    "只输出一行JSON：{\"sufficient\":true/false,\"missing\":\"缺失的信息点，充分时留空\",\"reason\":\"简短理由\"}"
)
RUBRIC_SHA=hashlib.sha256(RUBRIC.encode("utf-8")).hexdigest()[:16]


def load_json(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question,items):
    #证据按最终排名拼接，保留来源和章节便于人工复核
    blocks=[]
    for i,item in enumerate(items,1):
        text=(item.get("text") or "").strip().replace("\n"," ")
        blocks.append(f"[{i}] 论文：{item.get('source')}｜章节：{item.get('section')}\n{text[:1200]}")
    return f"问题：{question}\n\n检索回来的证据：\n" + "\n\n".join(blocks)


def parse_reply(text):
    text=(text or "").strip()
    start=text.find("{")
    end=text.rfind("}")
    if start<0 or end<=start:
        return {"sufficient":None,"missing":"","reason":f"解析失败：{text[:120]}"}
    try:
        data=json.loads(text[start:end+1])
    except Exception as e:
        return {"sufficient":None,"missing":"","reason":f"解析失败：{e}"}
    return {"sufficient":bool(data.get("sufficient")),"missing":str(data.get("missing","")),"reason":str(data.get("reason",""))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repro",default=os.path.join(BASE,"evaluation","results","v300_repro.json"))
    parser.add_argument("--sample",default=os.path.join(BASE,"evaluation","results","v300_label_audit_sample.json"))
    parser.add_argument("--config",default="sealed_candidate")
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--output",default=os.path.join(BASE,"evaluation","results","v300_evidence_sufficiency.json"))
    args=parser.parse_args()

    load_dotenv()
    repro=load_json(args.repro)
    sample=load_json(args.sample)
    wanted=[s["question"].strip() for s in sample["sample"]]
    rows={r["question"].strip():r for r in repro["rows"]}
    todo=[q for q in wanted if q in rows]
    if args.limit:
        todo=todo[:args.limit]

    results=[]
    started=time.time()
    for i,question in enumerate(todo,1):
        row=rows[question]
        items=row.get("candidate_top") or []
        reply=call_deepseek_once(
            [{"role":"system","content":RUBRIC},{"role":"user","content":build_prompt(question,items)}],
            temperature=0.0,
            max_tokens=300,
            timeout=30,
            thinking="disabled",
            record_usage=True,
        )
        verdict=parse_reply(reply["choices"][0]["message"]["content"])
        results.append({"question":question,"paper_id":row.get("paper_id"),"bucket":next((s["bucket"] for s in sample["sample"] if s["question"].strip()==question),"") ,
                        "config":args.config,"hit5":row["results"][args.config]["hit5"],**verdict})
        if i%10==0 or i==len(todo):
            print(f"  {i}/{len(todo)} 已跑 {time.time()-started:.0f}s")

    judged=[r for r in results if r["sufficient"] is not None]
    sufficient=sum(1 for r in judged if r["sufficient"])
    by_bucket={}
    for r in judged:
        by_bucket.setdefault(r["bucket"],[0,0])
        by_bucket[r["bucket"]][0]+=1
        by_bucket[r["bucket"]][1]+=1 if r["sufficient"] else 0
    out={
        "rubric":RUBRIC,
        "rubric_sha":RUBRIC_SHA,
        "config":args.config,
        "n":len(judged),
        "sufficient":sufficient,
        "rate":sufficient/len(judged) if judged else 0,
        "by_bucket":{k:{"n":v[0],"sufficient":v[1]} for k,v in by_bucket.items()},
        "rows":results,
    }
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)
    print(f"证据充分（LLM判读）：{sufficient}/{len(judged)}")
    for k,v in out["by_bucket"].items():
        print(f"  {k}: {v['sufficient']}/{v['n']}")
    print(f"rubric_sha={RUBRIC_SHA}")
    print(f"已保存：{args.output}")


if __name__=="__main__":
    main()
