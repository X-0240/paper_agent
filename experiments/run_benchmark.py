import logging
import os
import sys
import time
from pipeline import classify_task, simple_answer, survey_pipeline
from llm_api import set_tracker

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO)

QUESTIONS=[
    "什么是自注意力机制",
    "LoRA 低秩适配的原理是什么",
    "RAG 是什么",
    "对比 Transformer 和 BERT 的架构差异",
    "对比 FlashAttention 和标准 Transformer 注意力",
]

def run_one(question):
    #单题指标：成功、耗时、LLM调用次数、token
    tracker={"calls":0,"tokens":0}
    set_tracker(tracker)
    t0=time.time(); success=True; err=""
    try:
        route=classify_task(question)
        if route=="simple":
            simple_answer(question)
        else:
            survey_pipeline(question)
    except Exception as e:
        success=False
        err=str(e)[:80]
    cost=time.time()-t0
    set_tracker(None)
    return {
        "question":question,
        "route":route,
        "success":success,
        "time":round(cost,1),
        "calls":tracker["calls"],
        "tokens":tracker["tokens"],
        "err":err,
    }

rows=[run_one(q) for q in QUESTIONS]

#汇总统计
total=len(rows)
ok=[r for r in rows if r["success"]]
simple=[r for r in rows if r["route"]=="simple"]
survey=[r for r in rows if r["route"]=="survey"]

def avg(items,key):
    return round(sum(r[key] for r in items)/len(items),1) if items else 0

lines=[]
lines.append("## 流水线指标（5题）")
lines.append("| 问题 | 路由 | 成功 | 耗时s | LLM调用 | token | 失败原因 |")
lines.append("|---|---|---|---|---|---|---|")
for r in rows:
    lines.append(f"| {r['question']} | {r['route']} | {r['success']} | {r['time']} | {r['calls']} | {r['tokens']} | {r['err']} |")
lines.append("")
lines.append(f"总计：{total}题，成功率 {len(ok)/total:.0%}")
lines.append(f"全部：平均耗时 {avg(rows,'time')}s，平均调用 {avg(rows,'calls')}，平均token {avg(rows,'tokens')}")
lines.append(f"simple：平均耗时 {avg(simple,'time')}s，平均token {avg(simple,'tokens')}")
lines.append(f"survey：平均耗时 {avg(survey,'time')}s，平均token {avg(survey,'tokens')}")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
out_path=os.path.join(out_dir,"pipeline_metrics.md")
with open(out_path,"w",encoding="utf-8") as f:
    f.write(report+"\n")
print(f"\n已保存：{out_path}")
