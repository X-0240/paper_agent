import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import json
import logging
import sys
import time
from dotenv import load_dotenv
from agent_react import react
from agent1_retrieve import agent1_tools
from pipeline import survey_pipeline
from llm_api import safe_call_deepseek, set_tracker

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)

QUESTIONS=[
    "对比 Transformer 和 BERT 的架构差异，写一篇综述",
    "梳理注意力机制从 Transformer 到 FlashAttention 的演进，写一篇综述",
    "综述 RAG 和 ReAct 各自解决什么问题，以及它们如何协同",
]

SINGLE_PROMPT="""你是论文综述Agent。根据论文知识库中的内容写综述，必须引用来源（论文名和章节）。
规则：
1. 需要信息就调用search_papers检索，片段不完整就用read_section读完整章节
2. 只基于检索到的内容，不要编造
3. 最终用Finish输出完整综述正文
必须严格按格式输出：Thought/Action/Action Input。"""

JUDGE_PROMPT="""你是严格的评测官。比较同一问题下两个Agent的输出，从四个维度各打1-10分：
- 覆盖度：是否覆盖了题目要求的相关论文/角度
- 事实性：是否忠于资料，有无明显编造
- 结构：是否成体系（背景/对比/结论）
- 引用：是否标注来源，引用是否可信
- 幻觉计数：输出中明显没有资料支撑的断言数量
最后给出结论：哪个更好、为什么。
只输出JSON：
{
  "single": {"coverage":0,"factuality":0,"structure":0,"citation":0,"hallucinations":0},
  "multi": {"coverage":0,"factuality":0,"structure":0,"citation":0,"hallucinations":0},
  "verdict": "single/multi/平局",
  "reason": "简要原因"
}"""

def run_single(q):
    tracker={"calls":0,"tokens":0}
    set_tracker(tracker)
    t0=time.time()
    answer,history=react(q,agent1_tools,SINGLE_PROMPT,max_steps=8)
    cost=time.time()-t0
    set_tracker(None)
    return answer,cost,tracker["calls"],tracker["tokens"]

def run_multi(q):
    tracker={"calls":0,"tokens":0}
    set_tracker(tracker)
    t0=time.time()
    answer=survey_pipeline(q,human_confirm=False)
    cost=time.time()-t0
    set_tracker(None)
    return answer,cost,tracker["calls"],tracker["tokens"]

def judge(q,single,multi):
    user=(
        f"问题：{q}\n\n"
        f"### 单Agent输出：\n{single[:6000]}\n\n"
        f"### 多Agent输出：\n{multi[:6000]}"
    )
    messages=[{"role":"system","content":JUDGE_PROMPT},{"role":"user","content":user}]
    text=""
    for attempt in range(3):
        result=safe_call_deepseek(messages,temperature=0.1,max_tokens=2000)
        text=result["choices"][0]["message"]["content"] or ""
        text=text.strip()
        if text.startswith("```"):
            text=text.strip("`")
            if text.startswith("json"):
                text=text[4:]
        start=text.find("{"); end=text.rfind("}")
        try:
            data=json.loads(text[start:end+1]) if start!=-1 and end!=-1 else {}
            if data:
                return data
        except Exception:
            pass
        messages.append({"role":"user","content":"你上次的输出不是合法JSON，请只输出JSON对象。"})
    return {"verdict":"parse_error","reason":text[:200]}

rows=[]
for qi,q in enumerate(QUESTIONS):
    single,st,sc,stk=run_single(q)
    multi,mt,mc,mtk=run_multi(q)
    out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results","single_vs_multi_outputs")
    os.makedirs(out_dir,exist_ok=True)
    with open(os.path.join(out_dir,f"{qi}_single.txt"),"w",encoding="utf-8") as f:
        f.write(single)
    with open(os.path.join(out_dir,f"{qi}_multi.txt"),"w",encoding="utf-8") as f:
        f.write(multi)
    j=judge(q,single,multi)
    rows.append({
        "q":q,
        "single_time":round(st,1),"multi_time":round(mt,1),
        "single_calls":sc,"multi_calls":mc,
        "single_tokens":stk,"multi_tokens":mtk,
        "judge":j,
    })

lines=[]
lines.append("## 单 Agent vs 3-Agent 对比（3 题综述）")
lines.append("| 问题 | 单耗时s | 多耗时s | 单调用 | 多调用 | 单token | 多token | 判定 |")
lines.append("|---|---|---|---|---|---|---|---|")
for r in rows:
    j=r["judge"]
    lines.append(f"| {r['q']} | {r['single_time']} | {r['multi_time']} | {r['single_calls']} | {r['multi_calls']} | {r['single_tokens']} | {r['multi_tokens']} | {j.get('verdict')} |")
lines.append("")
for r in rows:
    j=r["judge"]
    lines.append(f"### {r['q']}")
    lines.append(f"判定：{j.get('verdict')}  原因：{j.get('reason')}")
    lines.append(f"- 单Agent评分：{j.get('single')}")
    lines.append(f"- 多Agent评分：{j.get('multi')}")
report="\n".join(lines)
print(report)

out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_results")
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,"single_vs_multi.md"),"w",encoding="utf-8") as f:
    f.write(report+"\n")
print("已保存：eval_results/single_vs_multi.md")
