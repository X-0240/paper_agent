# 多轮改写的批量对照：同一批代词型追问，在多轮改写开/关两种配置下各跑一遍
# 判定标准：追问的引用里是否出现目标论文（第一轮问的那篇）
import json
import time

import httpx

BASE="http://127.0.0.1:8000"

# 每项：第一轮问题、第一轮期望目标论文、第二轮追问、第二轮期望目标论文
CASES=[
    ("Attention Is All You Need 使用了什么机制？","Attention_Is_All_You_Need",
     "它的核心组件是什么？","Attention_Is_All_You_Need"),
    ("BERT 的预训练任务有哪两个？","BERT",
     "它的输入最大长度是多少？","BERT"),
    ("RoFormer 的位置编码是怎么做的？","RoFormer_RoPE",
     "它和其他位置编码方案比有什么优势？","RoFormer_RoPE"),
    ("FlashAttention 是怎么加速的？","FlashAttention",
     "它的显存占用降了多少？","FlashAttention"),
    ("LoRA 的原理是什么？","LoRA",
     "它训练时冻结了哪些参数？","LoRA"),
    ("Chain of Thought 是怎么提升推理的？","Chain_of_Thought",
     "它的主要局限是什么？","Chain_of_Thought"),
    ("GPT2 的模型结构是怎样的？","GPT2",
     "它的参数量有多大？","GPT2"),
]


def login():
    return httpx.post(BASE+"/login",json={"username":"admin","password":"admin123"},timeout=20).json()["access_token"]


def one_turn(tok,question,thread_id=None,timeout=180):
    #返回 (thread_id, 引用论文列表, 答案前 80 字)
    params={"question":question}
    if thread_id:
        params["thread_id"]=thread_id
    tid=None
    chips=[]
    text=""
    with httpx.stream("GET",BASE+"/ask/stream",params=params,
                      headers={"Authorization":"Bearer "+tok},timeout=timeout) as r:
        buf=""
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                evt=json.loads(line[5:].strip())
            except Exception:
                continue
            if evt.get("type")=="thread":
                tid=evt.get("thread_id")
            elif evt.get("type")=="sources":
                chips=[s.get("paper_id") or s.get("title") or "" for s in (evt.get("sources") or [])]
            elif evt.get("type")=="token":
                text+=evt.get("content") or ""
    return tid,chips,text


def hit(chips,target):
    return any(target.lower() in (c or "").lower() for c in chips)


def run(label):
    tok=login()
    print("="*78)
    print("配置：",label)
    print("="*78)
    first_ok=0
    second_ok=0
    rows=[]
    for q1,target1,q2,target2 in CASES:
        tid,chips1,_=one_turn(tok,q1)
        ok1=hit(chips1,target1)
        _,chips2,ans2=one_turn(tok,q2,thread_id=tid)
        ok2=hit(chips2,target2)
        first_ok+=1 if ok1 else 0
        second_ok+=1 if ok2 else 0
        rows.append((q2,target2,ok1,ok2,chips2[:3],ans2[:60]))
        print("  追问：%-26s 目标 %-24s 第1轮命中 %s | 追问命中 %s"
              %(q2[:26],target2,("是" if ok1 else "否"),("是" if ok2 else "否")))
        if not ok2:
            print("       追问实际引用：",", ".join(chips2[:4]) or "(无)")
            print("       答案开头：",ans2.replace("\n"," ")[:70])
    print()
    print("  %s：第一轮命中 %d/%d，追问命中 %d/%d"
          %(label,first_ok,len(CASES),second_ok,len(CASES)))
    print()
    return second_ok,len(CASES)


import sys

if "--off" in sys.argv:
    run("多轮改写 关（基准）")
else:
    run("多轮改写 开（当前配置）")
