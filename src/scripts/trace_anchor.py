# 追踪实体注入的实际行为：上一轮提到哪些论文、注入后检索拿到什么
import os
import sys
import json

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import api_server  # noqa: E402  需要它的提取函数与 store
import store  # noqa: E402
import retrieval_service  # noqa: E402
from paper_entities import extract_named_papers  # noqa: E402

print("="*74)
print("A. 两个失败样例的候选池追踪")
print("="*74)

svc=retrieval_service.get_service()

CASES=[
    ("LoRA","LoRA 的原理是什么？","它训练时冻结了哪些参数？"),
    ("GPT2","GPT2 的模型结构是怎样的？","它的参数量有多大？"),
]

for target,q1,q2 in CASES:
    print()
    print("目标论文：",target)
    print("  第一轮:",q1)
    print("  追问  :",q2)
    # 模拟"上一轮的内容"：用第一轮问题 + 一段助手回答
    messages=[{"role":"user","content":q1},
              {"role":"assistant","content":f"关于 {target} 的结构说明如下……"}]
    anchor=api_server._named_papers_in_dialog(messages)
    resolved=api_server._resolve_retrieval_question(q2,messages)
    print("  提取到的实体:",anchor or "(空)")
    print("  注入后的检索查询:",resolved)
    named=svc.named_papers(resolved)
    print("  注入查询能识别到点名:",named or "(无)")
    # 看候选池里到底有没有目标论文
    items=svc.search(resolved,candidate_k=50,top_k=5)
    top_sources=[it.get("source") for it in items]
    print("  注入后 Top5 来源:",top_sources)
    has_target=target.lower() in [ (s or "").lower() for s in top_sources ]
    print("  目标是否在 Top5:",("是" if has_target else "否"))

print()
print("="*74)
print("B. 别名匹配的误报检查（同一句里命中多个论文名）")
print("="*74)

PROBES=[
    "Transformer 的编码器是怎么工作的？",
    "FlashAttention 和普通注意力比有什么优势？",
    "RoFormer 用了旋转位置编码",
    "GPT 系列的模型有哪些？",
    "对比 BERT 和 GPT2 的结构",
    "什么是检索增强生成？",
]
for text in PROBES:
    print("  %-38s -> %s"%(text,extract_named_papers(text) or "(无)"))

print()
print("="*74)
print("C. 上一轮是综述（多篇论文）时，当前实现会怎么处理")
print("="*74)

# 模拟一次综述的上一轮：问题点到两篇，回答里又出现第三篇
survey_messages=[
    {"role":"user","content":"对比 BERT 和 RoFormer 的预训练方式"},
    {"role":"assistant","content":"结合 Attention Is All You Need 的基础，三篇论文的差异是……"},
]
for q in ["它在实验中用了多大显存？","它的训练数据规模多大？"]:
    print("  追问: %s"%q)
    print("     提取到的实体: %s"%api_server._named_papers_in_dialog(survey_messages))
    print("     实际注入的查询: %s"%api_server._resolve_retrieval_question(q,survey_messages))
