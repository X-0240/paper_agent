# 阈值工作点选择：用更大的样例集扫阈值，给出命中率与误命中率的取舍
# 样例分三类：完全同义、紧凑改写（同义但措辞差得多）、同实体不同问题
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

import retrieval_service  # noqa: E402

svc=retrieval_service.get_service()
svc._ensure_index()

# A 类：同义问法（语序/问法不同，意思相同）——命中是好事
PARAPHRASE=[
    ("什么是 BERT？","BERT 是什么？"),
    ("什么是自注意力机制？","自注意力机制是什么"),
    ("BERT 的输入最大长度是多少？","BERT 最多能输入多长"),
    ("Transformer 的架构是什么？","介绍一下 Transformer 的结构"),
    ("LoRA 是怎么微调的？","LoRA 的微调原理是什么"),
    ("RoFormer 的位置编码怎么做？","RoFormer 用的是什么位置编码"),
    ("FlashAttention 怎么加速？","FlashAttention 的加速原理"),
    ("GPT2 有多少参数？","GPT2 的参数量有多大"),
    ("思维链为什么有效？","Chain of Thought 有效的原因"),
    ("BERT 的预训练任务有哪两个？","BERT 预训练用了哪两个任务"),
    ("Transformer 有多少层？","Transformer 的层数是多少"),
    ("什么是有监督微调？","有监督微调是什么意思"),
    ("多头注意力是什么？","多头注意力机制的定义"),
    ("位置编码有什么用？","位置编码的作用是什么"),
    ("RAG 的流程是什么？","检索增强生成的流程"),
    ("预训练和微调有什么区别？","预训练与微调的不同"),
]

# B 类：同实体不同问题——命中就是答错（硬负例）
HARD_NEGATIVE=[
    ("什么是 BERT？","BERT 的预训练任务是什么？"),
    ("什么是 BERT？","BERT 的输入最大长度是多少？"),
    ("BERT 的预训练任务是什么？","BERT 的输入最大长度是多少？"),
    ("什么是自注意力机制？","多头注意力为什么要分头？"),
    ("自注意力机制是什么","注意力机制有什么局限"),
    ("Transformer 的架构是什么？","Transformer 的训练成本有多高"),
    ("Transformer 有多少层？","Transformer 的注意力头数是多少"),
    ("LoRA 是怎么微调的？","LoRA 相比全量微调省多少显存"),
    ("LoRA 是怎么微调的？","LoRA 的秩取多少合适"),
    ("RoFormer 的位置编码怎么做？","RoFormer 的编码怎么改善长文本"),
    ("FlashAttention 怎么加速？","FlashAttention 的显存占用降了多少"),
    ("GPT2 有多少参数？","GPT2 的模型结构是怎样的"),
    ("思维链为什么有效？","思维链有什么局限"),
    ("RAG 是什么？","RAG 相比微调有什么优势"),
    ("BERT 的预训练任务是什么？","BERT 和 GPT 的训练方式有什么区别"),
    ("什么是有监督微调？","有监督微调和指令微调一起用吗"),
    ("多头注意力是什么？","多头注意力的头数怎么选"),
    ("位置编码有什么用？","位置编码有哪几种实现"),
    ("预训练和微调有什么区别？","预训练要多少数据量"),
]


def scores(pairs):
    left=svc.model.encode([p[0] for p in pairs],normalize_embeddings=True)
    right=svc.model.encode([p[1] for p in pairs],normalize_embeddings=True)
    return [float(np.dot(l,r)) for l,r in zip(left,right)]


print("="*80)
print("A 类：同义问法（%d 对）" % len(PARAPHRASE))
print("="*80)
p=scores(PARAPHRASE)
for (a,b),s in sorted(zip(PARAPHRASE,p),key=lambda x:x[1]):
    print("  %.4f  %-26s | %s"%(s,a[:24],b[:30]))
print("  最低 %.4f | 25分位 %.4f | 中位 %.4f | 最高 %.4f"
      %(min(p),float(np.percentile(p,25)),float(np.median(p)),max(p)))

print()
print("="*80)
print("B 类：同实体不同问题（%d 对，命中即答错）" % len(HARD_NEGATIVE))
print("="*80)
n=scores(HARD_NEGATIVE)
for (a,b),s in sorted(zip(HARD_NEGATIVE,n),key=lambda x:-x[1])[:8]:
    print("  %.4f  %-26s | %s"%(s,a[:24],b[:30]))
print("  最高 %.4f | 中位 %.4f | 最低 %.4f"%(max(n),float(np.median(n)),min(n)))

print()
print("="*80)
print("阈值工作点")
print("="*80)
print("  %-8s %-12s %-12s %-10s %s"%("阈值","同义命中","异问误命中","命中绝对数","说明"))
for t in [0.80,0.84,0.86,0.88,0.90,0.92,0.94]:
    hit=sum(1 for s in p if s>=t)
    wrong=sum(1 for s in n if s>=t)
    note=""
    if wrong>0:
        note="有误命中风险"
    elif hit<=2:
        note="命中太少，缓存形同虚设"
    elif t<=0.86:
        note="命中多且无误命中"
    else:
        note="保守"
    print("  %-8.2f %-12s %-12s %-10s %s"
          %(t,"%d/%d"%(hit,len(p)),"%d/%d"%(wrong,len(n)),"%d 次"%hit,note))
