# 验证语义缓存与准入的效果
# A：同义问法第二次是否变快（缓存命中）；B：同实体不同问题是否被拒绝（阈值是否安全）
# C：并发压测下的准入表现（是否快速拒绝而不是全挂）
import json
import time

import httpx

B="http://127.0.0.1:8000"


def tok():
    return httpx.post(B+"/login",json={"username":"admin","password":"admin123"},timeout=20).json()["access_token"]


def ask(t,question,timeout=240):
    #返回 (首字节秒, 总秒, 状态码, 引用列表)
    t0=time.time()
    first=None
    chips=[]
    try:
        with httpx.stream("GET",B+"/ask/stream",params={"question":question},
                          headers={"Authorization":"Bearer "+t},timeout=timeout) as r:
            if r.status_code!=200:
                return None,time.time()-t0,r.status_code,None
            buf=""
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                if first is None:
                    first=time.time()-t0
                try:
                    e=json.loads(line[5:].strip())
                except Exception:
                    continue
                if e.get("type")=="sources":
                    chips=[s.get("paper_id") for s in (e.get("sources") or [])]
    except Exception as ex:
        return first,time.time()-t0,type(ex).__name__,chips
    return first,time.time()-t0,200,chips


print("="*80)
print("A. 语义缓存：同义问法第二次是否复用检索结果")
print("="*80)
t=tok()
pairs=[
    ("什么是 BERT？","BERT 是什么？"),
    ("BERT 的输入最大长度是多少？","BERT 最多能输入多长"),
    ("Transformer 有多少层？","Transformer 的层数是多少"),
    ("LoRA 是怎么微调的？","LoRA 的微调原理是什么"),
]
for q1,q2 in pairs:
    f1,tot1,c1,ch1=ask(t,q1)
    f2,tot2,c2,ch2=ask(t,q2)
    same = ch1==ch2
    print("  第1问 %-24s 耗时 %5.1fs | 第2问 %-20s 耗时 %5.1fs | 引用一致: %s"
          %(q1[:22],tot1,q2[:18],tot2,same))

print()
print("="*80)
print("B. 阈值安全性：同实体不同问题是否被误判命中")
print("="*80)
neg=[
    ("什么是 BERT？","BERT 的输入最大长度是多少？"),
    ("Transformer 有多少层？","Transformer 的注意力头数是多少"),
    ("LoRA 是怎么微调的？","LoRA 的秩取多少合适"),
]
for q1,q2 in neg:
    f1,tot1,c1,ch1=ask(t,q1)
    f2,tot2,c2,ch2=ask(t,q2)
    same = ch1==ch2
    print("  %-22s vs %-24s | 引用一致: %s %s"
          %(q1[:20],q2[:22],same,"（可疑：不同问题不该同引用）" if same else ""))

print()
print("="*80)
print("C. 运行统计（缓存命中率与准入拒绝次数）")
print("="*80)
try:
    m=httpx.get(B+"/metrics",params={"since_hours":1},
                headers={"Authorization":"Bearer "+t},timeout=20).json()
    print("  语义缓存：",m.get("semantic_cache"))
    print("  准入统计：",m.get("admission"))
except Exception as e:
    print("  读取失败：",e)
