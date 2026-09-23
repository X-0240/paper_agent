# 多轮追问的端到端回归：需要本地服务已启动（含模型与索引），未启动则跳过
# 这是 tests/test_dialog_anchor.py 的补充：那边测锚点逻辑，这里测真实检索是否命中
import json

import httpx
import pytest

BASE="http://127.0.0.1:8000"

# 第一轮问题、目标论文、代词型追问
CASES=[
    ("Attention Is All You Need 使用了什么机制？","Attention_Is_All_You_Need","它的核心组件是什么？"),
    ("BERT 的预训练任务有哪两个？","BERT","它的输入最大长度是多少？"),
    ("LoRA 的原理是什么？","LoRA","它训练时冻结了哪些参数？"),
    ("GPT2 的模型结构是怎样的？","GPT2","它的参数量有多大？"),
]


def _server_alive():
    try:
        return httpx.get(BASE+"/health",timeout=3).status_code==200
    except Exception:
        return False


pytestmark=pytest.mark.skipif(not _server_alive(),reason="本地服务未启动，跳过端到端多轮回归")


def _login():
    return httpx.post(BASE+"/login",json={"username":"admin","password":"admin123"},timeout=20).json()["access_token"]


def _turn(tok,question,thread_id=None):
    params={"question":question}
    if thread_id:
        params["thread_id"]=thread_id
    tid=None
    chips=[]
    with httpx.stream("GET",BASE+"/ask/stream",params=params,
                      headers={"Authorization":"Bearer "+tok},timeout=240) as r:
        buf=""
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                evt=json.loads(line[5:].strip())
            except Exception:
                continue
            if evt.get("type")=="thread":
                tid=evt["thread_id"]
            elif evt.get("type")=="sources":
                chips=[s.get("paper_id") or s.get("title") or "" for s in (evt.get("sources") or [])]
    return tid,chips


@pytest.mark.parametrize("first,target,follow",CASES)
def test_followup_retrieves_target_paper(first,target,follow):
    tok=_login()
    tid,chips1=_turn(tok,first)
    #第一轮必须命中，否则说明测试用例本身选的论文不在语料里
    assert any(target.lower() in (c or "").lower() for c in chips1),\
        f"第一轮未命中 {target}，检查语料是否包含该论文"
    _,chips2=_turn(tok,follow,thread_id=tid)
    assert any(target.lower() in (c or "").lower() for c in chips2),\
        f"追问未命中 {target}，实际引用 {chips2}"
