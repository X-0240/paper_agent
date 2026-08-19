import logging
import os
import sys
import time
from agent1_retrieve import agent_1
from agent2_parse import agent_2
from agent3_review import agent_3
from rag_tool import search_papers_rerank, search_papers_structured
from llm_api import safe_call_deepseek, BudgetExceeded
from task_router import classify_task

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

SIMPLE_SYSTEM_PROMPT="""你是一个论文问答助手。根据提供的论文片段回答用户问题，回答要引用来源。
如果信息不足，明确说'根据现有论文无法回答'。"""

REWRITE_SYSTEM_PROMPT="""原样保留用户问题，然后在后面追加3-5个英文专业术语关键词，用空格分隔。只输出这一行。"""

def rewrite_query(question):
    #模糊问题改写为更具体查询，检索效果更稳；失败时回退原问题
    try:
        result=safe_call_deepseek(
            [
                {"role":"system","content":REWRITE_SYSTEM_PROMPT},
                {"role":"user","content":question}
            ],
            temperature=0.1,
            max_tokens=200
        )
        rewritten=result["choices"][0]["message"]["content"].strip()
        return rewritten if rewritten else question
    except Exception as e:
        logger.warning(f"查询改写失败，使用原问题：{e}")
        return question

def simple_answer(question,k=5,use_rewrite=False):
    #短路路径：检索+直接回答，不启动多Agent
    #改写默认关闭：跨语言实测93%→92%，待优化为"保留中文+扩展术语"再开
    query=rewrite_query(question) if use_rewrite else question
    #Rerank默认关闭：ms-marco偏英文，中文题实测-1%，换bge后设USE_RERANK=1
    if os.getenv("USE_RERANK")=="1":
        results=search_papers_rerank(query,k=k)
    else:
        results=search_papers_structured(query,k=k)
    if not results:
        return "未找到相关论文"
    context="\n\n".join(
        f"[来源:{r['source']} - {r['section']}]\n{r['text']}"
        for r in results
    )
    messages=[
        {"role":"system","content":SIMPLE_SYSTEM_PROMPT},
        {"role":"user","content":f"以下是相关的论文内容：\n{context}\n\n用户问题：{question}"}
    ]
    try:
        result=safe_call_deepseek(messages,temperature=0.2,max_tokens=2000)
    except BudgetExceeded as e:
        return f"[预算保护] {e}"
    return result["choices"][0]["message"]["content"] or ""

def survey_pipeline(question,top_n=3,human_confirm=False):
    #编排：Agent-1检索→Agent-2解析对比→Agent-3校验综述
    t0=time.time()
    try:
        papers=agent_1(question,top_n)
        t1=time.time()
        out2=agent_2(papers)
        t2=time.time()
        out3=agent_3(out2["papers"],out2["comparison"],human_confirm)
        t3=time.time()
        logger.info(f"耗时 Agent-1={t1-t0:.1f}s Agent-2={t2-t1:.1f}s Agent-3={t3-t2:.1f}s 总={t3-t0:.1f}s")
        return out3["review"]
    except BudgetExceeded as e:
        return f"[预算保护] {e}"

if __name__=="__main__":
    print("论文调研系统（Ctrl+C退出）")
    while True:
        q=input("问题：")
        if not q:
            continue
        task=classify_task(q)
        logger.info(f"路由判定：{task}")
        print(f"[路由] {task}")
        if task=="simple":
            print(simple_answer(q))
        else:
            print(survey_pipeline(q,human_confirm=True))
