import asyncio
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
from web_search import hybrid_search

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

SIMPLE_SYSTEM_PROMPT="""你是一个论文问答助手。根据检索到的相关资料（本地论文/维基百科/arXiv）回答用户问题，回答要标注来源类型。
如果信息不足，明确说'根据现有资料无法回答'。"""

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

async def simple_context(question,k=5):
    #外网并发检索：返回统一sources与拼接context
    data=await hybrid_search(question,k)
    return data["sources"],data["context"]

async def simple_answer_async(question,k=5):
    #API异步入口：检索+LLM回答，LLM丢线程池避免阻塞事件循环
    sources,context=await simple_context(question,k)
    if not sources:
        return "未找到相关信息"
    messages=[
        {"role":"system","content":SIMPLE_SYSTEM_PROMPT},
        {"role":"user","content":f"以下是检索到的相关资料：\n{context}\n\n用户问题：{question}"}
    ]
    try:
        result=await asyncio.to_thread(safe_call_deepseek,messages,0.2,2000)
    except BudgetExceeded as e:
        return f"[预算保护] {e}"
    return result["choices"][0]["message"]["content"] or ""

def simple_answer(question,k=5,use_rewrite=False):
    #同步包装：仅供CLI/测试调用；API必须走simple_answer_async，避免asyncio.run跨循环
    return asyncio.run(simple_answer_async(question,k))

def survey_pipeline(question,top_n=3,human_confirm=False):
    #新链路：单ReAct Agent综述编排；旧链路保留为回退
    t0=time.time()
    if os.getenv("USE_NEW_SURVEY","1")=="1":
        from survey_agent import review_to_markdown, run_survey
        state=run_survey(question)
        t1=time.time()
        logger.info(f"新综述链路耗时{t1-t0:.1f}s papers={len(state.papers)} facts={len(state.facts)} conflicts={len(state.conflicts)}")
        return review_to_markdown(state.review) if state.review else "证据不足：未找到足够论文生成综述。"
    #旧链路（deprecated，仅回退用）
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
