import asyncio
import logging
import os
import sys
import time
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

MULTI_TURN_PROMPT="""把用户的追问补全成一个自足的问题，用于论文检索。只输出补全后的问题，不要解释。
规则：保留追问的原始意图；从对话历史里补全被指代的论文、方法或概念；历史里无关的话题不要带入。"""


def build_retrieval_question(question,history):
    #多轮追问补全：仅在开关打开且存在历史时调用，失败回退原问题
    if not history:
        return question
    lines=[]
    for item in history:
        content=(item.get("content") or "").replace("\n"," ")[:500]
        lines.append(f"{item.get('role','user')}: {content}")
    prompt="\n".join(lines)+f"\nuser: {question}"
    try:
        result=safe_call_deepseek(
            [{"role":"system","content":MULTI_TURN_PROMPT},{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=200
        )
        got=(result["choices"][0]["message"]["content"] or "").strip()
        return got if got else question
    except Exception as e:
        logger.warning(f"多轮改写失败，使用原问题：{e}")
        return question


async def simple_context(question,k=5,retrieval_question=None):
    #外网并发检索：检索用retrieval_question（多轮时是补全后的问题），回答仍用原问题
    data=await hybrid_search(retrieval_question or question,k)
    return data["sources"],data["context"]


async def simple_answer_with_sources_async(question,k=5,retrieval_question=None):
    #返回答案与来源，供接口层落库；simple_answer_async 委托到这里，保持单一实现
    sources,context=await simple_context(question,k,retrieval_question)
    if not sources:
        return {"answer":"未找到相关信息","sources":[]}
    messages=[
        {"role":"system","content":SIMPLE_SYSTEM_PROMPT},
        {"role":"user","content":f"以下是检索到的相关资料：\n{context}\n\n用户问题：{question}"}
    ]
    try:
        result=await asyncio.to_thread(safe_call_deepseek,messages,0.2,2000)
    except BudgetExceeded as e:
        return {"answer":f"[预算保护] {e}","sources":sources}
    return {"answer":result["choices"][0]["message"]["content"] or "","sources":sources}


async def simple_answer_async(question,k=5):
    #API异步入口：检索+LLM回答，LLM丢线程池避免阻塞事件循环
    data=await simple_answer_with_sources_async(question,k)
    return data["answer"]

def simple_answer(question,k=5,use_rewrite=False):
    #同步包装：仅供CLI/测试调用；API必须走simple_answer_async，避免asyncio.run跨循环
    return asyncio.run(simple_answer_async(question,k))

def survey_pipeline(question,top_n=3,human_confirm=False,on_progress=None):
    #新链路：单ReAct Agent综述编排；旧链路保留为回退
    t0=time.time()
    if os.getenv("USE_NEW_SURVEY","1")=="1":
        from survey_agent import review_to_markdown, run_survey
        state=run_survey(question,on_progress=on_progress)
        t1=time.time()
        logger.info(f"新综述链路耗时{t1-t0:.1f}s papers={len(state.papers)} facts={len(state.facts)} conflicts={len(state.conflicts)}")
        return review_to_markdown(state.review) if state.review else "证据不足：未找到足够论文生成综述。"
    #旧链路（deprecated，仅回退用）
    #注意：旧的 3-Agent 模块改成延迟导入，新链路启用时不再加载它们，避免多余依赖与启动开销
    try:
        from agent1_retrieve import agent_1
        from agent2_parse import agent_2
        from agent3_review import agent_3
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
