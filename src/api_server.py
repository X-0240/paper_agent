import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import store
import run_context
from llm_api import call_deepseek_stream
from pipeline import (build_retrieval_question,simple_answer_with_sources_async,simple_context,
                      survey_pipeline,SIMPLE_SYSTEM_PROMPT)
from task_router import classify_task, rate_limit_allowed

load_dotenv()
app=FastAPI(title="论文调研Agent API")
bearer=HTTPBearer(auto_error=False)
WEB_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"web")

#导入即迁移：schema 版本异常时拒绝启动，接口不会在无表状态下对外服务
store.init_db()

SECRET_KEY=os.getenv("SECRET_KEY","dev-secret")
ALGORITHM="HS256"
API_USERNAME=os.getenv("API_USERNAME","admin")
API_PASSWORD=os.getenv("API_PASSWORD","admin123")
TOKEN_EXPIRE_MINUTES=int(os.getenv("TOKEN_EXPIRE_MINUTES","1440"))
START_TS=time.time()

class LoginRequest(BaseModel):
    username: str
    password: str

class AskRequest(BaseModel):
    question: str
    use_survey: Optional[bool]=None
    thread_id: Optional[str]=None


def _multi_turn_enabled():
    return os.getenv("MULTI_TURN_QUERY_REWRITE","0")=="1"


def _ensure_thread(thread_id,user):
    #未传thread_id就新建；传了但归属不符按不存在处理，避免泄露存在性
    if not thread_id:
        return store.create_thread(user)
    if store.get_thread(thread_id,user) is None:
        raise HTTPException(status_code=404,detail="会话不存在")
    return thread_id


def _retrieval_question(question,thread_id,user):
    #多轮改写默认关闭：关闭时检索输入与改造前完全一致
    if not _multi_turn_enabled():
        return question
    history=store.recent_history(thread_id,user,3) or []
    if not history:
        return question
    return build_retrieval_question(question,history)


def _sources_brief(sources):
    #只落库来源摘要，避免把整段原文写进数据库
    out=[]
    for item in sources or []:
        out.append({
            "source_type":item.get("source_type",""),
            "title":item.get("title",""),
            "snippet":(item.get("snippet") or "")[:200],
        })
    return out

def create_token(username):
    exp=datetime.now(timezone.utc)+timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub":username,"exp":exp},SECRET_KEY,algorithm=ALGORITHM)

def verify_token(cred: Optional[HTTPAuthorizationCredentials]=Depends(bearer)):
    #无凭证与无效凭证统一返回401，符合标准鉴权语义
    if cred is None:
        raise HTTPException(status_code=401,detail="缺少Token")
    try:
        payload=jwt.decode(cred.credentials,SECRET_KEY,algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401,detail="无效或过期Token")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR,"index.html"))

app.mount("/static",StaticFiles(directory=WEB_DIR),name="static")

@app.post("/login")
def login(req: LoginRequest):
    if req.username!=API_USERNAME or req.password!=API_PASSWORD:
        raise HTTPException(status_code=401,detail="用户名或密码错误")
    return {"access_token":create_token(req.username),"token_type":"bearer"}

@app.post("/ask")
async def ask(req: AskRequest, user: str=Depends(verify_token)):
    if not rate_limit_allowed(user):
        raise HTTPException(status_code=429,detail="请求过于频繁，请稍后再试")
    thread_id=await asyncio.to_thread(_ensure_thread,req.thread_id,user)
    task=classify_task(req.question) if req.use_survey is None else ("survey" if req.use_survey else "simple")
    search_question=await asyncio.to_thread(_retrieval_question,req.question,thread_id,user)
    await asyncio.to_thread(store.add_message,thread_id,user,"user","user_query",req.question,None,
                            search_question if search_question!=req.question else None)
    task_id=await asyncio.to_thread(store.create_task,thread_id,user,task)
    #任务上下文：让本次请求内的LLM用量与工具调用都归因到这个task_id
    ctx_token=run_context.set_run_context(task_id,thread_id,user)
    try:
        if task=="survey":
            try:
                answer=await asyncio.to_thread(survey_pipeline,req.question,human_confirm=False)
            except Exception as e:
                await asyncio.to_thread(store.update_task,task_id,"failed",str(e))
                await asyncio.to_thread(store.add_message,thread_id,user,"assistant","system_error",str(e))
                raise HTTPException(status_code=500,detail=str(e))
            await asyncio.to_thread(store.update_task,task_id,"succeeded")
            await asyncio.to_thread(store.add_message,thread_id,user,"assistant","task_result",answer)
            sources=[]
        else:
            try:
                data=await simple_answer_with_sources_async(req.question,retrieval_question=search_question)
            except Exception as e:
                await asyncio.to_thread(store.update_task,task_id,"failed",str(e))
                await asyncio.to_thread(store.add_message,thread_id,user,"assistant","system_error",str(e))
                raise HTTPException(status_code=500,detail=str(e))
            answer=data["answer"]
            sources=_sources_brief(data["sources"])
            await asyncio.to_thread(store.update_task,task_id,"succeeded")
            await asyncio.to_thread(store.add_message,thread_id,user,"assistant","assistant_answer",answer,sources)
    finally:
        run_context.reset_run_context(ctx_token)
    return {"route":task,"answer":answer,"user":user,"thread_id":thread_id,"task_id":task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str,user: str=Depends(verify_token)):
    row=store.get_task(task_id,user)
    if row is None:
        raise HTTPException(status_code=404,detail="任务不存在")
    #任务详情带上成本与工具调用归因，一次请求就能看清这个任务花了什么
    attribution=store.task_attribution(task_id,user)
    if attribution:
        row["attribution"]=attribution
    return row


@app.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str,limit: int=0,user: str=Depends(verify_token)):
    rows=store.list_messages(thread_id,user,limit or None)
    if rows is None:
        raise HTTPException(status_code=404,detail="会话不存在")
    return {"thread_id":thread_id,"messages":rows}


@app.get("/threads/{thread_id}/tasks")
def get_thread_tasks(thread_id: str,user: str=Depends(verify_token)):
    rows=store.list_tasks(thread_id,user)
    if rows is None:
        raise HTTPException(status_code=404,detail="会话不存在")
    return {"thread_id":thread_id,"tasks":rows}


@app.get("/metrics")
def metrics(since_hours: int=24,user: str=Depends(verify_token)):
    #运行时指标：只看当前用户自己的数据，外加进程级信息
    data=store.metrics_summary(user,since_hours)
    data["process"]={
        "uptime_s":round(time.time()-START_TS,1),
        "schema_version":store.MAX_KNOWN_SCHEMA_VERSION,
        "candidate_k":os.getenv("RETRIEVAL_CANDIDATES"),
        "rerank_mode":os.getenv("RERANK_MODE"),
        "query_generation":os.getenv("QUERY_GENERATION_ENABLED"),
        "entity_boost":os.getenv("ENTITY_BOOST"),
        "multi_turn_rewrite":os.getenv("MULTI_TURN_QUERY_REWRITE"),
    }
    return data

def sse_event(obj):
    #numpy.float32不能直接JSON序列化，统一转float兜底
    return f"data: {json.dumps(obj,ensure_ascii=False,default=lambda o: float(o) if isinstance(o,np.float32) else str(o))}\n\n"

@app.get("/ask/stream")
def ask_stream(question: str, thread_id: Optional[str]=None, user: str=Depends(verify_token)):
    if not rate_limit_allowed(user):
        raise HTTPException(status_code=429,detail="请求过于频繁，请稍后再试")
    #同步路由跑在线程池里，这里直接落库不阻塞事件循环；会话不存在时能在响应开始前返回404
    thread= _ensure_thread(thread_id,user)
    task=classify_task(question)
    search_question=_retrieval_question(question,thread,user)
    store.add_message(thread,user,"user","user_query",question,None,
                      search_question if search_question!=question else None)
    task_id=store.create_task(thread,user,task)
    async def gen():
        #流式异常必须转成SSE事件，不能让连接无提示中断
        answer=[]
        #上下文在生成器里设置：同步路由里设的 contextvar 不会带进事件循环任务
        ctx_token=run_context.set_run_context(task_id,thread,user)
        try:
            yield sse_event({"type":"thread","thread_id":thread,"task_id":task_id})
            yield sse_event({"type":"route","route":task})
            if task=="simple":
                sources,context=await simple_context(question,5,search_question)
                yield sse_event({"type":"sources","sources":sources})
                messages=[
                    {"role":"system","content":SIMPLE_SYSTEM_PROMPT},
                    {"role":"user","content":f"以下是检索到的相关资料：\n{context}\n\n用户问题：{question}"}
                ]
                for token in call_deepseek_stream(messages):
                    answer.append(token)
                    yield sse_event({"type":"token","content":token})
                await asyncio.to_thread(store.add_message,thread,user,"assistant","assistant_answer",
                                        "".join(answer),_sources_brief(sources))
            else:
                yield sse_event({"type":"status","content":"正在生成综述（约 1-3 分钟）"})
                #综述是多步任务，没有token流可推；把每步进度推成事件，并用心跳防中间层判超时
                loop=asyncio.get_running_loop()
                queue=asyncio.Queue()
                progress_items=[]
                def on_progress(text):
                    #回调在工作线程里触发：先追加到列表再唤醒事件循环，避免最后一条进度丢失
                    progress_items.append(text)
                    loop.call_soon_threadsafe(queue.put_nowait,None)
                heartbeat=float(os.getenv("SSE_HEARTBEAT_SECONDS","10"))
                worker=asyncio.create_task(asyncio.to_thread(
                    survey_pipeline,question,human_confirm=False,on_progress=on_progress))
                sent=0
                while not worker.done() or sent<len(progress_items):
                    try:
                        await asyncio.wait_for(queue.get(),timeout=heartbeat)
                    except asyncio.TimeoutError:
                        yield sse_event({"type":"ping"})
                    while sent<len(progress_items):
                        yield sse_event({"type":"progress","content":progress_items[sent]})
                        sent+=1
                text=await worker
                for i in range(0,len(text),40):
                    answer.append(text[i:i+40])
                    yield sse_event({"type":"token","content":text[i:i+40]})
                await asyncio.to_thread(store.add_message,thread,user,"assistant","task_result","".join(answer))
            await asyncio.to_thread(store.update_task,task_id,"succeeded")
        except Exception as e:
            await asyncio.to_thread(store.update_task,task_id,"failed",str(e))
            await asyncio.to_thread(store.add_message,thread,user,"assistant","system_error",str(e))
            yield sse_event({"type":"error","message":str(e)})
        finally:
            run_context.reset_run_context(ctx_token)
        yield sse_event({"type":"done"})
    return StreamingResponse(gen(),media_type="text/event-stream")
