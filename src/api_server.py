import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
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
from paper_entities import extract_named_papers
from llm_api import call_deepseek_stream, call_deepseek_stream_async
from pipeline import (build_retrieval_question,simple_answer_with_sources_async,simple_context,
                      survey_pipeline,SIMPLE_SYSTEM_PROMPT)
from task_router import classify_task, rate_limit_allowed
import logging

load_dotenv()
logger=logging.getLogger(__name__)

#启动阶段预热：模型和索引改为服务起来就加载，避免第一次请求等二十多秒
async def _preheat():
    try:
        import retrieval_service
        service=retrieval_service.get_service()
        service._ensure_index()
        service._ensure_bm25()
        logger.info("预热完成：Embedding 模型与 FAISS 索引已就绪")
    except Exception as e:
        #预热失败不阻断启动，真正的报错留给请求链路暴露
        logger.warning(f"预热失败，改为首次请求时懒加载：{e}")
    try:
        from rerank import get_reranker
        get_reranker()
        logger.info("预热完成：重排模型已就绪")
    except Exception as e:
        logger.warning(f"重排模型预热失败：{e}")


@asynccontextmanager
async def lifespan(app):
    await _preheat()
    yield


app=FastAPI(title="论文调研Agent API",lifespan=lifespan)
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


def _recent_messages(thread_id,user,turns=3):
    #最近若干轮问答，按时间正序；取不到就返回空列表
    return store.recent_history(thread_id,user,turns) or []


def _named_papers_in_dialog(messages):
    #取上一轮的论文名作为追问的检索锚点。
    #只从用户提问里取：助手回答末尾带"来源类型：本地论文 [local:XXX]"这类引用标注，
    #扫回答会把引用过的所有论文都当成用户提到的对象（实测因此注入 4 个锚点、把正确论文挤掉）
    if not messages:
        return []
    ordered=[]
    for item in messages:
        if item.get("role")!="user":
            continue
        text=(item.get("content") or "").strip()
        if not text:
            continue
        try:
            found=extract_named_papers(text)
        except Exception:
            found=[]
        for name in found:
            if name not in ordered:
                ordered.append(name)
    #锚点过多会把检索查询稀释掉，只保留最近的几篇
    return ordered[::-1][:3][::-1]


def _resolve_retrieval_question(question,messages):
    #追问里没有论文名时，把上一轮点到的论文名补进检索查询。
    #实测依据：第一轮明确点名时命中 7/7，换成代词追问只剩 2/7，缺的就是实体名
    if not messages:
        return question
    try:
        if extract_named_papers(question):
            return question
    except Exception:
        return question
    anchors=_named_papers_in_dialog(messages)
    if not anchors:
        return question
    return " ".join(anchors)+" "+question


def _dialog_history_enabled():
    #多轮上下文开关：默认开。关掉后退回"只带当前问题"的老行为，便于做同题对照
    return os.getenv("DIALOG_HISTORY_ENABLED","1")=="1"


#零宽字符与纯标点等"没有语义内容"的输入，挡在调用模型之前
_MEANINGLESS=re.compile(r"[\s\W_]+",re.UNICODE)


def _question_reject_reason(question):
    #返回拒绝原因；返回空表示通过。空输入会白烧一次 LLM 调用，所以在入口挡掉
    text=(question or "").strip()
    if not text:
        return "问题不能为空"
    if not _MEANINGLESS.sub("",text):
        #只剩空白、标点、符号，没有可检索的语义
        return "请输入具体问题，当前内容只有标点或空白"
    return ""


def _recent_dialog(thread_id,user,turns=3,max_chars=600):
    #取最近若干轮问答拼进生成提示词。
    #必须在写入本轮提问之前调用，否则历史里会混进当前问题。
    #单条截断防止长会话把上下文撑爆；取不到历史就返回空串，不影响首轮问答
    if not _dialog_history_enabled():
        return ""
    rows=store.recent_history(thread_id,user,turns) or []
    return _format_dialog(rows,max_chars)


def _format_dialog(rows,max_chars=600):
    #把消息列表拼成提示词里的对话段；单条截断防止长会话撑爆上下文
    lines=[]
    for item in rows:
        content=(item.get("content") or "").strip().replace("\n"," ")
        if not content:
            continue
        who="用户" if item.get("role")=="user" else "助手"
        lines.append(f"{who}：{content[:max_chars]}")
    return "\n".join(lines)


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
    #前端不发缓存指令会被浏览器按启发式规则缓存，改完代码刷新仍是旧页面
    return FileResponse(os.path.join(WEB_DIR,"index.html"),
                        headers={"Cache-Control":"no-cache, must-revalidate"})

class _NoCacheStatic(StaticFiles):
    #静态资源强制每次向服务端校验，避免改了 app.js 但浏览器还在跑旧版本
    def file_response(self,*args,**kwargs):
        resp=super().file_response(*args,**kwargs)
        resp.headers["Cache-Control"]="no-cache, must-revalidate"
        return resp


app.mount("/static",_NoCacheStatic(directory=WEB_DIR),name="static")

@app.post("/login")
def login(req: LoginRequest):
    if req.username!=API_USERNAME or req.password!=API_PASSWORD:
        raise HTTPException(status_code=401,detail="用户名或密码错误")
    return {"access_token":create_token(req.username),"token_type":"bearer"}

@app.post("/ask")
async def ask(req: AskRequest, user: str=Depends(verify_token)):
    if not rate_limit_allowed(user):
        raise HTTPException(status_code=429,detail="请求过于频繁，请稍后再试")
    reason=_question_reject_reason(req.question)
    if reason:
        raise HTTPException(status_code=400,detail=reason)
    thread_id=await asyncio.to_thread(_ensure_thread,req.thread_id,user)
    task=classify_task(req.question) if req.use_survey is None else ("survey" if req.use_survey else "simple")
    history=await asyncio.to_thread(_recent_messages,thread_id,user)
    search_question=await asyncio.to_thread(_retrieval_question,req.question,thread_id,user)
    search_question=_resolve_retrieval_question(search_question,history)
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


@app.get("/threads")
def list_threads(user: str=Depends(verify_token)):
    #会话列表：只返回当前用户自己的会话，按最近更新排序
    return {"threads":store.list_threads(user)}


@app.delete("/threads/{thread_id}")
def remove_thread(thread_id: str,user: str=Depends(verify_token)):
    if not store.delete_thread(thread_id,user):
        raise HTTPException(status_code=404,detail="会话不存在")
    return {"deleted":thread_id}


class RenameRequest(BaseModel):
    title: str=""


@app.patch("/threads/{thread_id}")
def rename_thread(thread_id: str,req: RenameRequest,user: str=Depends(verify_token)):
    #标题留空表示恢复为按首条提问自动命名
    title=store.rename_thread(thread_id,user,req.title)
    if title is None:
        raise HTTPException(status_code=404,detail="会话不存在")
    return {"thread_id":thread_id,"title":title}


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
async def ask_stream(question: str, thread_id: Optional[str]=None, user: str=Depends(verify_token)):
    if not rate_limit_allowed(user):
        raise HTTPException(status_code=429,detail="请求过于频繁，请稍后再试")
    reason=_question_reject_reason(question)
    if reason:
        raise HTTPException(status_code=400,detail=reason)
    #同步路由跑在线程池里，这里直接落库不阻塞事件循环；会话不存在时能在响应开始前返回404
    thread= _ensure_thread(thread_id,user)
    task=classify_task(question)
    #取最近几轮对话：必须在本轮提问落库之前取，否则会把当前问题当成历史
    history=await asyncio.to_thread(_recent_messages,thread,user)
    search_question=await asyncio.to_thread(_retrieval_question,question,thread,user)
    #追问没点名论文时，把上一轮的论文名补进检索查询，否则代词原样去搜会捞回无关论文
    search_question=_resolve_retrieval_question(search_question,history)
    if os.getenv("DIALOG_DEBUG","0")=="1":
        logger.info(f"[dialog] 原始问题={question!r} 历史条数={len(history)} 检索查询={search_question!r}")
    dialog=_format_dialog(history) if _dialog_history_enabled() else ""
    #落库放到线程池，避免同步写库卡住事件循环
    await asyncio.to_thread(store.add_message,thread,user,"user","user_query",question,None,
                            search_question if search_question!=question else None)
    task_id=await asyncio.to_thread(store.create_task,thread,user,task)

    async def gen():
        #流式异常必须转成SSE事件，不能让连接无提示中断
        answer=[]
        debug_stream=os.getenv("STREAM_DEBUG","0")=="1"
        _t0=time.time()
        def _mark(tag,extra=""):
            #诊断用：记录事件被生产出来的时刻，定位是生产慢还是发送被缓冲
            if debug_stream:
                logger.info(f"[stream] {tag} +{time.time()-_t0:.2f}s {extra}")
        #上下文在生成器里设置：同步路由里设的 contextvar 不会带进事件循环任务
        ctx_token=run_context.set_run_context(task_id,thread,user)
        try:
            yield sse_event({"type":"thread","thread_id":thread,"task_id":task_id})
            _mark("thread")
            yield sse_event({"type":"route","route":task})
            _mark("route")
            if task=="simple":
                _mark("simple_start")
                _t_ctx=time.time()
                sources,context=await simple_context(question,5,search_question)
                _mark("sources_ready",f"耗时 {time.time()-_t_ctx:.2f}s")
                yield sse_event({"type":"sources","sources":sources})
                #有上文时先给对话历史再给资料：多轮里的代词指代靠这段解决
                user_block=f"以下是检索到的相关资料：\n{context}\n\n用户问题：{question}"
                if dialog:
                    user_block=f"之前的对话：\n{dialog}\n\n{user_block}\n\n注意：用户问题里如果出现代词（它/他/这个/那个）或省略主语，指代的是上一轮讨论的对象，请结合上面的对话判断，不要回答成别的论文或方法。"
                messages=[
                    {"role":"system","content":SIMPLE_SYSTEM_PROMPT},
                    {"role":"user","content":user_block}
                ]
                _mark("llm_start")
                _t_llm=time.time()
                #用 httpx 异步流式：同步 requests 会把线程池里的迭代变成整批返回
                async for token in call_deepseek_stream_async(messages):
                    answer.append(token)
                    if len(answer)<4 or len(answer)%50==0:
                        _mark("llm_token",f"index={len(answer)} 距开始 {time.time()-_t_llm:.2f}s")
                    yield sse_event({"type":"token","content":token})
                _mark("llm_end",f"共 {len(answer)} token，耗时 {time.time()-_t_llm:.2f}s")
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
                last_progress_at=time.time()
                while not worker.done() or sent<len(progress_items):
                    try:
                        await asyncio.wait_for(queue.get(),timeout=heartbeat)
                    except asyncio.TimeoutError:
                        yield sse_event({"type":"ping"})
                    while sent<len(progress_items):
                        yield sse_event({"type":"progress","content":progress_items[sent]})
                        sent+=1
                        last_progress_at=time.time()
                    #长时间只有心跳时补一条带等待时长的进度，避免界面看着像卡住
                    waited=time.time()-last_progress_at
                    if waited>=heartbeat and not worker.done():
                        yield sse_event({"type":"progress",
                                         "content":f"仍在处理｜距上一步已 {int(waited)} 秒"})
                        last_progress_at=time.time()
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
