import asyncio
import json
import os
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
from llm_api import call_deepseek_stream
from pipeline import simple_answer_async, simple_context, survey_pipeline, SIMPLE_SYSTEM_PROMPT
from task_router import classify_task, rate_limit_allowed

load_dotenv()
app=FastAPI(title="论文调研Agent API")
bearer=HTTPBearer(auto_error=False)
WEB_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"web")

SECRET_KEY=os.getenv("SECRET_KEY","dev-secret")
ALGORITHM="HS256"
API_USERNAME=os.getenv("API_USERNAME","admin")
API_PASSWORD=os.getenv("API_PASSWORD","admin123")
TOKEN_EXPIRE_MINUTES=int(os.getenv("TOKEN_EXPIRE_MINUTES","1440"))

class LoginRequest(BaseModel):
    username: str
    password: str

class AskRequest(BaseModel):
    question: str
    use_survey: Optional[bool]=None

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
    task=classify_task(req.question) if req.use_survey is None else ("survey" if req.use_survey else "simple")
    if task=="survey":
        answer=await asyncio.to_thread(survey_pipeline,req.question,human_confirm=False)
    else:
        answer=await simple_answer_async(req.question)
    return {"route":task,"answer":answer,"user":user}

def sse_event(obj):
    #numpy.float32不能直接JSON序列化，统一转float兜底
    return f"data: {json.dumps(obj,ensure_ascii=False,default=lambda o: float(o) if isinstance(o,np.float32) else str(o))}\n\n"

@app.get("/ask/stream")
def ask_stream(question: str, user: str=Depends(verify_token)):
    if not rate_limit_allowed(user):
        raise HTTPException(status_code=429,detail="请求过于频繁，请稍后再试")
    async def gen():
        #流式异常必须转成SSE事件，不能让连接无提示中断
        try:
            task=classify_task(question)
            yield sse_event({"type":"route","route":task})
            if task=="simple":
                sources,context=await simple_context(question)
                yield sse_event({"type":"sources","sources":sources})
                messages=[
                    {"role":"system","content":SIMPLE_SYSTEM_PROMPT},
                    {"role":"user","content":f"以下是检索到的相关资料：\n{context}\n\n用户问题：{question}"}
                ]
                for token in call_deepseek_stream(messages):
                    yield sse_event({"type":"token","content":token})
            else:
                yield sse_event({"type":"status","content":"正在生成综述（约 1-3 分钟）"})
                answer=await asyncio.to_thread(survey_pipeline,question,human_confirm=False)
                for i in range(0,len(answer),40):
                    yield sse_event({"type":"token","content":answer[i:i+40]})
        except Exception as e:
            yield sse_event({"type":"error","message":str(e)})
        yield sse_event({"type":"done"})
    return StreamingResponse(gen(),media_type="text/event-stream")
