import json
import unittest.mock as mock
from datetime import datetime, timedelta, timezone
import jwt
import numpy as np
from fastapi.testclient import TestClient
import api_server
from api_server import app

client=TestClient(app)

def login():
    r=client.post("/login",json={"username":"admin","password":"admin123"})
    assert r.status_code==200
    return r.json()["access_token"]

def auth_header(token):
    return {"Authorization":f"Bearer {token}"}


def token_for(username):
    #接口只有一个登录账号，这里直接签发其他用户的Token验证归属隔离
    exp=datetime.now(timezone.utc)+timedelta(minutes=5)
    return jwt.encode({"sub":username,"exp":exp},api_server.SECRET_KEY,algorithm="HS256")


def test_metrics_endpoint_reports_own_data(monkeypatch):
    monkeypatch.setattr("api_server.simple_answer_with_sources_async",
                        mock.AsyncMock(return_value={"answer":"a","sources":[]}))
    token=login()
    assert client.post("/ask",json={"question":"什么是Transformer？"},headers=auth_header(token)).status_code==200
    data=client.get("/metrics",headers=auth_header(token)).json()
    assert data["tasks"]["total"]>=1 and data["tasks"]["succeeded"]>=1
    assert data["messages"]["by_type"]["user_query"]>=1
    assert "uptime_s" in data["process"]
    #指标同样按用户隔离
    other=client.get("/metrics",headers=auth_header(token_for("bob"))).json()
    assert other["tasks"]["total"]==0


def test_thread_persistence_and_ownership(monkeypatch):
    monkeypatch.setattr("api_server.simple_answer_with_sources_async",
                        mock.AsyncMock(return_value={"answer":"mock answer","sources":[]}))
    token=login()
    r=client.post("/ask",json={"question":"什么是Transformer？"},headers=auth_header(token))
    thread_id=r.json()["thread_id"]
    task_id=r.json()["task_id"]
    msgs=client.get(f"/threads/{thread_id}/messages",headers=auth_header(token))
    assert msgs.status_code==200
    assert [m["message_type"] for m in msgs.json()["messages"]]==["user_query","assistant_answer"]
    tasks=client.get(f"/threads/{thread_id}/tasks",headers=auth_header(token))
    assert tasks.status_code==200
    assert tasks.json()["tasks"][0]["status"]=="succeeded"
    assert client.get(f"/tasks/{task_id}",headers=auth_header(token)).status_code==200
    #其他用户即使拿到thread_id也不能读，统一404不泄露存在性
    other=auth_header(token_for("bob"))
    assert client.get(f"/threads/{thread_id}/messages",headers=other).status_code==404
    assert client.get(f"/threads/{thread_id}/tasks",headers=other).status_code==404
    assert client.get(f"/tasks/{task_id}",headers=other).status_code==404
    assert client.post("/ask",json={"question":"x","thread_id":thread_id},headers=other).status_code==404

def test_health():
    assert client.get("/health").json()=={"status":"ok"}

def test_index_served():
    r=client.get("/")
    assert r.status_code==200
    assert "text/html" in r.headers["content-type"]

def test_login_wrong_password():
    assert client.post("/login",json={"username":"admin","password":"x"}).status_code==401

def test_ask_requires_token():
    assert client.post("/ask",json={"question":"x"}).status_code==401

def test_ask_simple(monkeypatch):
    monkeypatch.setattr("api_server.simple_answer_with_sources_async",
                        mock.AsyncMock(return_value={"answer":"mock answer","sources":[]}))
    r=client.post("/ask",json={"question":"什么是Transformer？"},headers=auth_header(login()))
    assert r.status_code==200
    assert r.json()["route"]=="simple"
    assert r.json()["answer"]=="mock answer"
    assert r.json()["thread_id"].startswith("th_")
    assert r.json()["task_id"].startswith("tk_")

def test_ask_forced_survey(monkeypatch):
    monkeypatch.setattr("api_server.survey_pipeline",lambda q,**kw:"mock review")
    r=client.post("/ask",json={"question":"对比Transformer和BERT","use_survey":True},headers=auth_header(login()))
    assert r.status_code==200
    assert r.json()["route"]=="survey"
    assert r.json()["answer"]=="mock review"

async def _fake_stream(tokens):
    #接口层改用异步流式，测试同时提供异步生成器替身
    for token in tokens:
        yield token


def test_stream_simple(monkeypatch):
    sources=[{"source_type":"local","title":"s","snippet":"t","paper_id":"s","section":"sec","url":None,"arxiv_id":None}]
    monkeypatch.setattr("api_server.simple_context",mock.AsyncMock(return_value=(sources,"ctx")))
    monkeypatch.setattr("api_server.call_deepseek_stream_async",lambda messages:_fake_stream(["你","好"]))
    r=client.get("/ask/stream",params={"question":"什么是Transformer？"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert [e["type"] for e in events]==["thread","route","sources","token","token","done"]
    sources_event=[e for e in events if e["type"]=="sources"][0]
    assert sources_event["sources"][0]["title"]=="s"

def test_stream_error_event(monkeypatch):
    sources=[{"source_type":"local","title":"s","snippet":"t","paper_id":"s","section":"sec","url":None,"arxiv_id":None}]
    monkeypatch.setattr("api_server.simple_context",mock.AsyncMock(return_value=(sources,"ctx")))
    async def boom(tokens):
        raise RuntimeError("mock boom")
        yield  # pragma: no cover
    monkeypatch.setattr("api_server.call_deepseek_stream_async",lambda messages:boom(["x"]))
    r=client.get("/ask/stream",params={"question":"什么是Transformer？"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert any(e["type"]=="error" for e in events)
    assert events[-1]["type"]=="done"

def test_stream_simple_float32_score(monkeypatch):
    #来源里若混入numpy.float32，也必须能过JSON序列化
    sources=[{"source_type":"local","title":"s","snippet":"t","paper_id":"s","section":"sec","url":None,"arxiv_id":None,"score":np.float32(0.5)}]
    monkeypatch.setattr("api_server.simple_context",mock.AsyncMock(return_value=(sources,"ctx")))
    monkeypatch.setattr("api_server.call_deepseek_stream_async",lambda messages:_fake_stream(["ok"]))
    r=client.get("/ask/stream",params={"question":"什么是Transformer？"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert not any(e["type"]=="error" for e in events)
    assert [e for e in events if e["type"]=="sources"]

def test_stream_survey(monkeypatch):
    def fake_survey(q,**kw):
        #模拟编排层推两步进度
        cb=kw.get("on_progress")
        if cb:
            cb("检索候选论文｜论文3篇｜卡片0张｜事实0条｜冲突0件")
            cb("生成卡片与事实｜论文3篇｜卡片3张｜事实12条｜冲突2件")
        return "A"*100
    monkeypatch.setattr("api_server.survey_pipeline",fake_survey)
    r=client.get("/ask/stream",params={"question":"对比Transformer和BERT"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert [e for e in events if e["type"]=="route"][0]["route"]=="survey"
    assert [e for e in events if e["type"]=="status"]
    progress=[e for e in events if e["type"]=="progress"]
    assert len(progress)==2
    assert "论文3篇" in progress[0]["content"]
    assert events[-1]["type"]=="done"
    tokens=[e for e in events if e["type"]=="token"]
    assert sum(len(t["content"]) for t in tokens)==100
    assert events[-1]["type"]=="done"


def test_list_threads_and_delete_requires_ownership():
    #会话列表返回当前用户的会话，标题取第一条提问
    hdr=auth_header(login())
    r=client.get("/ask/stream",params={"question":"什么是Transformer？"},headers=hdr)
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    thread_id=[e for e in events if e["type"]=="thread"][0]["thread_id"]

    listed=client.get("/threads",headers=hdr)
    assert listed.status_code==200
    ids=[t["thread_id"] for t in listed.json()["threads"]]
    assert thread_id in ids
    target=[t for t in listed.json()["threads"] if t["thread_id"]==thread_id][0]
    assert target["title"]=="什么是Transformer？"

    #不存在的会话删除返回404，不会误删
    assert client.delete("/threads/th_not_exist",headers=hdr).status_code==404

    #改名：空标题回退自动命名，改名后列表标题跟着变
    renamed=client.patch("/threads/"+thread_id,json={"title":"我的专题"},headers=hdr)
    assert renamed.status_code==200 and renamed.json()["title"]=="我的专题"
    titles={t["thread_id"]:t["title"] for t in client.get("/threads",headers=hdr).json()["threads"]}
    assert titles[thread_id]=="我的专题"
    assert client.patch("/threads/th_not_exist",json={"title":"x"},headers=hdr).status_code==404

    assert client.delete("/threads/"+thread_id,headers=hdr).status_code==200
    assert client.get("/threads/"+thread_id+"/messages",headers=hdr).status_code==404
