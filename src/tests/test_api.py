import json
import numpy as np
from fastapi.testclient import TestClient
from api_server import app

client=TestClient(app)

def login():
    r=client.post("/login",json={"username":"admin","password":"admin123"})
    assert r.status_code==200
    return r.json()["access_token"]

def auth_header(token):
    return {"Authorization":f"Bearer {token}"}

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
    monkeypatch.setattr("api_server.simple_answer",lambda q,**kw:"mock answer")
    r=client.post("/ask",json={"question":"什么是Transformer？"},headers=auth_header(login()))
    assert r.status_code==200
    assert r.json()["route"]=="simple"
    assert r.json()["answer"]=="mock answer"

def test_ask_forced_survey(monkeypatch):
    monkeypatch.setattr("api_server.survey_pipeline",lambda q,**kw:"mock review")
    r=client.post("/ask",json={"question":"对比Transformer和BERT","use_survey":True},headers=auth_header(login()))
    assert r.status_code==200
    assert r.json()["route"]=="survey"
    assert r.json()["answer"]=="mock review"

def test_stream_simple(monkeypatch):
    monkeypatch.setattr("api_server.search_papers_structured",lambda q,**kw:[{"source":"s","section":"sec","text":"t"}])
    monkeypatch.setattr("api_server.call_deepseek_stream",lambda messages:iter(["你","好"]))
    r=client.get("/ask/stream",params={"question":"什么是Transformer？"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert [e["type"] for e in events]==["route","sources","token","token","done"]
    assert events[1]["sources"][0]["source"]=="s"

def test_stream_error_event(monkeypatch):
    monkeypatch.setattr("api_server.search_papers_structured",lambda q,**kw:[{"source":"s","section":"sec","text":"t"}])
    def boom(messages):
        raise RuntimeError("mock boom")
    monkeypatch.setattr("api_server.call_deepseek_stream",boom)
    r=client.get("/ask/stream",params={"question":"什么是Transformer？"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert any(e["type"]=="error" for e in events)
    assert events[-1]["type"]=="done"

def test_stream_simple_float32_score(monkeypatch):
    #FAISS分数是numpy.float32，必须能过JSON序列化，否则sources事件报错
    monkeypatch.setattr("api_server.search_papers_structured",lambda q,**kw:[{"source":"s","section":"sec","text":"t","score":np.float32(0.5)}])
    monkeypatch.setattr("api_server.call_deepseek_stream",lambda messages:iter(["ok"]))
    r=client.get("/ask/stream",params={"question":"什么是Transformer？"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert not any(e["type"]=="error" for e in events)
    assert events[1]["type"]=="sources"

def test_stream_survey(monkeypatch):
    monkeypatch.setattr("api_server.survey_pipeline",lambda q,**kw:"A"*100)
    r=client.get("/ask/stream",params={"question":"对比Transformer和BERT"},headers=auth_header(login()))
    events=[json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert events[0]["type"]=="route" and events[0]["route"]=="survey"
    assert events[1]["type"]=="status"
    tokens=[e for e in events if e["type"]=="token"]
    assert sum(len(t["content"]) for t in tokens)==100
    assert events[-1]["type"]=="done"
