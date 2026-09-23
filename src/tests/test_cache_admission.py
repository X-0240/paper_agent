# 语义召回缓存与准入门的纯逻辑测试：不加载模型，可进 CI
import asyncio

import numpy as np
import pytest

import api_server
import retrieval_service


def _vec(*values):
    #构造归一化向量，避免测试依赖模型
    v=np.asarray(values,dtype="float32")
    return v/np.linalg.norm(v)


def test_semantic_cache_exact_hit_and_stats():
    cache=retrieval_service._SemanticRecallCache(threshold=0.85)
    cache.store("什么是 BERT？",_vec(1.0,0.0),"snap1",["c1","c2","c3"])
    ids,score=cache.lookup("什么是 BERT？",_vec(1.0,0.0),"snap1")
    assert ids==["c1","c2","c3"]
    assert score==1.0
    st=cache.stats()
    assert st["hit"]==1 and st["miss"]==0


def test_semantic_cache_paraphrase_hit_above_threshold():
    cache=retrieval_service._SemanticRecallCache(threshold=0.85)
    cache.store("什么是 BERT？",_vec(1.0,0.0),"snap1",["c1"])
    #同义问法：向量接近，应该命中
    ids,_=cache.lookup("BERT 是什么？",_vec(0.99,0.1),"snap1")
    assert ids==["c1"]
    assert cache.stats()["hit"]==1


def test_semantic_cache_rejects_hard_negative_below_threshold():
    #同实体不同问题：实测最高 0.7844，阈值 0.85 必须拒绝，否则会给出错误证据
    cache=retrieval_service._SemanticRecallCache(threshold=0.85)
    cache.store("什么是 BERT？",_vec(1.0,0.0),"snap1",["c1"])
    ids,score=cache.lookup("BERT 的输入最大长度是多少？",_vec(0.75,0.66),"snap1")
    assert ids is None
    assert score<0.85
    assert cache.stats()["miss"]==1


def test_semantic_cache_snapshot_change_invalidates():
    #索引快照变了（重建索引）就不能复用旧 chunk
    cache=retrieval_service._SemanticRecallCache(threshold=0.85)
    cache.store("什么是 BERT？",_vec(1.0,0.0),"snap1",["c1"])
    ids,_=cache.lookup("什么是 BERT？",_vec(1.0,0.0),"snap2")
    assert ids is None


def test_semantic_cache_ignores_whitespace_difference():
    #只差空格/大小写的问法应该走精确命中，不必做向量比对
    cache=retrieval_service._SemanticRecallCache(threshold=0.85)
    cache.store("What  is BERT?",_vec(1.0,0.0),"snap1",["c1"])
    ids,score=cache.lookup("what is bert?",_vec(0.2,0.98),"snap1")
    assert ids==["c1"] and score==1.0


def test_semantic_cache_capacity_limit():
    cache=retrieval_service._SemanticRecallCache(threshold=0.85,max_size=3)
    for i in range(5):
        cache.store(f"问题{i}",_vec(1.0,0.01*i),"snap1",[f"c{i}"])
    assert cache.stats()["size"]==3


def test_admission_accepts_until_full_then_rejects():
    gate=api_server._AdmissionGate(max_inflight=2,wait_budget=60,avg_seconds=17)

    async def run():
        ok1,_=await gate.acquire()
        ok2,_=await gate.acquire()
        ok3,why=await gate.acquire()
        return ok1,ok2,ok3,why

    ok1,ok2,ok3,why=asyncio.run(run())
    assert ok1 and ok2
    assert not ok3
    assert "并发已达上限" in why
    assert gate.stats()["rejected_busy"]==1


def test_admission_rejects_when_estimated_wait_exceeds_budget():
    #等待预算写成"用户能接受的最长排队时长"，不是"预计等待"；
    #写成后者会让 inflight>=2 就拒，白白浪费名额（实测踩过）
    gate=api_server._AdmissionGate(max_inflight=8,wait_budget=60,avg_seconds=17)

    async def run():
        for _ in range(3):
            await gate.acquire()
        #3 个在跑 -> 预计等待 51s <= 60s，应该还接收
        ok4,_=await gate.acquire()
        #4 个在跑 -> 预计等待 68s > 60s，应该拒绝
        ok5,why=await gate.acquire()
        return ok4,ok5,why

    ok4,ok5,why=asyncio.run(run())
    assert ok4 is True
    assert ok5 is False
    assert "预计等待" in why


def test_admission_release_frees_slot():
    gate=api_server._AdmissionGate(max_inflight=1,wait_budget=60,avg_seconds=17)

    async def run():
        ok1,_=await gate.acquire()
        ok2,why=await gate.acquire()
        await gate.release()
        ok3,_=await gate.acquire()
        return ok1,ok2,ok3,gate.stats()

    ok1,ok2,ok3,st=asyncio.run(run())
    assert ok1 and not ok2 and ok3
    assert st["inflight"]==1 and st["peak"]==1
