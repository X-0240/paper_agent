import os
import sqlite3
import tempfile
import threading

import pytest

import store


@pytest.fixture(autouse=True)
def temp_db(monkeypatch,tmp_path):
    monkeypatch.setenv("SQLITE_PATH",str(tmp_path/"agent.db"))
    store.close_conn()
    store.init_db()
    yield
    store.close_conn()


def test_init_db_idempotent():
    #重复初始化不应重复执行迁移
    assert store.init_db()==store.MAX_KNOWN_SCHEMA_VERSION
    assert store.init_db()==store.MAX_KNOWN_SCHEMA_VERSION


def test_thread_ownership_isolation():
    thread_id=store.create_thread("alice")
    assert store.get_thread(thread_id,"alice")["thread_id"]==thread_id
    #其他用户查不到，写入也拒绝
    assert store.get_thread(thread_id,"bob") is None
    assert store.add_message(thread_id,"bob","user","user_query","偷写") is None
    assert store.list_messages(thread_id,"bob") is None
    assert store.list_tasks(thread_id,"bob") is None


def test_list_threads_uses_first_question_as_title():
    #会话列表的标题取该会话第一条提问，供前端侧边栏展示
    thread_id=store.create_thread("alice")
    store.add_message(thread_id,"alice","user","user_query","什么是Transformer？")
    store.add_message(thread_id,"alice","assistant","assistant_answer","答案")
    store.add_message(thread_id,"alice","user","user_query","第二个问题")
    rows=store.list_threads("alice")
    assert len(rows)==1
    assert rows[0]["thread_id"]==thread_id
    assert rows[0]["title"]=="什么是Transformer？"


def test_list_threads_isolated_by_user():
    store.create_thread("alice")
    store.create_thread("bob")
    assert len(store.list_threads("alice"))==1
    assert len(store.list_threads("bob"))==1


def test_delete_thread_removes_messages_and_checks_ownership():
    thread_id=store.create_thread("alice")
    store.add_message(thread_id,"alice","user","user_query","问题")
    store.create_task(thread_id,"alice","simple")
    #别人的会话删不掉，也不该被误删
    assert store.delete_thread(thread_id,"bob") is False
    assert store.get_thread(thread_id,"alice") is not None
    assert store.delete_thread(thread_id,"alice") is True
    assert store.get_thread(thread_id,"alice") is None
    assert store.list_messages(thread_id,"alice") is None


def test_rename_thread_overrides_auto_title_without_reordering():
    thread_id=store.create_thread("alice")
    store.add_message(thread_id,"alice","user","user_query","原标题")
    before=store.list_threads("alice")[0]
    assert before["title"]=="原标题" and before["title_source"]=="auto"

    assert store.rename_thread(thread_id,"alice","自定义名字")=="自定义名字"
    after=store.list_threads("alice")[0]
    assert after["title"]=="自定义名字" and after["title_source"]=="custom"
    #改名不算新活动，不能把会话顶到列表最前
    assert after["updated_at"]==before["updated_at"]


def test_rename_thread_empty_title_falls_back_to_first_question():
    thread_id=store.create_thread("alice")
    store.add_message(thread_id,"alice","user","user_query","首条提问")
    store.rename_thread(thread_id,"alice","临时名字")
    assert store.list_threads("alice")[0]["title"]=="临时名字"
    #传空标题视为恢复自动命名
    store.rename_thread(thread_id,"alice","")
    row=store.list_threads("alice")[0]
    assert row["title"]=="首条提问" and row["title_source"]=="auto"


def test_rename_thread_checks_ownership():
    thread_id=store.create_thread("alice")
    assert store.rename_thread(thread_id,"bob","偷改") is None
    assert store.rename_thread("th_not_exist","alice","不存在") is None


def test_message_roundtrip_keeps_sources():
    thread_id=store.create_thread("alice")
    store.add_message(thread_id,"alice","user","user_query","什么是Transformer？")
    sources=[{"source_type":"local","title":"Attention Is All You Need","snippet":"..."}]
    store.add_message(thread_id,"alice","assistant","assistant_answer","答案",sources)
    rows=store.list_messages(thread_id,"alice")
    assert [r["message_type"] for r in rows]==["user_query","assistant_answer"]
    assert rows[1]["sources"][0]["title"]=="Attention Is All You Need"
    assert rows[0]["sources"]==[]


def test_recent_history_returns_tail_in_order():
    thread_id=store.create_thread("alice")
    for i in range(6):
        store.add_message(thread_id,"alice","user","user_query",f"问题{i}")
    history=store.recent_history(thread_id,"alice",turns=2)
    assert [h["content"] for h in history]==["问题2","问题3","问题4","问题5"]


def test_task_lifecycle_records_status_and_finish_time():
    thread_id=store.create_thread("alice")
    task_id=store.create_task(thread_id,"alice","survey")
    row=store.get_task(task_id,"alice")
    assert row["status"]=="running" and row["finished_at"] is None
    store.update_task(task_id,"failed","模型超时")
    row=store.get_task(task_id,"alice")
    assert row["status"]=="failed" and row["error_message"]=="模型超时" and row["finished_at"]
    assert store.get_task(task_id,"bob") is None
    assert len(store.list_tasks(thread_id,"alice"))==1


def test_newer_schema_refuses_to_start(monkeypatch,tmp_path):
    #库版本高于代码已知版本时必须拒绝启动，防止旧代码写新库
    path=tmp_path/"future.db"
    conn=sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version(version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version(version) VALUES(99)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("SQLITE_PATH",str(path))
    store.close_conn()
    with pytest.raises(RuntimeError):
        store.init_db()


def test_cost_and_tool_attribution_roundtrip():
    #成本与工具调用都要能挂到具体任务上，并按用户隔离
    thread_id=store.create_thread("alice")
    task_id=store.create_task(thread_id,"alice","survey")
    store.record_cost(task_id,thread_id,"alice","deepseek-v4-flash",100,50,150,0.0021)
    store.record_tool_call(task_id,thread_id,"alice","search_papers",'{"query":"x"}',True,1200,340)
    store.record_tool_call(task_id,thread_id,"alice","write_review","{}",False,5,20)
    attr=store.task_attribution(task_id,"alice")
    assert attr["llm_calls"]==1 and abs(attr["cost"]-0.0021)<1e-9
    assert attr["tool_calls_total"]==2 and attr["tool_calls_failed"]==1
    assert store.task_attribution(task_id,"bob") is None


def test_metrics_include_cost_and_tool_counts():
    thread_id=store.create_thread("alice")
    task_id=store.create_task(thread_id,"alice","simple")
    store.record_cost(task_id,thread_id,"alice","m",10,5,15,0.001)
    store.record_tool_call(task_id,thread_id,"alice","search_papers","{}",True,10,5)
    m=store.metrics_summary("alice",24)
    assert m["cost"]["total"]>0 and m["tool_calls"]["total"]==1
    assert m["cost"]["by_model"]["m"]>0


def test_concurrent_writes_do_not_lock():
    #多线程并发写：验证 thread-local 连接与 busy_timeout 下不会出现 database is locked
    thread_id=store.create_thread("alice")
    errors=[]

    def worker(idx):
        try:
            for i in range(10):
                store.add_message(thread_id,"alice","user","user_query",f"{idx}-{i}")
        except Exception as e:
            errors.append(repr(e))

    threads=[threading.Thread(target=worker,args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors==[]
    assert len(store.list_messages(thread_id,"alice"))==80
