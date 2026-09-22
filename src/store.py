import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime,timedelta,timezone

BASE=os.path.dirname(os.path.abspath(__file__))
MIGRATIONS_DIR=os.path.join(BASE,"migrations")
#代码已知的最高迁移编号：硬编码并在启动时与目录核对，避免目录缺失时校验失效
MAX_KNOWN_SCHEMA_VERSION=3

#每线程一个连接：sqlite3连接不能跨线程共享，线程池下同线程复用同一连接
_local=threading.local()


def db_path():
    return os.getenv("SQLITE_PATH") or os.path.join(BASE,"data","agent.db")


def get_conn():
    path=db_path()
    conn=getattr(_local,"conn",None)
    if conn is not None and getattr(_local,"conn_path","")==path:
        return conn
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    conn=sqlite3.connect(path,timeout=5.0)
    conn.row_factory=sqlite3.Row
    #显式声明锁等待，防止后续有人调整 connect 参数时静默改变行为
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn=conn
    _local.conn_path=path
    return conn


def close_conn():
    #测试与进程退出用：释放当前线程连接
    conn=getattr(_local,"conn",None)
    if conn is not None:
        conn.close()
        _local.conn=None
        _local.conn_path=""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _migration_files():
    if not os.path.isdir(MIGRATIONS_DIR):
        return []
    files=[]
    for name in sorted(os.listdir(MIGRATIONS_DIR)):
        if not name.endswith(".sql"):
            continue
        files.append((int(name.split("_",1)[0]),os.path.join(MIGRATIONS_DIR,name)))
    return sorted(files)


def _current_version(conn):
    row=conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone()
    if row is None:
        return 0
    got=conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(got["v"]) if got and got["v"] is not None else 0


def init_db():
    #迁移入口：先拒绝比代码更新的库，再按编号升序逐个执行，DDL与版本号同事务
    files=_migration_files()
    highest=files[-1][0] if files else 0
    if highest!=MAX_KNOWN_SCHEMA_VERSION:
        raise RuntimeError(f"迁移目录最高编号 {highest} 与代码已知版本 {MAX_KNOWN_SCHEMA_VERSION} 不一致")
    conn=get_conn()
    current=_current_version(conn)
    if current>MAX_KNOWN_SCHEMA_VERSION:
        raise RuntimeError(f"数据库 schema 版本 {current} 高于代码已知版本 {MAX_KNOWN_SCHEMA_VERSION}，拒绝启动")
    for version,path in files:
        if version<=current:
            continue
        with open(path,encoding="utf-8") as f:
            sql=f.read()
        #executescript 会先隐式提交，因此把 BEGIN/COMMIT 写进同一个脚本保证原子性
        script=f"BEGIN;\n{sql}\nINSERT INTO schema_version(version) VALUES({version});\nCOMMIT;"
        try:
            conn.executescript(script)
        except Exception:
            conn.rollback()
            raise
        current=version
    return current


def create_thread(user):
    thread_id="th_"+uuid.uuid4().hex[:12]
    conn=get_conn()
    now=_now()
    with conn:
        conn.execute("INSERT INTO threads(thread_id,user,created_at,updated_at) VALUES(?,?,?,?)",
                     (thread_id,user,now,now))
    return thread_id


def get_thread(thread_id,user):
    #归属校验在数据层完成：查不到或不属于该用户都返回None，由接口层统一转404
    row=get_conn().execute("SELECT thread_id,user,created_at,updated_at FROM threads WHERE thread_id=? AND user=?",
                           (thread_id,user)).fetchone()
    return dict(row) if row else None


def list_threads(user,limit=50):
    #会话列表按最近更新排序；标题优先用用户自定义的，没改名才回退到第一条提问
    rows=get_conn().execute(
        "SELECT thread_id,title,created_at,updated_at FROM threads WHERE user=?"
        " ORDER BY updated_at DESC LIMIT ?",(user,limit)).fetchall()
    out=[]
    for row in rows:
        item=dict(row)
        custom=(item.pop("title","") or "").strip()
        if custom:
            item["title"]=custom
            item["title_source"]="custom"
            out.append(item)
            continue
        first=get_conn().execute(
            "SELECT content FROM messages WHERE thread_id=? AND role='user'"
            " ORDER BY message_id ASC LIMIT 1",(item["thread_id"],)).fetchone()
        item["title"]=((first["content"] if first else "") or "新对话")[:40]
        item["title_source"]="auto"
        out.append(item)
    return out


def rename_thread(thread_id,user,title):
    #改名同样做归属校验；空标题视为恢复为自动命名
    #不更新 updated_at：改名不算"有新活动"，否则会话会在列表里跳位置
    if get_thread(thread_id,user) is None:
        return None
    clean=(title or "").strip()[:60]
    conn=get_conn()
    with conn:
        conn.execute("UPDATE threads SET title=? WHERE thread_id=?",(clean,thread_id))
    return clean


def delete_thread(thread_id,user):
    #归属校验：不是本人的会话一律当作不存在，由接口层转404
    if get_thread(thread_id,user) is None:
        return False
    conn=get_conn()
    with conn:
        conn.execute("DELETE FROM messages WHERE thread_id=?",(thread_id,))
        conn.execute("DELETE FROM tasks WHERE thread_id=?",(thread_id,))
        conn.execute("DELETE FROM threads WHERE thread_id=?",(thread_id,))
    return True


def add_message(thread_id,user,role,message_type,content,sources=None,rewritten_query=None):
    if get_thread(thread_id,user) is None:
        return None
    conn=get_conn()
    now=_now()
    payload=json.dumps(sources or [],ensure_ascii=False)
    with conn:
        cur=conn.execute(
            "INSERT INTO messages(thread_id,role,message_type,content,sources_json,rewritten_query,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (thread_id,role,message_type,content,payload,rewritten_query,now))
        conn.execute("UPDATE threads SET updated_at=? WHERE thread_id=?",(now,thread_id))
    return int(cur.lastrowid)


def list_messages(thread_id,user,limit=None):
    if get_thread(thread_id,user) is None:
        return None
    sql="SELECT message_id,role,message_type,content,sources_json,rewritten_query,created_at FROM messages WHERE thread_id=? ORDER BY message_id ASC"
    rows=get_conn().execute(sql,(thread_id,)).fetchall()
    out=[_message_row(r) for r in rows]
    return out[-limit:] if limit else out


def recent_history(thread_id,user,turns=3):
    #多轮改写只取最近若干条问答，按时间正序返回，供纯函数拼接使用
    rows=list_messages(thread_id,user,limit=turns*2)
    if rows is None:
        return None
    return [{"role":r["role"],"content":r["content"]} for r in rows]


def _message_row(row):
    item=dict(row)
    raw=item.get("sources_json") or ""
    try:
        item["sources"]=json.loads(raw) if raw else []
    except Exception:
        item["sources"]=[]
    item.pop("sources_json",None)
    return item


def create_task(thread_id,user,route):
    if get_thread(thread_id,user) is None:
        return None
    task_id="tk_"+uuid.uuid4().hex[:12]
    conn=get_conn()
    now=_now()
    with conn:
        conn.execute(
            "INSERT INTO tasks(task_id,thread_id,user,route,status,created_at,started_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (task_id,thread_id,user,route,"running",now,now))
    return task_id


def update_task(task_id,status,error_message="",checkpoint=""):
    #状态落库走短事务：不在这里做长任务的中间写入，避免长时间持有写锁
    conn=get_conn()
    finished=_now() if status in ("succeeded","failed","cancelled") else None
    with conn:
        conn.execute("UPDATE tasks SET status=?,error_message=?,checkpoint=?,finished_at=COALESCE(?,finished_at) WHERE task_id=?",
                     (status,error_message,checkpoint,finished,task_id))
    return True


def get_task(task_id,user):
    row=get_conn().execute("SELECT * FROM tasks WHERE task_id=? AND user=?",(task_id,user)).fetchone()
    return dict(row) if row else None


def record_cost(task_id,thread_id,user,model,prompt_tokens=0,completion_tokens=0,total_tokens=0,cost=0.0):
    #成本归因：llm_usage.jsonl 仍是预算权威，这张表只做按任务/按用户的查询
    conn=get_conn()
    with conn:
        conn.execute(
            "INSERT INTO costs(task_id,thread_id,user,model,prompt_tokens,completion_tokens,"
            "total_tokens,cost,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (task_id,thread_id,user,model,prompt_tokens,completion_tokens,total_tokens,float(cost),_now()))
    return True


def record_tool_call(task_id,thread_id,user,tool,args_digest="",ok=True,duration_ms=None,result_size=None):
    #工具归因：记调用名、参数摘要、成功与否、耗时与返回体量
    conn=get_conn()
    with conn:
        conn.execute(
            "INSERT INTO tool_calls(task_id,thread_id,user,tool,args_digest,ok,duration_ms,"
            "result_size,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (task_id,thread_id,user,tool,args_digest,1 if ok else 0,duration_ms,result_size,_now()))
    return True


def task_attribution(task_id,user):
    #单个任务的成本与工具调用汇总，供 GET /tasks/{id} 直接展示
    conn=get_conn()
    if get_task(task_id,user) is None:
        return None
    cost=conn.execute(
        "SELECT COALESCE(SUM(cost),0) AS c,COALESCE(SUM(total_tokens),0) AS t,COUNT(*) AS n"
        " FROM costs WHERE task_id=? AND user=?",(task_id,user)).fetchone()
    tools=conn.execute(
        "SELECT tool,COUNT(*) AS n,COALESCE(SUM(duration_ms),0) AS ms,"
        "SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) AS bad"
        " FROM tool_calls WHERE task_id=? AND user=? GROUP BY tool ORDER BY n DESC",
        (task_id,user)).fetchall()
    return {
        "cost":round(cost["c"],6),
        "tokens":int(cost["t"]),
        "llm_calls":cost["n"],
        "tool_calls_total":sum(r["n"] for r in tools),
        "tool_calls_failed":sum(r["bad"] or 0 for r in tools),
        "tools":[{"tool":r["tool"],"calls":r["n"],"duration_ms":int(r["ms"] or 0),"failed":r["bad"] or 0} for r in tools],
    }


def list_tasks(thread_id,user):
    if get_thread(thread_id,user) is None:
        return None
    rows=get_conn().execute("SELECT * FROM tasks WHERE thread_id=? ORDER BY created_at ASC",(thread_id,)).fetchall()
    return [dict(r) for r in rows]


def _p95(values):
    if not values:
        return 0.0
    ordered=sorted(values)
    return ordered[min(len(ordered)-1,int(round(0.95*(len(ordered)-1))))]


def metrics_summary(user,since_hours=24):
    #按用户汇总运行时可观测指标：任务状态分布、耗时、消息类型计数
    conn=get_conn()
    since=(datetime.now(timezone.utc)-timedelta(hours=since_hours)).isoformat(timespec="seconds")
    tasks=conn.execute(
        "SELECT route,status,started_at,finished_at FROM tasks WHERE user=? AND created_at>=?",
        (user,since)).fetchall()
    by_status={}
    by_route={}
    durations=[]
    for row in tasks:
        by_status[row["status"]]=by_status.get(row["status"],0)+1
        by_route[row["route"]]=by_route.get(row["route"],0)+1
        if row["started_at"] and row["finished_at"]:
            try:
                start=datetime.fromisoformat(row["started_at"])
                end=datetime.fromisoformat(row["finished_at"])
                durations.append((end-start).total_seconds())
            except Exception:
                pass
    messages=conn.execute(
        "SELECT m.message_type AS t,COUNT(*) AS c FROM messages m"
        " JOIN threads th ON m.thread_id=th.thread_id"
        " WHERE th.user=? AND m.created_at>=? GROUP BY m.message_type",
        (user,since)).fetchall()
    cost_rows=conn.execute(
        "SELECT model,SUM(cost) AS c,COUNT(*) AS n FROM costs WHERE user=? AND created_at>=? GROUP BY model",
        (user,since)).fetchall()
    cost_total=sum(r["c"] or 0 for r in cost_rows)
    cost_by_model={r["model"]:round(r["c"] or 0,6) for r in cost_rows}
    tool_rows=conn.execute(
        "SELECT tool,COUNT(*) AS n,SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) AS bad"
        " FROM tool_calls WHERE user=? AND created_at>=? GROUP BY tool",
        (user,since)).fetchall()
    tool_total=sum(r["n"] for r in tool_rows)
    tool_failed=sum(r["bad"] or 0 for r in tool_rows)
    tool_by_name={r["tool"]:r["n"] for r in tool_rows}
    total=len(tasks)
    succeeded=by_status.get("succeeded",0)
    failed=by_status.get("failed",0)
    return {
        "window_hours":since_hours,
        "cost":{"total":round(cost_total,6),"by_model":cost_by_model},
        "tool_calls":{"total":tool_total,"failed":tool_failed,"by_tool":tool_by_name},
        "tasks":{"total":total,"by_status":by_status,"by_route":by_route,
                 "succeeded":succeeded,"failed":failed,
                 "success_rate":(succeeded/(succeeded+failed)) if (succeeded+failed) else None,
                 "avg_duration_s":round(sum(durations)/len(durations),2) if durations else None,
                 "p95_duration_s":round(_p95(durations),2) if durations else None},
        "messages":{"total":sum(r["c"] for r in messages),
                    "by_type":{r["t"]:r["c"] for r in messages}},
        "threads":conn.execute("SELECT COUNT(*) AS c FROM threads WHERE user=?",(user,)).fetchone()["c"],
    }
