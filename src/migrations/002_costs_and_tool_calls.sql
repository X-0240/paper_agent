-- 归因表：把每次LLM调用与每次工具调用挂到具体任务上
CREATE TABLE costs(
    cost_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    thread_id TEXT,
    user TEXT,
    model TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE tool_calls(
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    thread_id TEXT,
    user TEXT,
    tool TEXT NOT NULL,
    args_digest TEXT,
    ok INTEGER NOT NULL DEFAULT 1,
    duration_ms INTEGER,
    result_size INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_costs_task ON costs(task_id);
CREATE INDEX idx_tool_calls_task ON tool_calls(task_id);
