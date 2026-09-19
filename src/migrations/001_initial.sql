-- 一期持久化：会话、消息、任务与版本表
CREATE TABLE schema_version(
    version INTEGER NOT NULL
);

CREATE TABLE threads(
    thread_id TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages(
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    message_type TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '',
    rewritten_query TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE tasks(
    task_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    user TEXT NOT NULL,
    route TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error_message TEXT,
    checkpoint TEXT
);

CREATE INDEX idx_messages_thread ON messages(thread_id,message_id);
CREATE INDEX idx_tasks_thread ON tasks(thread_id,created_at);
CREATE INDEX idx_threads_user ON threads(user,updated_at);
