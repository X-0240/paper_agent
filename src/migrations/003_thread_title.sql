-- 会话自定义标题：为空时前端回退到该会话第一条提问
ALTER TABLE threads ADD COLUMN title TEXT NOT NULL DEFAULT '';
