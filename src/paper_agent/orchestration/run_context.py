import contextvars

#当前任务上下文：让 LLM 用量与工具调用能归因到具体任务
#用 contextvars 而不是线程局部变量：asyncio.to_thread 会把上下文复制进工作线程
_ctx=contextvars.ContextVar("run_context",default=None)


def set_run_context(task_id="",thread_id="",user=""):
    return _ctx.set({"task_id":task_id,"thread_id":thread_id,"user":user})


def reset_run_context(token):
    try:
        _ctx.reset(token)
    except Exception:
        pass


def current_run_context():
    return _ctx.get() or {}
