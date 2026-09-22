import logging
import json
import os
import datetime
import threading
import requests
import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
logger=logging.getLogger(__name__)

#DeepSeek配置
api_key=os.getenv("DEEPSEEK_API_KEY")
headers={
    "Authorization":f"Bearer {api_key}",
    "Content-Type":"application/json"
}

TRACKER=None
USAGE_FILE=os.getenv("LLM_USAGE_FILE","llm_usage.jsonl")
DAILY_COST_BUDGET=float(os.getenv("DAILY_COST_BUDGET","5"))
_BUDGET_LOCK=threading.Lock()
_RESERVED_COST=0.0

#DeepSeek V4-Flash峰谷价：高峰9-12点、14-18点，闲时半价（2026-08-17生效）
PEAK_PRICE={"in_hit":0.1,"in_miss":3.0,"out":9.0}
OFFPEAK_PRICE={"in_hit":0.05,"in_miss":1.5,"out":4.5}

class BudgetExceeded(Exception):
    pass

def is_peak_now():
    h=datetime.datetime.now().hour
    return (9<=h<12) or (14<=h<18)

def estimate_cost(usage,peak=None):
    #按官方峰谷价估算单次调用成本，缓存命中/未命中分开计
    in_hit=usage.get("prompt_cache_hit_tokens",0) or 0
    in_miss=usage.get("prompt_cache_miss_tokens",0) or max(0,usage.get("prompt_tokens",0)-in_hit)
    out=usage.get("completion_tokens",0) or 0
    p=PEAK_PRICE if (peak if peak is not None else is_peak_now()) else OFFPEAK_PRICE
    return (in_hit*p["in_hit"]+in_miss*p["in_miss"]+out*p["out"])/1000000

def today_spent():
    #汇总当日已记录成本，用于预算判断
    total=0.0
    today=datetime.date.today().isoformat()
    try:
        with open(USAGE_FILE,encoding="utf-8") as f:
            for line in f:
                try:
                    item=json.loads(line)
                    if item.get("date")==today:
                        total+=item.get("cost",0)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return 0.0
    return total

def log_usage(data):
    #每次调用后落盘：时间/模型/token/估算成本，形成可审计账单
    usage=data.get("usage",{}) or {}
    item={
        "date":datetime.date.today().isoformat(),
        "time":datetime.datetime.now().strftime("%H:%M:%S"),
        "model":data.get("model","deepseek-v4-flash"),
        "prompt_tokens":usage.get("prompt_tokens",0),
        "completion_tokens":usage.get("completion_tokens",0),
        "total_tokens":usage.get("total_tokens",0),
        "cost":round(estimate_cost(usage),6)
    }
    with open(USAGE_FILE,"a",encoding="utf-8") as f:
        f.write(json.dumps(item,ensure_ascii=False)+"\n")
    _record_cost_attribution(item)
    logger.info(f"LLM调用 token={item['total_tokens']} 约{item['cost']:.4f}元 今日累计{today_spent():.4f}元")


def _record_cost_attribution(item):
    #把本次调用归因到当前任务；归因失败绝不能影响主流程，因此整体包在 try 里
    try:
        from run_context import current_run_context
        ctx=current_run_context()
        if not ctx:
            return
        import store
        store.record_cost(ctx.get("task_id",""),ctx.get("thread_id",""),ctx.get("user",""),
                          item.get("model",""),item.get("prompt_tokens",0),
                          item.get("completion_tokens",0),item.get("total_tokens",0),item.get("cost",0))
    except Exception as e:
        logger.warning(f"成本归因写入失败（不影响主流程）：{e}")

def enforce_budget():
    #每日预算硬闸：超过后直接拒绝新调用，防止无人值守烧钱
    if DAILY_COST_BUDGET<=0:
        return
    if today_spent()>=DAILY_COST_BUDGET:
        raise BudgetExceeded(f"当日API预算已用尽：{DAILY_COST_BUDGET}元，可在.env调高或明天再试")

def reserve_budget(estimate=0.005):
    #并发调用前先预留预算，避免多个请求同时穿透每日硬闸
    global _RESERVED_COST
    if DAILY_COST_BUDGET<=0:
        return 0.0
    with _BUDGET_LOCK:
        if today_spent()+_RESERVED_COST+estimate>DAILY_COST_BUDGET:
            raise BudgetExceeded(f"当日API预算不足以预留：{DAILY_COST_BUDGET}元")
        _RESERVED_COST+=estimate
    return estimate

def release_budget(reserved):
    #失败或未产生计费时释放预留额度
    global _RESERVED_COST
    if reserved<=0:
        return
    with _BUDGET_LOCK:
        _RESERVED_COST=max(0.0,_RESERVED_COST-reserved)

def settle_budget(reserved,data):
    #成功后释放预留并按真实usage记账，保证只有一个预算入口
    try:
        log_usage(data)
    finally:
        if reserved>0:
            release_budget(reserved)

def set_tracker(tracker):
    #设置指标追踪器，记录每次LLM调用的次数和token
    global TRACKER
    TRACKER=tracker

def call_deepseek_once(messages,temperature=0.1,max_tokens=None,timeout=60,thinking=None,record_usage=True):
    #低层单次调用：可选关闭thinking，供检索query生成等短超时场景使用
    payload={"model":"deepseek-v4-flash","messages":messages,"temperature":temperature}
    if max_tokens:
        payload["max_tokens"]=max_tokens
    if thinking:
        payload["thinking"]={"type":thinking}
    response=requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    data=response.json()
    if record_usage:
        log_usage(data)
    if TRACKER is not None:
        TRACKER["calls"]=TRACKER.get("calls",0)+1
        TRACKER["tokens"]=TRACKER.get("tokens",0)+data.get("usage",{}).get("total_tokens",0)
    return data

def call_deepseek(messages,temperature=0.1,max_tokens=None):
    #通用LLM调用：默认低温保证稳定，超时60秒防网络挂起
    enforce_budget()
    return call_deepseek_once(messages,temperature,max_tokens)

def call_deepseek_stream(messages,temperature=0.1):
    #SSE流式：逐token产出内容，供接口实时推送
    enforce_budget()
    payload={"model":"deepseek-v4-flash","messages":messages,"temperature":temperature,"stream":True}
    usage=None
    with requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=60
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data=line[5:].strip()
            if data=="[DONE]":
                break
            chunk=json.loads(data)
            if chunk.get("usage"):
                usage=chunk["usage"]
            delta=chunk["choices"][0].get("delta",{}).get("content","")
            if delta:
                yield delta
    if usage:
        log_usage({"model":"deepseek-v4-flash","usage":usage})

#异步流式：同步 requests 在线程里迭代时，每个token都会阻塞线程池，
#事件循环拿到的是"批量"结果；用 httpx 异步客户端才能边收边推
async def call_deepseek_stream_async(messages,temperature=0.1):
    enforce_budget()
    payload={"model":"deepseek-v4-flash","messages":messages,"temperature":temperature,"stream":True}
    usage=None
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0,connect=10.0)) as client:
        async with client.stream("POST","https://api.deepseek.com/v1/chat/completions",
                                 headers=headers,json=payload) as response:
            response.raise_for_status()
            #按字节读再自己切行：aiter_lines 会攒批，token 到手时已经过了好几秒
            buffer=""
            async for raw in response.aiter_bytes():
                buffer+=raw.decode("utf-8","ignore")
                while "\n" in buffer:
                    line,buffer=buffer.split("\n",1)
                    line=line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data=line[5:].strip()
                    if data=="[DONE]":
                        buffer=""
                        break
                    chunk=json.loads(data)
                    if chunk.get("usage"):
                        usage=chunk["usage"]
                    delta=chunk["choices"][0].get("delta",{}).get("content","")
                    if delta:
                        yield delta
    if usage:
        log_usage({"model":"deepseek-v4-flash","usage":usage})

#网络异常重试3次，指数退避，避免偶发抖动直接失败
@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1,min=1,max=10),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def safe_call_deepseek(messages,temperature=0.1,max_tokens=None):
    return call_deepseek(messages,temperature,max_tokens)
