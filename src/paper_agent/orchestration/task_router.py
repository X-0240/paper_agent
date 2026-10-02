import os
import time
from collections import defaultdict

RATE_LIMIT_PER_MINUTE=int(os.getenv("RATE_LIMIT_PER_MINUTE","10"))
_rate_buckets=defaultdict(lambda:{"tokens":0.0,"last":None})

def rate_limit_allowed(key,limit=RATE_LIMIT_PER_MINUTE,now=None):
    #进程内令牌桶：按key限流，默认每分钟limit次；now可注入便于测试
    now=time.time() if now is None else now
    bucket=_rate_buckets[key]
    if bucket["last"] is None:
        bucket["tokens"]=float(limit)
    else:
        elapsed=max(0.0,now-bucket["last"])
        bucket["tokens"]=min(float(limit),bucket["tokens"]+elapsed*(limit/60.0))
    bucket["last"]=now
    if bucket["tokens"]<1.0:
        return False
    bucket["tokens"]-=1.0
    return True

SURVEY_KEYWORDS=["对比","比较","区别","差异","异同","综述","演进","发展","总结","梳理","关系","脉络"]

def classify_task(question):
    #规则路由：综述/对比类关键词走全链路，其余走短路；模糊场景可后续换LLM分类
    return "survey" if any(k in question for k in SURVEY_KEYWORDS) else "simple"
