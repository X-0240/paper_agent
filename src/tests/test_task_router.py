from task_router import classify_task, rate_limit_allowed

def test_survey_keywords():
    #对比/综述类问题必须走全链路
    assert classify_task("对比Transformer和BERT")=="survey"
    assert classify_task("梳理大模型技术演进")=="survey"

def test_simple_question():
    #普通问答走短路，不启动多Agent
    assert classify_task("什么是Transformer")=="simple"

def test_rate_limit_allowed():
    #连续请求超过限额后拒绝
    assert rate_limit_allowed("limit_user",limit=2,now=0.0)
    assert rate_limit_allowed("limit_user",limit=2,now=0.1)
    assert not rate_limit_allowed("limit_user",limit=2,now=0.2)

def test_rate_limit_refill():
    #一分钟后令牌恢复，允许再次请求
    assert rate_limit_allowed("refill_user",limit=1,now=100.0)
    assert not rate_limit_allowed("refill_user",limit=1,now=100.1)
    assert rate_limit_allowed("refill_user",limit=1,now=161.0)
