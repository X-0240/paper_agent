from task_router import classify_task

def test_survey_keywords():
    #对比/综述类问题必须走全链路
    assert classify_task("对比Transformer和BERT")=="survey"
    assert classify_task("梳理大模型技术演进")=="survey"

def test_simple_question():
    #普通问答走短路，不启动多Agent
    assert classify_task("什么是Transformer")=="simple"
