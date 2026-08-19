SURVEY_KEYWORDS=["对比","比较","区别","差异","异同","综述","演进","发展","总结","梳理","关系","脉络"]

def classify_task(question):
    #规则路由：综述/对比类关键词走全链路，其余走短路；模糊场景可后续换LLM分类
    return "survey" if any(k in question for k in SURVEY_KEYWORDS) else "simple"
