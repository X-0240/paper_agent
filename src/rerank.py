import os
#国内网络直连HuggingFace可能超时，必须在import sentence_transformers前设置镜像
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import logging
from sentence_transformers import CrossEncoder

logger=logging.getLogger(__name__)
#模型下载时httpx会刷大量INFO日志，压到WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)
_MODEL=None

def get_reranker():
    #懒加载：首次调用才加载模型，避免拖慢启动
    global _MODEL
    if _MODEL is None:
        model_name=os.getenv("RERANK_MODEL_NAME","cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info(f"加载Rerank模型：{model_name}")
        _MODEL=CrossEncoder(model_name)
    return _MODEL

def rerank(query,items,top_n=5):
    #Cross-Encoder精排：query和片段拼一起打分，比向量粗排更准
    if not items:
        return []
    try:
        model=get_reranker()
    except Exception as e:
        #模型不可用时降级：按原混合检索顺序返回前top_n，不让链路断
        logger.warning(f"Rerank模型不可用，按原顺序返回：{e}")
        return items[:top_n]
    pairs=[[query,it["text"]] for it in items]
    scores=model.predict(pairs)
    ranked=sorted(zip(items,scores),key=lambda x:x[1],reverse=True)
    return [it for it,_ in ranked[:top_n]]
