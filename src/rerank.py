import os
#国内网络直连HuggingFace可能超时，必须在import sentence_transformers前设置镜像
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import logging
from sentence_transformers import CrossEncoder
from rerank_policy import RerankCache,cache_key

logger=logging.getLogger(__name__)
#模型下载时httpx会刷大量INFO日志，压到WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)
_MODEL=None
RERANK_MAX_CHARS=int(os.getenv("RERANK_MAX_CHARS","1500"))
_CACHE=RerankCache(int(os.getenv("RERANK_CACHE_SIZE","256")))

def get_reranker():
    #懒加载：首次调用才加载模型，避免拖慢启动
    global _MODEL
    if _MODEL is None:
        model_name=os.getenv("RERANK_MODEL_NAME","cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info(f"加载Rerank模型：{model_name}")
        _MODEL=CrossEncoder(model_name)
    return _MODEL

def rerank(query,items,top_n=5,snapshot="",max_chars=None):
    #Cross-Encoder精排：query和片段拼一起打分，比向量粗排更准
    if not items:
        return []
    max_chars=max_chars or RERANK_MAX_CHARS
    key=cache_key(query,snapshot)
    cached=_CACHE.get(key)
    if cached is not None:
        logger.info("Rerank缓存命中")
        return cached[:top_n]
    try:
        model=get_reranker()
    except Exception as e:
        #模型不可用时降级：按原混合检索顺序返回前top_n，不让链路断
        logger.warning(f"Rerank模型不可用，按原顺序返回：{e}")
        return items[:top_n]
    #全文片段参与打分，过长的再截断，避免证据在片段后半段被忽略
    pairs=[[query,it["text"][:max_chars]] for it in items]
    scores=model.predict(pairs)
    ranked=[it for it,_ in sorted(zip(items,scores),key=lambda x:x[1],reverse=True)]
    _CACHE.set(key,ranked)
    return ranked[:top_n]
