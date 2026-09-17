from retrieval_service import get_service

SERVICE=get_service()


def _get_service():
    return SERVICE


def __getattr__(name):
    #兼容旧实验脚本的 index/model/bm25 导入，不在模块导入时加载重资源
    service=_get_service()
    if name=="sources":
        return service.sources
    if name=="sections":
        return service.sections
    if name=="documents":
        return service.documents
    if name=="index":
        service._ensure_index()
        return service.index
    if name=="model":
        service._ensure_index()
        return service.model
    if name=="bm25":
        service._ensure_bm25()
        return service._bm25
    raise AttributeError(name)


def read_section(source,section,max_chars=4000):
    #旧接口适配：全文展开统一收敛到tools.read_section
    from tools import read_section as _read
    result=_read(source,section,max_chars=max_chars)
    return result if result is not None else f"未找到章节：{section}"


def hybrid_search(query,k=5,alpha=0.5):
    #兼容旧接口：只做混合召回，返回(index,score)
    items,status=_get_service().recall(query,k,alpha)
    return [(item["_index"],item["recall_score"]) for item in items],status


def search_papers_hybrid(query,k=5,alpha=0.5):
    service=_get_service()
    items,status=service.recall(query,k,alpha)
    if status=="failed":
        return "检索失败：向量与关键词检索均不可用，请检查索引或模型配置"
    if not items:
        return "未找到相关论文"
    return "\n".join(
        f"[{item['source']} - {item['section']}]（相关度{item['recall_score']:.3f}）{item['text'][:300]}"
        for item in items
    )


def search_papers_structured(query,k=5,alpha=0.5):
    service=_get_service()
    items=service.search(query,candidate_k=max(k,service.candidate_k),rerank_mode="none",top_k=k)
    return [
        {
            "source":item["source"],
            "section":item["section"],
            "score":item["recall_score"],
            "text":item["text"][:300],
            "chunk_id":item["chunk_id"],
            "page_num":None,
            "parent_id":f"{item['source']}::{item['section']}",
        }
        for item in items
    ]


def search_papers_rerank(query,k=5,alpha=0.5,candidates=None):
    service=_get_service()
    candidate_k=candidates or service.candidate_k
    items=service.search(query,candidate_k=candidate_k,rerank_mode=service.rerank_mode,top_k=k)
    for item in items:
        item["text"]=item["text"][:300]
    return items


def search_papers_rerank_text(query,k=5,alpha=0.5,candidates=None):
    items=search_papers_rerank(query,k,alpha,candidates)
    if not items:
        return "未找到相关论文"
    return "\n".join(
        f"[{item['source']} - {item['section']}]（相关度{item['recall_score']:.3f}）{item['text']}"
        for item in items
    )
