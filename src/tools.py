import os

from config import MAX_PAPER_PER_QUERY
from state import PaperMeta

def build_search_result(results):
    #检索结果统一映射为PaperMeta+Chunk，同一论文只保留一份元数据
    from doc_ingest import Chunk, token_len
    papers=[]
    seen=set()
    chunks=[]
    for i,r in enumerate(results):
        paper_id=r["source"]
        if paper_id not in seen:
            seen.add(paper_id)
            papers.append(PaperMeta(paper_id=paper_id,title=paper_id))
        section=r.get("section","")
        text=r.get("text","")
        chunks.append(Chunk(
            chunk_id=f"{paper_id}::{section}::{i}",
            doc_id=paper_id,
            section_name=section,
            text=text,
            token_len=token_len(text),
            page_num=None,
            parent_id=f"{paper_id}::{section}",
            position=""
        ))
    return {"papers":papers,"chunks":chunks}

def search_papers(query,limit=MAX_PAPER_PER_QUERY):
    #本地检索入口：复用rag_tool混合检索，输出契约化的papers+chunks
    from rag_tool import search_papers_rerank, search_papers_structured
    if os.getenv("USE_RERANK")=="1":
        results=search_papers_rerank(query,k=limit)
    else:
        results=search_papers_structured(query,k=limit)
    return build_search_result(results)
