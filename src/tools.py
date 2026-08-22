import os

from config import MAX_CHUNKS_PER_PAPER, MAX_PAPER_PER_QUERY
from state import PaperMeta

def build_search_result(results):
    #检索结果统一映射为PaperMeta+Chunk，同一论文只保留一份元数据
    from doc_ingest import Chunk, token_len
    from agent2_parse import get_paper_title
    papers=[]
    seen=set()
    chunks=[]
    for i,r in enumerate(results):
        paper_id=r["source"]
        if paper_id not in seen:
            seen.add(paper_id)
            title=get_paper_title(paper_id) or paper_id
            papers.append(PaperMeta(paper_id=paper_id,title=title))
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

def read_section(paper_id,section_name,cache=None,max_chars=4000):
    #精确匹配→模糊匹配→None；命中后写入会话缓存，避免反复读取
    key=f"{paper_id}||{section_name}"
    if cache and key in cache:
        return cache[key]
    from agent2_parse import get_paper_sections
    try:
        sections=get_paper_sections(paper_id)
    except Exception:
        return None
    for title,text in sections:
        if section_name.strip()==title.strip() or section_name.strip() in title or title.strip() in section_name:
            text=text[:max_chars]
            if cache is not None:
                cache[key]=text
            return text
    import difflib
    titles=[t for t,_ in sections]
    matches=difflib.get_close_matches(section_name,titles,n=1,cutoff=0.6)
    if matches:
        matched=dict(sections)[matches[0]][:max_chars]
        if cache is not None:
            cache[key]=matched
        return matched
    return None
