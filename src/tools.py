import os

from config import MAX_CHUNKS_PER_PAPER, MAX_PAPER_PER_QUERY
from state import PaperCard, PaperMeta, SectionRef

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

def build_paper_card(paper_id,cache=None):
    #旧建卡逻辑复用，输出对齐新PaperCard；命中缓存不重复调LLM
    if cache and paper_id in cache:
        return cache[paper_id]
    from agent2_parse import build_paper_card as old_build_card, get_paper_sections
    card=old_build_card(paper_id)
    sections_ref=[]
    try:
        for title,_ in get_paper_sections(paper_id):
            sections_ref.append(SectionRef(name=title,page=None,chunk_ids=[]))
    except Exception:
        pass
    key_findings=card.get("innovation",[])
    if isinstance(key_findings,str):
        key_findings=[key_findings] if key_findings!="未提及" else []
    limitations=card.get("limitations","")
    if isinstance(limitations,str):
        limitations=[] if limitations=="未提及" else [limitations]
    paper_card=PaperCard(
        paper_id=paper_id,
        title=card.get("title") or paper_id,
        abstract=card.get("background",""),
        key_findings=key_findings,
        methodology=card.get("method",""),
        limitations=limitations,
        sections_ref=sections_ref
    )
    if cache is not None:
        cache[paper_id]=paper_card
    return paper_card
