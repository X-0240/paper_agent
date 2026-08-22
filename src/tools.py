import json
import os

from config import MAX_CHUNKS_PER_PAPER, MAX_PAPER_PER_QUERY
from state import PaperCard, PaperMeta, SectionRef

SECTIONS_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"papers_sections")
TITLES_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"qasper_titles.json")
_titles_cache=None

def _paper_title(paper_id):
    #标题映射：qasper_titles.json优先，查不到回退文件名
    global _titles_cache
    if _titles_cache is None:
        _titles_cache=json.load(open(TITLES_FILE,encoding="utf-8")) if os.path.exists(TITLES_FILE) else {}
    return _titles_cache.get(paper_id,"")

def _load_sections(paper_id):
    #优先读章节缓存，其次doc_ingest解析PDF；统一为dict列表，不依赖deprecated模块
    path=os.path.join(SECTIONS_DIR,f"{paper_id}.json")
    if os.path.exists(path):
        data=json.load(open(path,encoding="utf-8"))
        out=[]
        for item in data:
            if isinstance(item,dict):
                out.append(item)
            else:
                out.append({"title":item[0],"text":item[1],"page":None})
        return out
    pdf_path=os.path.join(os.getenv("PAPERS_DIR",""),f"{paper_id}.pdf")
    if os.path.exists(pdf_path):
        from doc_ingest import parse_document
        record=parse_document(pdf_path)
        return [{"title":s.get("title",""),"text":s.get("text",""),"page":s.get("page")} for s in record.sections]
    return []

def _truncate_boundary(text,max_chars):
    #尽量在句子边界截断，避免切在半句话
    if len(text)<=max_chars:
        return text
    part=text[:max_chars]
    idx=max(part.rfind(c) for c in "。！？.!?；;\n")
    return part[:idx+1] if idx>max_chars//2 else part

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
            title=_paper_title(paper_id) or paper_id
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
    #精确匹配→模糊匹配→None；缓存同时写入请求名和实际章节名
    key=f"{paper_id}||{section_name}"
    if cache and key in cache:
        return cache[key]
    sections=_load_sections(paper_id)
    if not sections:
        return None
    target=section_name.strip()
    for sec in sections:
        title=sec.get("title","").strip()
        if target==title or (target and target in title) or (title and title in target):
            text=_truncate_boundary(sec.get("text",""),max_chars)
            if cache is not None:
                cache[key]=text
            return text
    titles=[sec.get("title","") for sec in sections]
    import difflib
    matches=difflib.get_close_matches(section_name,titles,n=1,cutoff=0.6)
    if matches:
        matched_title=matches[0]
        for sec in sections:
            if sec.get("title","")==matched_title:
                text=_truncate_boundary(sec.get("text",""),max_chars)
                if cache is not None:
                    cache[key]=text
                    cache[f"{paper_id}||{matched_title}"]=text
                return text
    return None

def build_paper_card(paper_id,cache=None):
    #旧建卡逻辑复用，输出对齐新PaperCard；命中缓存不重复调LLM
    if cache and paper_id in cache:
        return cache[paper_id]
    #旧建卡逻辑待移植到新链前暂时保留依赖
    from agent2_parse import build_paper_card as old_build_card
    card=old_build_card(paper_id)
    sections_ref=[]
    for sec in _load_sections(paper_id):
        sections_ref.append(SectionRef(name=sec.get("title",""),page=sec.get("page"),chunk_ids=[]))
    key_findings=card.get("innovation",[])
    if isinstance(key_findings,str):
        key_findings=[key_findings] if key_findings!="未提及" else []
    limitations=card.get("limitations","")
    if isinstance(limitations,str):
        limitations=[] if limitations=="未提及" else [limitations]
    paper_card=PaperCard(
        paper_id=paper_id,
        title=card.get("title") or _paper_title(paper_id) or paper_id,
        abstract=card.get("background",""),
        key_findings=key_findings,
        methodology=card.get("method",""),
        limitations=limitations,
        sections_ref=sections_ref
    )
    if cache is not None:
        cache[paper_id]=paper_card
    return paper_card
