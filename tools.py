import json
import logging
import os
import re
from dataclasses import asdict
from agent2_parse import build_paper_card as old_build_card, compare_papers
from config import MAX_CHUNKS_PER_PAPER, MAX_CONFLICT_PER_SESSION, MAX_FACTS_PER_SESSION, MAX_PAPER_PER_QUERY, MAX_PENDING_IN_REVIEW
from doc_ingest import Chunk, DocumentRecord, chunk_splitter, load_sections, paper_title, token_len
from llm_api import safe_call_deepseek
from state import Conflict, FactItem, PaperCard, PaperMeta, ReviewReport, SectionRef

logger=logging.getLogger(__name__)

FACT_EXTRACT_PROMPT="""你是论文事实抽取器。根据论文卡片抽取可跨论文比较的事实主张，只输出JSON数组：
[{"paper_id":"论文ID","entity":"实体","attribute":"属性","value":"值","content":"一句话描述","section_name":"来源章节名","raw_quote":"原文摘录不超过100字"}]
约束：必须带paper_id和section_name；没有明确来源的不输出；不要编造。"""

VERIFY_PROMPT="""你是冲突裁决器。根据两个事实主张、双方年份和原文证据，裁决冲突类型，只输出JSON对象：
{"category":"true_conflict|superseded|misunderstanding|card_error|insufficient_evidence","title":"简短标题","detail":"详细裁决分析","evidence_supplementary":"补充证据"}
规则：
- true_conflict：两个结论在相同范围内都没有被对方推翻，属于实质矛盾
- superseded：后发表论文显式引用并否定同范围旧结论，新结论取代旧结论（注意双方年份）
- misunderstanding：理解错误
- card_error：卡片抽取错误
- insufficient_evidence：证据不足
不要因为发布时间晚就自动判 superseded，必须看到显式否定证据。不要编造。"""

REVIEW_PROMPT="""你是综述写作器。根据事实、冲突和用户问题生成结构化综述，只输出JSON对象：
{"title":"综述标题","consensus":["核心共识1"],"disagreements":[{"conflict_id":"","summary":"分歧摘要"}],"superseded_conclusions":["被推翻的历史结论"],"open_questions":["未解决问题"],"conflict_mark_list":["冲突标注"]}
如果给了pending_conflicts，必须在conflict_mark_list里用“⚠待核实”标注，表示尚未完成深度核验。
引用由系统生成，不要自己编引用。不要编造。"""

_chunk_cache={}
_fact_counter=0
_conflict_counter=0

def _truncate_boundary(text,max_chars):
    #尽量在句子边界截断，避免切在半句话
    if len(text)<=max_chars:
        return text
    part=text[:max_chars]
    idx=max(part.rfind(c) for c in "。！？.!?；;\n")
    return part[:idx+1] if idx>max_chars//2 else part

def build_search_result(results):
    #检索结果统一映射为PaperMeta+Chunk，同一论文只保留一份元数据
    papers=[]
    seen=set()
    chunks=[]
    for i,r in enumerate(results):
        paper_id=r["source"]
        if paper_id not in seen:
            seen.add(paper_id)
            title=paper_title(paper_id) or paper_id
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
    #rag_tool顶层会加载SentenceTransformer，保持函数内导入避免拖慢测试与导入
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
    sections=load_sections(paper_id)
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
    card=old_build_card(paper_id)
    sections_ref=[]
    for sec in load_sections(paper_id):
        sections_ref.append(SectionRef(name=sec.get("title",""),page=sec.get("page"),chunk_ids=[]))
    key_findings=card.get("innovation",[])
    if isinstance(key_findings,str):
        key_findings=[key_findings] if key_findings!="未提及" else []
    limitations=card.get("limitations","")
    if isinstance(limitations,str):
        limitations=[] if limitations=="未提及" else [limitations]
    paper_card=PaperCard(
        paper_id=paper_id,
        title=card.get("title") or paper_title(paper_id) or paper_id,
        abstract=card.get("background",""),
        key_findings=key_findings,
        methodology=card.get("method",""),
        limitations=limitations,
        sections_ref=sections_ref
    )
    if cache is not None:
        cache[paper_id]=paper_card
    return paper_card

def _extract_json_array(text):
    #兼容数组/对象两种输出；对象里第一个list字段视为事实列表
    start=text.find("[")
    end=text.rfind("]")
    if start!=-1 and end!=-1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    start=text.find("{")
    end=text.rfind("}")
    if start!=-1 and end!=-1:
        try:
            obj=json.loads(text[start:end+1])
            if isinstance(obj,list):
                return obj
            if isinstance(obj,dict):
                for v in obj.values():
                    if isinstance(v,list):
                        return v
                return [obj]
        except Exception:
            pass
    return []

def _trim_card(card):
    #长字段裁剪后再序列化，避免截断JSON破坏结构
    for k in ("abstract","methodology"):
        if isinstance(card.get(k),str):
            card[k]=card[k][:2000]
    for k in ("key_findings","limitations"):
        if isinstance(card.get(k),list):
            card[k]=[str(x)[:500] for x in card[k]][:20]
    return card

def build_pending_conflicts(facts):
    #确定性粗筛：相同entity+attribute、value不同 → 待核实冲突候选
    groups={}
    for f in facts:
        key=(f.entity,f.attribute)
        groups.setdefault(key,{}).setdefault(f.value,[]).append(f.fact_id)
    pending=[]
    for (entity,attribute),values in groups.items():
        if len(values)>1:
            pending.append({
                "entity":entity,
                "attribute":attribute,
                "values":list(values.keys()),
                "fact_ids":[fid for v in values.values() for fid in v]
            })
    return pending

def _extract_facts(cards_text):
    #LLM事实抽取：JSON格式失败时带纠正指令重试一次，仍失败返回空
    messages=[
        {"role":"system","content":FACT_EXTRACT_PROMPT},
        {"role":"user","content":cards_text}
    ]
    for attempt in range(2):
        result=safe_call_deepseek(messages,temperature=0.2,max_tokens=3000)
        items=_extract_json_array(result["choices"][0]["message"]["content"])
        if items:
            return items
        #把失败输出回填给LLM，让它看到自己的错误样板再纠正
        messages.append({"role":"assistant","content":result["choices"][0]["message"]["content"]})
        messages.append({"role":"user","content":"上述输出不是合法JSON数组，请只输出JSON数组。"})
    logger.warning("事实抽取两次均失败，本次返回空事实列表")
    return []

def _resolve_chunk_id(paper_id,section_name):
    #章节名映射到真实chunk_id：精确→模糊→None，来源不可追溯的事实直接丢弃
    if paper_id not in _chunk_cache:
        sections=load_sections(paper_id)
        _chunk_cache[paper_id]=chunk_splitter(DocumentRecord(source=paper_id,doc_type="pdf_text",text="",sections=sections)) if sections else []
    target=section_name.strip()
    for c in _chunk_cache[paper_id]:
        t=c.section_name.strip()
        if target==t or (target and target in t) or (t and t in target):
            return c.chunk_id
    import difflib
    titles=[c.section_name for c in _chunk_cache[paper_id]]
    matches=difflib.get_close_matches(target,titles,n=1,cutoff=0.6)
    if matches:
        for c in _chunk_cache[paper_id]:
            if c.section_name==matches[0]:
                return c.chunk_id
    return None

def analyze_paper_relations(paper_ids,cache=None):
    #横向对比复用旧逻辑；事实抽取后必须带真实chunk_id，否则丢弃
    comparison=compare_papers(paper_ids)
    cards=[build_paper_card(pid,cache) for pid in paper_ids]
    cards_text=json.dumps([_trim_card(asdict(c)) for c in cards],ensure_ascii=False,default=str)
    items=_extract_facts(cards_text)
    global _fact_counter
    facts=[]
    for it in items:
        pid=it.get("paper_id","")
        section=it.get("section_name","")
        if pid not in paper_ids or not section:
            continue
        chunk_id=_resolve_chunk_id(pid,section)
        if not chunk_id:
            continue
        _fact_counter+=1
        facts.append(FactItem(
            fact_id=f"fact-{_fact_counter}",
            paper_id=pid,
            entity=it.get("entity",""),
            attribute=it.get("attribute",""),
            value=it.get("value",""),
            content=it.get("content",""),
            source_chunk_id=chunk_id,
            section_name=section,
            raw_quote=it.get("raw_quote",""),
            confidence="high"
        ))
    return {"comparison":comparison,"facts":facts}

def _extract_verdict(text):
    #遍历所有JSON对象候选，取带category的裁决对象，兼容前后缀和多个对象
    decoder=json.JSONDecoder()
    for m in re.finditer(r"\{",text):
        try:
            obj,_=decoder.raw_decode(text[m.start():])
        except Exception:
            continue
        if isinstance(obj,dict) and "category" in obj:
            return obj
    return {}

def verify_claim(fact_a,fact_b):
    #冲突二次取证：读取双方章节原文，LLM裁决，category必须落在四种枚举
    evidence_a=read_section(fact_a.paper_id,fact_a.section_name) or fact_a.raw_quote
    evidence_b=read_section(fact_b.paper_id,fact_b.section_name) or fact_b.raw_quote
    payload={
        "fact_a":asdict(fact_a),
        "fact_b":asdict(fact_b),
        "evidence_a":evidence_a[:2000],
        "evidence_b":evidence_b[:2000]
    }
    result=safe_call_deepseek(
        [{"role":"system","content":VERIFY_PROMPT},{"role":"user","content":json.dumps(payload,ensure_ascii=False,default=str)}],
        temperature=0.1,
        max_tokens=1000
    )
    verdict=_extract_verdict(result["choices"][0]["message"]["content"])
    allowed={"true_conflict","superseded","misunderstanding","card_error","insufficient_evidence"}
    category=verdict.get("category","insufficient_evidence")
    if category not in allowed:
        logger.warning(f"裁决返回非法category：{category}，回退insufficient_evidence")
        category="insufficient_evidence"
    global _conflict_counter
    _conflict_counter+=1
    return Conflict(
        conflict_id=f"conflict-{_conflict_counter}",
        fact_ids=[fact_a.fact_id,fact_b.fact_id],
        category=category,
        description=verdict.get("title",""),
        verdict=verdict.get("detail",""),
        confidence="low" if category=="insufficient_evidence" else "high",
        evidence_supplementary=verdict.get("evidence_supplementary","")
    )

def _extract_review_report(text):
    #多重解析：raw_decode候选优先，正则整体兜底，仍失败记warning
    decoder=json.JSONDecoder()
    for m in re.finditer(r"\{",text):
        try:
            obj,_=decoder.raw_decode(text[m.start():])
        except Exception:
            continue
        if isinstance(obj,dict) and "title" in obj:
            return obj
    m=re.search(r"\{.*\}",text,re.S)
    if m:
        try:
            obj=json.loads(m.group())
            if isinstance(obj,dict) and "title" in obj:
                return obj
        except Exception:
            pass
    logger.warning("综述JSON解析失败，返回空结构")
    return {}

def write_review(query,facts,conflicts,pending_conflicts=None):
    #引用从facts确定性生成（保证chunk_id），LLM只负责综述内容
    references=[]
    seen=set()
    for f in facts:
        ref=f"{f.paper_id}, {f.year or 'N/A'}, §{f.source_chunk_id}"
        if ref not in seen:
            seen.add(ref)
            references.append(ref)
    #限制数量+裁剪长字段，保证JSON完整，不再截断payload
    facts_payload=[asdict(f) for f in facts[:MAX_FACTS_PER_SESSION]]
    for f in facts_payload:
        for k in ("content","raw_quote"):
            if isinstance(f.get(k),str):
                f[k]=f[k][:500]
    conflicts_payload=[asdict(c) for c in conflicts[:MAX_CONFLICT_PER_SESSION]]
    for c in conflicts_payload:
        for k in ("description","verdict","evidence_supplementary"):
            if isinstance(c.get(k),str):
                c[k]=c[k][:1000]
    pending_payload=[p for p in (pending_conflicts or [])][:MAX_PENDING_IN_REVIEW]
    payload={"query":query,"facts":facts_payload,"conflicts":conflicts_payload,"pending_conflicts":pending_payload}
    result=safe_call_deepseek(
        [{"role":"system","content":REVIEW_PROMPT},{"role":"user","content":json.dumps(payload,ensure_ascii=False,default=str)}],
        temperature=0.2,
        max_tokens=4000
    )
    data=_extract_review_report(result["choices"][0]["message"]["content"])
    def _as_list(v):
        return v if isinstance(v,list) else []
    return ReviewReport(
        title=data.get("title") or query,
        consensus=_as_list(data.get("consensus")),
        disagreements=_as_list(data.get("disagreements")),
        superseded_conclusions=_as_list(data.get("superseded_conclusions")),
        open_questions=_as_list(data.get("open_questions")),
        references=references,
        conflict_mark_list=_as_list(data.get("conflict_mark_list"))
    )
