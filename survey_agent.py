import logging

from agent_react import react
from config import MAX_AGENT_STEP, MAX_PAPER_PER_QUERY, MAX_SEARCH_PER_SESSION
from llm_api import BudgetExceeded
from state import AgentState, ReviewReport
from tools import (analyze_paper_relations as _analyze, build_paper_card as _build_card,
                   build_pending_conflicts, read_section as _read_section,
                   search_papers as _search_papers, verify_claim as _verify_claim,
                   write_review as _write_review)

logger=logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE="""你是论文调研综述Agent。用户给一个综述/对比类问题，你按需调用工具完成检索、读章节、建卡、抽取事实、核验冲突、生成综述。
工具规则：
1. 先search_papers找到论文，再决定是否read_section/build_paper_card
2. 卡片足够后必须调用analyze_paper_relations抽取事实，再决定是否verify_claim
3. facts不足不能写综述；有冲突候选时用verify_claim核验
4. 最后必须调用write_review生成综述，然后Finish
5. 搜索最多{max_search}次，不要重复调用已读章节/重复建卡
6. facts为空时不要调用write_review，直接Finish，答案写“证据不足”
注意：总步数最多{max_steps}，检索/读章节/建卡不要过度重复，尽快进入事实抽取和综述阶段。
只能调用以下工具，工具名和参数必须精确匹配，禁止发明工具：
- search_papers: {{"query": "..."}}
- read_section: {{"paper_id": "...", "section_name": "..."}}
- build_paper_card: {{"paper_id": "..."}}
- analyze_paper_relations: {{"paper_ids": ["...", "..."]}}
- verify_claim: {{"fact_id_a": "...", "fact_id_b": "..."}}
- write_review: {{"query": "..."}}
当前状态：{state_summary}
每次输出格式：
Thought: ...
Action: 工具名
Action Input: {{"参数":"值"}}"""

def _paper_ids(state):
    return {p.paper_id for p in state.papers}

def _summarize_state(state):
    return (f"papers={len(state.papers)}, cards={len(state.cards)}, facts={len(state.facts)}, "
            f"conflicts={len(state.conflicts)}, pending={len(state.pending_conflicts)}, "
            f"search_count={state.search_count}/{MAX_SEARCH_PER_SESSION}")

def _enrich_trace(entry,state):
    #每步审计快照：工具名/入参/返回由history自带，这里补状态计数
    entry["search_count"]=state.search_count
    entry["step_count"]=entry.get("step")
    entry["facts_count"]=len(state.facts)
    entry["conflicts_count"]=len(state.conflicts)
    entry["pending_count"]=len(state.pending_conflicts)

def _make_tools(state):
    #每个工具包装：前置校验、State读写由编排层完成，工具本身无副作用
    def search_papers(query):
        if state.search_count>=MAX_SEARCH_PER_SESSION:
            return "错误：搜索次数已达上限，请基于已有论文继续或调用write_review"
        result=_search_papers(query,limit=MAX_PAPER_PER_QUERY)
        state.search_count+=1
        existing=_paper_ids(state)
        for p in result["papers"]:
            if p.paper_id not in existing:
                state.papers.append(p)
                existing.add(p.paper_id)
        state.candidate_chunks.extend(result["chunks"])
        names=", ".join(p.paper_id for p in state.papers)
        return f"检索完成，当前论文：{names}"

    def read_section(paper_id,section_name):
        if paper_id not in _paper_ids(state):
            return f"错误：论文{paper_id}不在已检索论文中"
        text=_read_section(paper_id,section_name,cache=state.section_cache)
        if text is None:
            return f"未找到章节：{section_name}"
        return f"【{paper_id}::{section_name}】\n{text[:1500]}"

    def build_paper_card(paper_id):
        if paper_id not in _paper_ids(state):
            return f"错误：论文{paper_id}不在已检索论文中"
        card=_build_card(paper_id,cache=state.card_cache)
        if not any(c.paper_id==card.paper_id for c in state.cards):
            state.cards.append(card)
        return f"卡片已生成：{card.title}"

    def analyze_paper_relations(paper_ids):
        missing=[p for p in paper_ids if p not in _paper_ids(state)]
        if missing:
            return f"错误：以下论文未检索：{missing}"
        result=_analyze(paper_ids,cache=state.card_cache)
        state.facts.extend(result["facts"])
        state.pending_conflicts=build_pending_conflicts(state.facts)
        return f"抽取事实{len(result['facts'])}条，当前facts={len(state.facts)}，待核实冲突={len(state.pending_conflicts)}"

    def verify_claim(fact_id_a,fact_id_b):
        by_id={f.fact_id:f for f in state.facts}
        if fact_id_a not in by_id or fact_id_b not in by_id:
            return "错误：fact_id不存在，请先调用analyze_paper_relations"
        conflict=_verify_claim(by_id[fact_id_a],by_id[fact_id_b])
        state.conflicts.append(conflict)
        pair={fact_id_a,fact_id_b}
        state.pending_conflicts=[p for p in state.pending_conflicts if not pair.issubset(set(p.get("fact_ids",[])))]
        return f"裁决完成：{conflict.category} {conflict.description}"

    def write_review(query):
        if not state.facts:
            return "错误：没有事实，不能写综述"
        report=_write_review(query,state.facts,state.conflicts,state.pending_conflicts)
        state.review=report
        return (f"综述已生成：{report.title}"
                f"（{len(report.consensus)}条共识，{len(report.disagreements)}条分歧，"
                f"{len(report.superseded_conclusions)}条被推翻结论）")

    return {
        "search_papers":{"func":search_papers,"description":"检索论文","schema":{"query":{"type":"str","required":True}}},
        "read_section":{"func":read_section,"description":"读取章节","schema":{"paper_id":{"type":"str","required":True},"section_name":{"type":"str","required":True}}},
        "build_paper_card":{"func":build_paper_card,"description":"生成论文卡片","schema":{"paper_id":{"type":"str","required":True}}},
        "analyze_paper_relations":{"func":analyze_paper_relations,"description":"多论文抽取事实","schema":{"paper_ids":{"type":"list","required":True}}},
        "verify_claim":{"func":verify_claim,"description":"核验冲突","schema":{"fact_id_a":{"type":"str","required":True},"fact_id_b":{"type":"str","required":True}}},
        "write_review":{"func":write_review,"description":"生成综述","schema":{"query":{"type":"str","required":True}}}
    }

def _finalize_review(state,query):
    #循环结束后若还没写综述且有facts，补写一份
    if state.review is None and state.facts:
        state.review=_write_review(query,state.facts,state.conflicts,state.pending_conflicts)

def _auto_finalize(state,query):
    #编排层兜底：循环结束后有卡片但没事实，自动补一次抽取和综述
    if state.review is None and not state.facts and state.cards:
        ids=[c.paper_id for c in state.cards]
        result=_analyze(ids,cache=state.card_cache)
        state.facts.extend(result["facts"])
        state.pending_conflicts=build_pending_conflicts(state.facts)
    _finalize_review(state,query)

def run_survey(query):
    #综述入口：实例化全新State，跑ReAct循环，预算不足时降级生成
    state=AgentState(query=query,complexity="COMPLEX")
    tools=_make_tools(state)
    system_prompt=SYSTEM_PROMPT_TEMPLATE.format(max_search=MAX_SEARCH_PER_SESSION,max_steps=MAX_AGENT_STEP,state_summary=_summarize_state(state))
    history=[]
    try:
        answer,history=react(query,tools,system_prompt,max_steps=MAX_AGENT_STEP,
                             on_step=lambda e: _enrich_trace(e,state))
    except BudgetExceeded as e:
        logger.warning(f"预算受限：{e}")
        if state.facts:
            _finalize_review(state,query)
            answer="预算受限，已生成不完整综述"
        else:
            answer="预算不足，请明日再试"
    state.trace_log.extend(history)
    _auto_finalize(state,query)
    return state
