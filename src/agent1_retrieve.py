import logging
import os
import re
import sys
from agent_react import react
from paper_entities import extract_named_papers as _extract_named, normalize_paper_name as _normalize_name
from rag_tool import search_papers_rerank_text, search_papers_hybrid, search_papers_structured, sources, read_section

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if sys.stderr and hasattr(sys.stderr,"reconfigure"):
    sys.stderr.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def normalize_paper_name(name):
    #点名论文识别的唯一实现在 paper_entities，这里只做委托
    return _normalize_name(name,set(sources))

def extract_named_papers(question):
    #用户可能明确点名论文，检索不一定召回，需显式识别并优先加入
    return _extract_named(question,set(sources))

#Agent-1：检索，手写ReAct决定检索词和论文清单
AGENT1_PROMPT="""你是检索Agent（Agent-1）。根据用户课题，从论文知识库检索并确定需要解析的论文。
可用工具：
- search_papers：混合检索论文库，输入query
- read_section：读取论文完整章节，输入source和section
必须严格按格式输出：
Thought: 思考
Action: search_papers（或read_section或Finish）
Action Input: {"query": "检索关键词"} 或 {"source": "论文名", "section": "章节名"} 或 {"papers": ["论文名1", "论文名2"]}
示例：
Thought: 用户对比Transformer和BERT，先检索Transformer架构。
Action: search_papers
Action Input: {"query": "Transformer architecture"}
规则：
1. 检索1-3次，覆盖课题涉及的论文
2. 片段相关但不完整时，用read_section读取完整章节
3. 用户明确点名的论文必须出现在papers里
4. 信息足够后用Finish输出论文名列表，不要输出正文"""

def agent1_search(query,k=3):
    #Rerank默认关闭：中文题实测-1%，换bge后设USE_RERANK=1
    if os.getenv("USE_RERANK")=="1":
        return search_papers_rerank_text(query,k=k)
    return search_papers_hybrid(query,k=k)

agent1_tools={
    "search_papers":{"func":agent1_search,"description":"混合检索论文库，输入query","schema":{"query":{"type":"str","required":True}}},
    "read_section":{"func":read_section,"description":"读取论文完整章节，输入source和section","schema":{"source":{"type":"str","required":True},"section":{"type":"str","required":True}}}
}

def agent_1(question,top_n=3):
    #Agent-1：ReAct检索，返回要解析的论文名列表
    named=extract_named_papers(question)
    answer,history=react(question,agent1_tools,AGENT1_PROMPT,max_steps=6)
    papers=[]
    if history:
        last=history[-1]
        if last.get("action")=="Finish":
            papers=last.get("input",{}).get("papers",[]) or []
    #只保留库内真实论文名，用户点名论文必须包含
    valid=set(sources)
    papers=[normalize_paper_name(p) for p in papers]
    papers=[p for p in papers if p in valid]
    #用户直接给arxiv id（如1603.09381）也视为点名论文
    for m in re.findall(r"\b\d{4}\.\d{4,5}\b",question):
        if m in valid and m not in named:
            named.append(m)
    for n in named:
        if n not in papers:
            papers.append(n)
    #模型没输出有效名单时，回退到结构化检索
    if not papers:
        results=search_papers_structured(question,k=top_n)
        papers=[r["source"] for r in results]
    logger.info(f"Agent-1 选择论文：{papers[:top_n]}")
    return papers[:top_n]
