import asyncio
import logging
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
import httpx

logger=logging.getLogger(__name__)

WEB_SEARCH_TIMEOUT=float(os.getenv("WEB_SEARCH_TIMEOUT","8"))
WEB_SEARCH_MAX_RESULTS=int(os.getenv("WEB_SEARCH_MAX_RESULTS","3"))
WEB_SEARCH_CONCURRENCY=int(os.getenv("WEB_SEARCH_CONCURRENCY","5"))
USER_AGENT="paper-agent-demo/1.0 (paper-agent-demo@example.com)"
ARXIV_MIN_INTERVAL=3.0

_semaphore=None
_arxiv_last=0.0
_arxiv_lock=asyncio.Lock()

def _get_semaphore():
    #懒加载单例：在事件循环内首次创建，避免跨循环绑定问题
    global _semaphore
    if _semaphore is None:
        _semaphore=asyncio.Semaphore(WEB_SEARCH_CONCURRENCY)
    return _semaphore

def _source(source_type,title,snippet,paper_id=None,section=None,url=None,arxiv_id=None):
    #固定字段结构，缺失填None，避免前端缺key
    return {"source_type":source_type,"title":title,"snippet":snippet,
            "paper_id":paper_id,"section":section,"url":url,"arxiv_id":arxiv_id}

async def _fetch_json(url,params):
    #统一外网请求：Semaphore+超时+所有错误降级为空
    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as client:
                resp=await client.get(url,params=params,headers={"User-Agent":USER_AGENT})
                resp.raise_for_status()
                return resp.json()
    except Exception as e:
        logger.warning(f"外网请求失败 {url}: {e}")
        return None

async def search_wikipedia(query,limit=WEB_SEARCH_MAX_RESULTS):
    #中文维基搜索：取标题+摘要+链接，snippet剥掉HTML标签
    params={"action":"query","list":"search","srsearch":query,
            "srlimit":str(limit),"format":"json","formatversion":"2"}
    data=await _fetch_json("https://zh.wikipedia.org/w/api.php",params)
    if not data:
        return []
    results=[]
    for item in (data.get("query",{}).get("search") or [])[:limit]:
        title=item.get("title","")
        snippet=re.sub(r"<[^>]+>","",item.get("snippet","") or "")
        url=f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(title)}"
        results.append(_source("wikipedia",title,snippet,url=url))
    return results

async def search_arxiv(query,limit=WEB_SEARCH_MAX_RESULTS):
    #arXiv Atom接口：按官方要求间隔>=3秒，XML解析全量兜底
    global _arxiv_last
    async with _arxiv_lock:
        wait=ARXIV_MIN_INTERVAL-(time.time()-_arxiv_last)
        if wait>0:
            await asyncio.sleep(wait)
        _arxiv_last=time.time()
    params={"search_query":f"all:{query}","start":"0","max_results":str(limit)}
    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as client:
                resp=await client.get("https://export.arxiv.org/api/query",params=params,
                                      headers={"User-Agent":USER_AGENT})
                resp.raise_for_status()
                root=ET.fromstring(resp.text)
    except Exception as e:
        logger.warning(f"arXiv请求失败：{e}")
        return []
    ns={"a":"http://www.w3.org/2005/Atom"}
    results=[]
    for entry in (root.findall("a:entry",ns) or [])[:limit]:
        title=(entry.findtext("a:title",default="",namespaces=ns) or "").strip()
        summary=(entry.findtext("a:summary",default="",namespaces=ns) or "").strip()
        link=(entry.findtext("a:id",default="",namespaces=ns) or "").strip()
        arxiv_id=re.sub(r"^https?://arxiv\.org/abs/","",link)
        results.append(_source("arxiv",title,summary,url=link,arxiv_id=arxiv_id))
    return results

async def web_search(query,limit=WEB_SEARCH_MAX_RESULTS):
    #维基与arXiv并行，单源失败不影响另一源
    wiki=asyncio.create_task(search_wikipedia(query,limit))
    arxiv=asyncio.create_task(search_arxiv(query,limit))
    w,a=await asyncio.gather(wiki,arxiv,return_exceptions=True)
    w=w if isinstance(w,list) else []
    a=a if isinstance(a,list) else []
    return w+a

async def hybrid_search(query,top_k=5):
    #本地检索在线程池，外网并行；本地加载模型延迟到首次调用
    def _local():
        from rag_tool import search_papers_rerank, search_papers_structured
        if os.getenv("USE_RERANK")=="1":
            return search_papers_rerank(query,k=top_k)
        return search_papers_structured(query,k=top_k)
    local_task=asyncio.create_task(asyncio.to_thread(_local))
    web_task=asyncio.create_task(web_search(query))
    local,web=await asyncio.gather(local_task,web_task,return_exceptions=True)
    local=local if isinstance(local,list) else []
    web=web if isinstance(web,list) else []
    sources=[_source("local",r.get("source",""),r.get("text",""),
                     paper_id=r.get("source"),section=r.get("section")) for r in local]
    sources+=web
    context="\n\n".join(f"[{s['source_type']}:{s['title']}]\n{s['snippet']}" for s in sources)
    return {"sources":sources,"context":context}
