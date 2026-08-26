import asyncio
import unittest.mock as mock
import httpx
import web_search

def _wiki_resp():
    resp=mock.MagicMock()
    resp.raise_for_status=mock.MagicMock()
    resp.json.return_value={"query":{"search":[
        {"title":"Transformer (machine learning model)","snippet":"<b>Transformer</b> is a model","pageid":123}
    ]}}
    return resp

def _arxiv_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
    <summary>We introduce BERT.</summary>
    <id>http://arxiv.org/abs/1810.04805v2</id>
  </entry>
</feed>"""

def test_wikipedia_returns_sources():
    #维基正常返回：解析成统一source结构
    async def run():
        with mock.patch("web_search.httpx.AsyncClient.get",return_value=_wiki_resp()):
            return await web_search.search_wikipedia("Transformer",limit=1)
    res=asyncio.run(run())
    assert len(res)==1
    assert res[0]["source_type"]=="wikipedia"
    assert "Transformer" in res[0]["title"]
    assert res[0]["url"].startswith("https://zh.wikipedia.org/wiki/")

def test_wikipedia_http_error_returns_empty():
    #429等HTTP错误：降级为空列表，不抛异常
    resp=mock.MagicMock()
    resp.raise_for_status=mock.MagicMock(side_effect=httpx.HTTPStatusError(
        "rate",request=httpx.Request("GET","http://x"),response=httpx.Response(429)))
    async def run():
        with mock.patch("web_search.httpx.AsyncClient.get",return_value=resp):
            return await web_search.search_wikipedia("BERT",limit=1)
    assert asyncio.run(run())==[]

def test_arxiv_parses_entries():
    #arXiv XML正常解析
    resp=mock.MagicMock()
    resp.raise_for_status=mock.MagicMock()
    resp.text=_arxiv_xml()
    async def run():
        with mock.patch("web_search.httpx.AsyncClient.get",return_value=resp):
            return await web_search.search_arxiv("BERT",limit=1)
    res=asyncio.run(run())
    assert len(res)==1
    assert res[0]["source_type"]=="arxiv"
    assert res[0]["arxiv_id"]=="1810.04805v2"

def test_web_search_merges_both_sources():
    #维基+arXiv并行合并
    async def fake_get(url,params=None,headers=None):
        resp=mock.MagicMock()
        resp.raise_for_status=mock.MagicMock()
        if "wikipedia" in url:
            resp.json.return_value={"query":{"search":[{"title":"Transformer","snippet":"text","pageid":1}]}}
        else:
            resp.text=_arxiv_xml()
            resp.json.side_effect=Exception("not json")
        return resp
    async def run():
        with mock.patch("web_search.httpx.AsyncClient.get",side_effect=fake_get):
            return await web_search.web_search("BERT",limit=1)
    res=asyncio.run(run())
    types={r["source_type"] for r in res}
    assert types=={"wikipedia","arxiv"}

def test_semaphore_limits_concurrency():
    #并发上限：10个任务同时跑，活跃协程不超过配置值
    async def run():
        active=0
        peak=0
        async def worker():
            nonlocal active,peak
            async with web_search._get_semaphore():
                active+=1
                peak=max(peak,active)
                await asyncio.sleep(0.05)
                active-=1
        await asyncio.gather(*[worker() for _ in range(10)])
        return peak
    assert asyncio.run(run())<=web_search.WEB_SEARCH_CONCURRENCY
