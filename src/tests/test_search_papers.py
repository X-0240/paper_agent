import json

from doc_ingest import Chunk
from state import FactItem, PaperCard
from tools import analyze_paper_relations, build_paper_card, build_search_result, read_section, verify_claim, _extract_facts, _extract_verdict, _resolve_chunk_id

def test_build_search_result_dedup_papers():
    #同一论文多片段只产生一条PaperMeta，但保留全部Chunk
    results=[
        {"source":"Attention_Is_All_You_Need","section":"Method","score":0.9,"text":"多头注意力"},
        {"source":"Attention_Is_All_You_Need","section":"Experiment","score":0.8,"text":"实验指标"},
        {"source":"BERT","section":"Method","score":0.7,"text":"双向编码器"}
    ]
    out=build_search_result(results)
    assert len(out["papers"])==2
    assert len(out["chunks"])==3
    assert all(c.parent_id for c in out["chunks"])
    assert all(c.doc_id for c in out["chunks"])

def test_build_search_result_empty():
    #检索无结果时返回空列表，不抛异常
    out=build_search_result([])
    assert out=={"papers":[],"chunks":[]}

def test_read_section_exact_fuzzy_and_cache(monkeypatch):
    #精确匹配返回章节全文，模糊匹配兜底，缓存命中后不再读章节
    sections=[{"title":"1 Introduction","text":"intro text","page":None},{"title":"3 Method","text":"method text","page":None}]
    monkeypatch.setattr("tools.load_sections",lambda paper_id:sections)
    cache={}
    assert read_section("P","1 Introduction",cache)=="intro text"
    assert read_section("P","Introduction",cache)=="intro text"
    assert read_section("P","Method",cache)=="method text"
    assert read_section("P","NotExist",cache) is None
    calls={"n":0}
    def counting(paper_id):
        calls["n"]+=1
        return sections
    monkeypatch.setattr("tools.load_sections",counting)
    read_section("P","1 Introduction",cache)
    assert calls["n"]==0

def test_read_section_fuzzy_cache_uses_real_title(monkeypatch):
    #模糊命中后，实际章节名也要写缓存，后续精确请求不再读章节
    sections=[{"title":"1 Introduction","text":"intro text","page":None}]
    monkeypatch.setattr("tools.load_sections",lambda paper_id:sections)
    cache={}
    read_section("P","Introductin",cache)
    calls={"n":0}
    monkeypatch.setattr("tools.load_sections",lambda paper_id:(calls.__setitem__("n",calls["n"]+1) or sections))
    assert read_section("P","1 Introduction",cache)=="intro text"
    assert calls["n"]==0

def test_read_section_truncate_at_sentence_boundary(monkeypatch):
    #截断点尽量停在句子边界，不切在半句话
    long_text="第一句"*30+"。"+"第二句很长"*200
    sections=[{"title":"1 Introduction","text":long_text,"page":None}]
    monkeypatch.setattr("tools.load_sections",lambda paper_id:sections)
    text=read_section("P","1 Introduction",max_chars=100)
    assert len(text)<=100
    assert text.endswith("。")

def test_build_paper_card_maps_fields_and_cache(monkeypatch):
    #旧卡片字段映射到新PaperCard，缓存命中后不再调LLM
    monkeypatch.setattr("tools.old_build_card",lambda paper_id:{
        "title":"Attention Is All You Need",
        "background":"解决序列转换问题",
        "method":"多头注意力",
        "innovation":["完全基于注意力"],
        "limitations":"未提及"
    })
    monkeypatch.setattr("tools.load_sections",lambda paper_id:[{"title":"1 Introduction","text":"intro","page":None},{"title":"3 Method","text":"method","page":None}])
    cache={}
    card=build_paper_card("P",cache)
    assert card.title=="Attention Is All You Need"
    assert card.key_findings==["完全基于注意力"]
    assert card.limitations==[]
    assert len(card.sections_ref)==2
    calls={"n":0}
    monkeypatch.setattr("tools.old_build_card",lambda paper_id:(calls.__setitem__("n",calls["n"]+1) or {}))
    build_paper_card("P",cache)
    assert calls["n"]==0

def test_analyze_paper_relations_filters_unresolvable_facts(monkeypatch):
    #无paper_id/无章节/来源解析不到的事实全部丢弃
    monkeypatch.setattr("tools.compare_papers",lambda paper_ids:"对比矩阵")
    monkeypatch.setattr("tools.build_paper_card",lambda paper_id,cache=None:PaperCard(paper_id=paper_id,title=paper_id))
    monkeypatch.setattr("tools.safe_call_deepseek",lambda messages,**kw:{"choices":[{"message":{"content":json.dumps([
        {"paper_id":"P1","entity":"BERT","attribute":"param_count","value":"110M","content":"参数量","section_name":"1 Introduction","raw_quote":"quote"},
        {"paper_id":"P1","entity":"BERT","attribute":"type","value":"encoder","content":"类型","section_name":""},
        {"paper_id":"P999","entity":"X","attribute":"y","value":"z","content":"bad","section_name":"Method","raw_quote":"q"}
    ])}}]})
    monkeypatch.setattr("tools._resolve_chunk_id",lambda pid,sec:"P1::1 Introduction::0" if pid=="P1" and sec=="1 Introduction" else None)
    out=analyze_paper_relations(["P1","P2"])
    assert out["comparison"]=="对比矩阵"
    assert len(out["facts"])==1
    assert out["facts"][0].source_chunk_id=="P1::1 Introduction::0"

def test_resolve_chunk_id_exact_and_fuzzy(monkeypatch):
    #章节名映射到真实chunk_id：精确优先，模糊兜底
    fake=[Chunk(chunk_id="P::1 Introduction::0",doc_id="P",section_name="1 Introduction",text="intro",token_len=1,page_num=None,parent_id="P::1 Introduction",position="")]
    monkeypatch.setattr("tools.load_sections",lambda paper_id:[{"title":"1 Introduction","text":"intro"}])
    monkeypatch.setattr("tools.chunk_splitter",lambda record,chunk_tokens=800,overlap=100:fake)
    assert _resolve_chunk_id("P","1 Introduction")=="P::1 Introduction::0"
    assert _resolve_chunk_id("P","Introduction")=="P::1 Introduction::0"
    assert _resolve_chunk_id("P","NotExist") is None

def test_extract_facts_retry_on_bad_json(monkeypatch):
    #LLM第一次输出非法JSON时带纠正指令重试一次
    calls={"n":0}
    def fake(messages,**kw):
        calls["n"]+=1
        if calls["n"]==1:
            return {"choices":[{"message":{"content":"不是JSON"}}]}
        #第二次必须能看到自己第一次的错误输出
        assert any(m.get("role")=="assistant" and m.get("content")=="不是JSON" for m in messages)
        return {"choices":[{"message":{"content":json.dumps([{"paper_id":"P1","entity":"E","attribute":"A","value":"V","content":"C","section_name":"S","raw_quote":"q"}])}}]}
    monkeypatch.setattr("tools.safe_call_deepseek",fake)
    items=_extract_facts("cards")
    assert len(items)==1
    assert calls["n"]==2

def test_extract_json_array_accepts_object(monkeypatch):
    #LLM返回对象或含facts字段的对象时也能解析成事实列表
    from tools import _extract_json_array
    assert _extract_json_array('{"facts":[{"entity":"E"}]}')==[{"entity":"E"}]
    assert _extract_json_array('{"entity":"E","attribute":"A"}')==[{"entity":"E","attribute":"A"}]

def test_fact_id_not_duplicated(monkeypatch):
    #多次调用analyze_paper_relations时fact_id不能重复
    monkeypatch.setattr("tools.compare_papers",lambda paper_ids:"对比矩阵")
    monkeypatch.setattr("tools.build_paper_card",lambda paper_id,cache=None:PaperCard(paper_id=paper_id,title=paper_id))
    monkeypatch.setattr("tools.safe_call_deepseek",lambda messages,**kw:{"choices":[{"message":{"content":json.dumps([{"paper_id":"P1","entity":"E","attribute":"A","value":"V","content":"C","section_name":"S","raw_quote":"q"}])}}]})
    monkeypatch.setattr("tools._resolve_chunk_id",lambda pid,sec:"chunk")
    out1=analyze_paper_relations(["P1"])
    out2=analyze_paper_relations(["P1"])
    assert out1["facts"][0].fact_id!=out2["facts"][0].fact_id

def test_verify_claim_validates_category(monkeypatch):
    #非法category回退insufficient_evidence，合法category保留
    monkeypatch.setattr("tools.read_section",lambda pid,sec,cache=None,max_chars=4000:"原文证据")
    mode={"n":0}
    def fake(messages,**kw):
        mode["n"]+=1
        category="bad_category" if mode["n"]==1 else "true_conflict"
        if mode["n"]==3:
            category="superseded"
        return {"choices":[{"message":{"content":json.dumps({"category":category,"title":"标题","detail":"详细结论","evidence_supplementary":"证据"})}}]}
    monkeypatch.setattr("tools.safe_call_deepseek",fake)
    fa=FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",content="c1",source_chunk_id="c1",section_name="S")
    fb=FactItem(fact_id="f2",paper_id="P2",entity="E",attribute="A",value="2",content="c2",source_chunk_id="c2",section_name="S")
    bad=verify_claim(fa,fb)
    assert bad.category=="insufficient_evidence"
    ok=verify_claim(fa,fb)
    assert ok.category=="true_conflict"
    assert ok.description=="标题"
    assert ok.verdict=="详细结论"
    assert ok.description!=ok.verdict
    newer=verify_claim(fa,fb)
    assert newer.category=="superseded"
    assert ok.fact_ids==["f1","f2"]
    assert bad.conflict_id!=ok.conflict_id

def test_extract_verdict_skips_explanation_object():
    #LLM先输出解释对象再输出裁决对象时，只取带category的裁决
    text='先解释：{"note":"说明"} 最终结论：{"category":"true_conflict","title":"t","detail":"d"}'
    assert _extract_verdict(text)=={"category":"true_conflict","title":"t","detail":"d"}
