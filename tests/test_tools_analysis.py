import json

from state import FactItem, PaperCard
from tools import analyze_paper_relations, verify_claim, write_review, _extract_facts, _extract_verdict

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

def test_write_review_generates_references(monkeypatch):
    #引用从facts确定性生成，LLM只负责内容，不编引用
    def fake(messages,**kw):
        #payload必须是完整合法JSON，不能截断
        payload=json.loads(messages[-1]["content"])
        assert "facts" in payload and "conflicts" in payload
        return {"choices":[{"message":{"content":json.dumps({
            "title":"综述","consensus":["共识"],"disagreements":[],"superseded_conclusions":[],"open_questions":[],"conflict_mark_list":[]
        })}}]}
    monkeypatch.setattr("tools.safe_call_deepseek",fake)
    facts=[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",content="c",source_chunk_id="c1",section_name="S",year=2020)]
    out=write_review("问题",facts,[])
    assert out.title=="综述"
    assert out.references==["P1, 2020, §c1"]

def test_write_review_fallback_on_bad_json(monkeypatch):
    #LLM输出非法JSON时返回空结构综述，标题回退为query，引用仍保留
    monkeypatch.setattr("tools.safe_call_deepseek",lambda messages,**kw:{"choices":[{"message":{"content":"不是JSON"}}]})
    facts=[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",content="c",source_chunk_id="c1",section_name="S")]
    out=write_review("问题",facts,[])
    assert out.title=="问题"
    assert out.references==["P1, N/A, §c1"]
