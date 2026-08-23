from state import AgentState, Conflict, FactItem, PaperMeta, ReviewReport
from survey_agent import _finalize_review, _make_tools
from tools import build_pending_conflicts

def test_build_pending_conflicts_groups_by_entity_attribute():
    #相同实体+属性、不同值才进入待核实候选
    facts=[
        FactItem(fact_id="f1",paper_id="P1",entity="BERT",attribute="param_count",value="110M",content="c",source_chunk_id="c1",section_name="S"),
        FactItem(fact_id="f2",paper_id="P2",entity="BERT",attribute="param_count",value="340M",content="c",source_chunk_id="c2",section_name="S"),
        FactItem(fact_id="f3",paper_id="P3",entity="BERT",attribute="task",value="QA",content="c",source_chunk_id="c3",section_name="S")
    ]
    pending=build_pending_conflicts(facts)
    assert len(pending)==1
    assert pending[0]["entity"]=="BERT"
    assert pending[0]["attribute"]=="param_count"

def test_tool_preconditions(monkeypatch):
    #前置不满足时返回错误Observation，不执行真实工具
    state=AgentState()
    tools=_make_tools(state)
    assert "不在已检索论文中" in tools["read_section"]["func"]("P","S")
    assert "不在已检索论文中" in tools["build_paper_card"]["func"]("P")
    assert "fact_id不存在" in tools["verify_claim"]["func"]("f1","f2")

def test_search_updates_state(monkeypatch):
    #search写入papers并去重，search_count递增
    state=AgentState()
    monkeypatch.setattr("survey_agent._search_papers",lambda q,limit=10:{"papers":[PaperMeta(paper_id="P1",title="T1"),PaperMeta(paper_id="P1",title="T1")],"chunks":[]})
    tools=_make_tools(state)
    out=tools["search_papers"]["func"]("q")
    assert "检索完成" in out
    assert state.search_count==1
    assert len(state.papers)==1

def test_finalize_review_fills_missing_review(monkeypatch):
    #循环结束后有facts没review时补写综述，无facts不生成
    monkeypatch.setattr("survey_agent._write_review",lambda query,facts,conflicts,pending=None:ReviewReport(title="综述"))
    state=AgentState()
    _finalize_review(state,"q")
    assert state.review is None
    state.facts=[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",content="c",source_chunk_id="c1",section_name="S")]
    _finalize_review(state,"q")
    assert state.review is not None
    assert state.review.title=="综述"
