from state import AgentState, Conflict, FactItem, PaperCard, PaperMeta, ReviewReport
from survey_agent import _auto_finalize, _enrich_trace, _finalize_review, _make_tools, review_to_markdown
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

def test_contract_fields_present():
    #State必须带编排层依赖的字段，防止契约漂移
    state=AgentState()
    assert hasattr(state,"search_count")
    assert hasattr(state,"pending_conflicts")
    assert hasattr(state,"card_cache")
    assert len(_make_tools(state))==6

def test_write_review_observation_has_stats(monkeypatch):
    #Observation返回综述统计信息，让Agent能判断是否补充
    monkeypatch.setattr("survey_agent._write_review",lambda query,facts,conflicts,pending=None:ReviewReport(
        title="综述",consensus=["c1","c2"],disagreements=[{"x":1}],superseded_conclusions=["s"]
    ))
    state=AgentState(facts=[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",content="c",source_chunk_id="c1",section_name="S")])
    tools=_make_tools(state)
    out=tools["write_review"]["func"]("q")
    assert "2条共识" in out
    assert "1条分歧" in out
    assert "1条被推翻结论" in out

def test_enrich_trace_fills_audit_fields():
    #审计快照必须包含搜索数/步数/facts/conflicts计数
    state=AgentState(search_count=2,facts=[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",content="c",source_chunk_id="c1",section_name="S")])
    entry={"step":3,"action":"Search","input":{"q":"x"},"observation":"ok"}
    _enrich_trace(entry,state)
    assert entry["search_count"]==2
    assert entry["step_count"]==3
    assert entry["facts_count"]==1
    assert entry["conflicts_count"]==0

def test_auto_finalize_analyzes_when_cards_exist(monkeypatch):
    #循环结束后有卡片没事实，编排层自动补事实抽取和综述
    monkeypatch.setattr("survey_agent._analyze",lambda ids,cache=None:{"comparison":"c","facts":[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",content="c",source_chunk_id="c1",section_name="S")]})
    monkeypatch.setattr("survey_agent._write_review",lambda query,facts,conflicts,pending=None:ReviewReport(title="兜底综述"))
    state=AgentState(cards=[PaperCard(paper_id="P1",title="T1")])
    _auto_finalize(state,"q")
    assert state.review is not None
    assert state.review.title=="兜底综述"

def test_review_to_markdown_renders_sections():
    #综述对象渲染为Markdown，引用必须出现
    report=ReviewReport(title="综述",consensus=["共识1"],references=["P1, 2020, §c1"])
    md=review_to_markdown(report)
    assert "## 核心共识" in md
    assert "§c1" in md
