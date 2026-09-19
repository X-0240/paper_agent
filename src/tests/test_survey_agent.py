from state import AgentState, Conflict, FactItem, PaperCard, PaperMeta, ReviewReport
from survey_agent import (TOOL_LABELS,_auto_finalize, _enrich_trace, _finalize_review, _make_tools,
                          _merge_cards, _progress_text, _record_tool_call, review_to_markdown, run_survey)
from tools import build_pending_conflicts


def test_record_tool_call_uses_run_context(tmp_path,monkeypatch):
    #设置任务上下文后，工具调用必须落到对应任务上
    import store
    from run_context import reset_run_context,set_run_context
    monkeypatch.setenv("SQLITE_PATH",str(tmp_path/"a.db"))
    store.close_conn()
    store.init_db()
    thread_id=store.create_thread("alice")
    task_id=store.create_task(thread_id,"alice","survey")
    token=set_run_context(task_id,thread_id,"alice")
    try:
        _record_tool_call({"action":"search_papers","action_input":{"query":"x"},
                           "observation":"检索完成","duration_ms":12})
    finally:
        reset_run_context(token)
    attr=store.task_attribution(task_id,"alice")
    assert attr["tool_calls_total"]==1
    assert attr["tools"][0]["tool"]=="search_papers" and attr["tools"][0]["duration_ms"]==12
    store.close_conn()


def test_merge_cards_dedupes_by_paper_id():
    #工具内部建的卡片并回状态时要按论文去重，避免重复计数
    state=AgentState(query="t",complexity="COMPLEX")
    _merge_cards(state,[PaperCard(paper_id="P1",title="a"),PaperCard(paper_id="P1",title="b"),
                        PaperCard(paper_id="P2",title="c")])
    assert [c.paper_id for c in state.cards]==["P1","P2"]


def test_all_tools_have_progress_label():
    #每个注册工具都必须有中文进度文案，否则流式进度会直接露出原始工具名
    tools=_make_tools(AgentState(query="t",complexity="COMPLEX"))
    assert set(tools)<=set(TOOL_LABELS)


def test_progress_text_carries_state_counts():
    state=AgentState(query="t",complexity="COMPLEX")
    text=_progress_text({"action":"search_papers"},state)
    assert "检索候选论文" in text and "论文0篇" in text

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

def test_auto_finalize_covers_papers_without_cards(monkeypatch):
    #模型提前收尾时可能只检索到论文、没建卡片：兜底也要能补出事实与综述
    monkeypatch.setattr("survey_agent._analyze",lambda ids,cache=None:{
        "comparison":"c",
        "cards":[PaperCard(paper_id="P1",title="T1")],
        "facts":[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",
                          content="c",source_chunk_id="c1",section_name="S")]})
    monkeypatch.setattr("survey_agent._write_review",lambda query,facts,conflicts,pending=None:ReviewReport(title="兜底综述"))
    state=AgentState(papers=[PaperMeta(paper_id="P1",title="T1")])
    _auto_finalize(state,"q")
    assert state.review is not None
    assert [c.paper_id for c in state.cards]==["P1"]
    assert len(state.facts)==1


def test_run_survey_forces_search_when_no_papers(monkeypatch):
    #C类兜底：循环结束一篇论文都没有时，用原问题强制检索一次再走兜底
    monkeypatch.setattr("survey_agent.react",lambda *a,**kw:("",[]))
    monkeypatch.setattr("survey_agent._search_papers",
                        lambda query,limit=3:{"papers":[PaperMeta(paper_id="P1",title="T1")],"chunks":[]})
    monkeypatch.setattr("survey_agent._analyze",lambda ids,cache=None:{
        "comparison":"c","cards":[PaperCard(paper_id="P1",title="T1")],
        "facts":[FactItem(fact_id="f1",paper_id="P1",entity="E",attribute="A",value="1",
                          content="c",source_chunk_id="c1",section_name="S")]})
    monkeypatch.setattr("survey_agent._write_review",lambda query,facts,conflicts,pending=None:ReviewReport(title="兜底综述"))
    state=run_survey("对比 A 和 B")
    assert [p.paper_id for p in state.papers]==["P1"]
    assert state.review is not None


def test_write_review_without_facts_guides_next_action():
    #错误提示必须给出下一步动作，否则模型容易直接放弃并判证据不足
    state=AgentState(query="t",complexity="COMPLEX")
    msg=_make_tools(state)["write_review"]["func"](query="q")
    assert "analyze_paper_relations" in msg


def test_review_to_markdown_renders_sections():
    #综述对象渲染为Markdown，引用必须出现
    report=ReviewReport(title="综述",consensus=["共识1"],references=["P1, 2020, §c1"])
    md=review_to_markdown(report)
    assert "## 核心共识" in md
    assert "§c1" in md
