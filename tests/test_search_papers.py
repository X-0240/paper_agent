from tools import build_paper_card, build_search_result, read_section

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
    monkeypatch.setattr("tools._load_sections",lambda paper_id:sections)
    cache={}
    assert read_section("P","1 Introduction",cache)=="intro text"
    assert read_section("P","Introduction",cache)=="intro text"
    assert read_section("P","Method",cache)=="method text"
    assert read_section("P","NotExist",cache) is None
    calls={"n":0}
    def counting(paper_id):
        calls["n"]+=1
        return sections
    monkeypatch.setattr("tools._load_sections",counting)
    read_section("P","1 Introduction",cache)
    assert calls["n"]==0

def test_read_section_fuzzy_cache_uses_real_title(monkeypatch):
    #模糊命中后，实际章节名也要写缓存，后续精确请求不再读章节
    sections=[{"title":"1 Introduction","text":"intro text","page":None}]
    monkeypatch.setattr("tools._load_sections",lambda paper_id:sections)
    cache={}
    read_section("P","Introductin",cache)
    calls={"n":0}
    monkeypatch.setattr("tools._load_sections",lambda paper_id:(calls.__setitem__("n",calls["n"]+1) or sections))
    assert read_section("P","1 Introduction",cache)=="intro text"
    assert calls["n"]==0

def test_read_section_truncate_at_sentence_boundary(monkeypatch):
    #截断点尽量停在句子边界，不切在半句话
    long_text="第一句"*30+"。"+"第二句很长"*200
    sections=[{"title":"1 Introduction","text":long_text,"page":None}]
    monkeypatch.setattr("tools._load_sections",lambda paper_id:sections)
    text=read_section("P","1 Introduction",max_chars=100)
    assert len(text)<=100
    assert text.endswith("。")

def test_build_paper_card_maps_fields_and_cache(monkeypatch):
    #旧卡片字段映射到新PaperCard，缓存命中后不再调LLM
    monkeypatch.setattr("agent2_parse.build_paper_card",lambda paper_id:{
        "title":"Attention Is All You Need",
        "background":"解决序列转换问题",
        "method":"多头注意力",
        "innovation":["完全基于注意力"],
        "limitations":"未提及"
    })
    monkeypatch.setattr("tools._load_sections",lambda paper_id:[{"title":"1 Introduction","text":"intro","page":None},{"title":"3 Method","text":"method","page":None}])
    cache={}
    card=build_paper_card("P",cache)
    assert card.title=="Attention Is All You Need"
    assert card.key_findings==["完全基于注意力"]
    assert card.limitations==[]
    assert len(card.sections_ref)==2
    calls={"n":0}
    monkeypatch.setattr("agent2_parse.build_paper_card",lambda paper_id:(calls.__setitem__("n",calls["n"]+1) or {}))
    build_paper_card("P",cache)
    assert calls["n"]==0
