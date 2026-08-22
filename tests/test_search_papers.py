from tools import build_search_result, read_section

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
    sections=[("1 Introduction","intro text"),("3 Method","method text")]
    monkeypatch.setattr("agent2_parse.get_paper_sections",lambda paper_id:sections)
    cache={}
    assert read_section("P","1 Introduction",cache)=="intro text"
    assert read_section("P","Introduction",cache)=="intro text"
    assert read_section("P","Method",cache)=="method text"
    assert read_section("P","NotExist",cache) is None
    calls={"n":0}
    def counting(paper_id):
        calls["n"]+=1
        return sections
    monkeypatch.setattr("agent2_parse.get_paper_sections",counting)
    read_section("P","1 Introduction",cache)
    assert calls["n"]==0
