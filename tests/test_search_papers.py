from tools import build_search_result

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
