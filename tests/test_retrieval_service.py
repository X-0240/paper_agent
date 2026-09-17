from retrieval_service import RetrievalService,_valid_retrieval_query


def make_service(monkeypatch,**env):
    default={
        "APP_MODE":"production",
        "QUERY_GENERATION_ENABLED":"0",
        "RETRIEVAL_CANDIDATES":"25",
        "RERANK_MODE":"conditional",
        "QUERY_CACHE_FILE":"",
    }
    default.update(env)
    for key,value in default.items():
        monkeypatch.setenv(key,value)
    return RetrievalService(
        faiss_path="unused",
        model_path="unused",
        index=object(),
        meta={"sources":["P"],"sections":["S"],"documents":["evidence text"]}
    )


def test_query_generation_disabled_uses_original(monkeypatch):
    #默认保护：未显式开启时不调用LLM，直接使用原问题
    service=make_service(monkeypatch)
    query,cache_hit,fallback=service._build_retrieval_query("什么是Transformer？")
    assert query=="什么是Transformer？"
    assert not cache_hit
    assert not fallback


def test_injected_search_query_is_used(monkeypatch):
    #评测可以注入预生成query，但生产默认只走原问题或内部生成
    service=make_service(monkeypatch)
    query,cache_hit,fallback=service._build_retrieval_query("问题","offline english query")
    assert query=="offline english query"
    assert not cache_hit
    assert not fallback


def test_invalid_generated_query_falls_back(monkeypatch):
    #含中文的生成结果非法，必须回退原问题
    service=make_service(monkeypatch,QUERY_GENERATION_ENABLED="1")
    monkeypatch.setattr(service,"_generate_query",lambda question:"这是中文解释")
    query,cache_hit,fallback=service._build_retrieval_query("什么是Transformer？")
    assert query=="什么是Transformer？"
    assert not cache_hit
    assert fallback=="invalid_generated_query"


def test_search_deduplicates_and_preserves_chunk_id(monkeypatch):
    #同一chunk重复召回时只保留一条，并透传稳定chunk_id
    service=make_service(monkeypatch)
    monkeypatch.setattr(service,"_recall",lambda query,k,alpha=None: ([(0,1.0),(0,0.8)],"mock"))
    items=service.search("query",search_query="query",candidate_k=2,rerank_mode="none",top_k=5)
    assert len(items)==1
    assert items[0]["chunk_id"]==service.chunk_ids[0]
    assert items[0]["final_rank"]==1
    assert items[0]["text"]=="evidence text"


def test_retrieval_query_validation():
    #查询生成校验只接受单行英文检索query
    assert _valid_retrieval_query("What mechanism does Transformer use?")
    assert not _valid_retrieval_query("这里包含中文")
    assert not _valid_retrieval_query("Explanation: this is a long answer")
    assert not _valid_retrieval_query("first line\nsecond line")
