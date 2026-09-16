from rerank_policy import RerankCache,cache_key,should_rerank

def test_should_rerank_by_margin():
    #分差大信任粗排，分差小才触发重排
    assert not should_rerank([("a",0.9),("b",0.5),("c",0.4),("d",0.3),("e",0.2)],5,0.1)
    assert should_rerank([("a",0.6),("b",0.58),("c",0.56),("d",0.55),("e",0.54)],5,0.1)

def test_should_rerank_single_item():
    assert not should_rerank([("a",0.9)],5,0.1)

def test_rerank_cache_lru():
    cache=RerankCache(max_size=2)
    cache.set("a",[1]); cache.set("b",[2])
    assert cache.get("a")==[1]
    cache.set("c",[3])
    assert cache.get("b") is None
    assert cache.get("a")==[1] and cache.get("c")==[3]

def test_cache_key_includes_snapshot():
    assert cache_key("Transformer","snap1")!=cache_key("Transformer","snap2")
