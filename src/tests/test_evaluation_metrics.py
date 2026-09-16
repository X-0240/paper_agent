from evaluation.metrics import norm,evidence_hit,wilson_ci

def test_norm_handles_ligature_and_math_symbols():
    #连字和数学斜体要归一化后再比较
    assert norm("ﬁne-tuning")==norm("fine-tuning")
    assert norm("ℎ𝑒𝑙𝑙𝑜")==norm("hello")

def test_evidence_hit():
    retrieved=[{"source":"P","section":"S","text":"We use the BooksCorpus and English Wikipedia."}]
    assert evidence_hit(retrieved,["BooksCorpus and English Wikipedia"])
    assert not evidence_hit(retrieved,["something not present"])

def test_wilson_ci_bounds():
    lo,hi=wilson_ci(80,100)
    assert 0.70<lo<0.72
    assert 0.86<hi<0.88
