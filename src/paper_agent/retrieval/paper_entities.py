import re

#别名字典：模型与用户会用可读名字指代论文，这里统一映射回库内source名
#语料内论文名本身可以直接精确匹配，别名字典只补那些无法从名字推出来的常用叫法
PAPER_ALIASES={
    "Attention_Is_All_You_Need":["attention is all you need","transformer"],
    "BERT":["bert"],
    "Chain_of_Thought":["chain of thought","chain-of-thought"],
    "FlashAttention":["flashattention","flash attention"],
    "GraphRAG":["graphrag","graph rag"],
    "LoRA":["lora"],
    "RAG_Original_Paper":["retrieval augmented generation","rag论文","rag原始论文"],
    "ReAct":["react"],
    "GPT2":["gpt2","gpt-2","gpt"],
    "RoFormer_RoPE":["roformer","rope","rotary position","旋转位置"],
}

ALIAS_TO_SOURCE={alias:source for source,aliases in PAPER_ALIASES.items() for alias in aliases}


def _alias_hit(alias,text):
    #英文别名用词边界：连字符也视为词的一部分，避免"BERT-NER"命中原版BERT、"GPT-6"命中GPT2
    if re.search(r"[a-z]",alias):
        return bool(re.search(r"(?<![a-z0-9-])"+re.escape(alias)+r"(?![a-z0-9-])",text))
    return alias in text


def normalize_paper_name(name,valid_sources=None):
    #模型输出可读标题时映射回库内source名；按别名长度从长到短匹配
    low=(name or "").lower()
    for alias in sorted(ALIAS_TO_SOURCE,key=len,reverse=True):
        if _alias_hit(alias,low):
            return ALIAS_TO_SOURCE[alias]
    if valid_sources and name in valid_sources:
        return name
    return name


def extract_named_papers(question,corpus_sources=None):
    #两条来源：别名字典命中；语料论文名（含arxiv id）在问题里直接出现
    text=(question or "").lower()
    named=[]
    for source,keys in PAPER_ALIASES.items():
        if any(_alias_hit(k,text) for k in keys):
            named.append(source)
    if corpus_sources:
        for source in corpus_sources:
            low=source.lower()
            if low and _alias_hit(low,text) and source not in named:
                named.append(source)
        for m in re.findall(r"\b\d{4}\.\d{4,5}\b",question or ""):
            if m in corpus_sources and m not in named:
                named.append(m)
    return named
