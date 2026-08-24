import json
import logging
import os
import re
import sys
import time
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from llm_api import safe_call_deepseek

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

FAISS_PATH=os.getenv("FAISS_PATH")
MODEL_PATH=os.getenv("MODEL_PATH")
CACHE_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"eval_query_transform_cache.json")
TRANSFORMS=os.getenv("QUERY_TRANSFORMS","all").split(",")

TRANSLATE_PROMPT=(
    "你是专业英文学术翻译。把用户的中文科研问题翻译成用于论文检索的英文query。"
    "规则：只输出英文翻译本身，禁止输出中文、解释、引号或任何额外内容。\n"
    "示例：\n"
    "中文：什么是Transformer？\n"
    "英文：What is Transformer?\n"
    "中文：BERT的参数量是多少？\n"
    "英文：What is the parameter count of BERT?"
)
REWRITE_PROMPT="把下面的问题改写为信息更完整、更利于论文检索的中文query，只输出改写结果，不要解释。"

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

def norm_text(text):
    return re.sub(r"\s+"," ",text).lower()

def section_match(section,expected_section):
    def norm(s):
        s=re.sub(r"^[\d.]+\s*","",s.lower())
        s=re.sub(r"^(figure|table|abstract|appendix|proposition|theorem|section)\s*[\d.]*\s*","",s)
        return re.sub(r"[^a-z ]","",s).strip()
    e=norm(expected_section)
    a=norm(section)
    if not e or not a:
        return False
    return e in a or a in e

def chapter_hit(idxs,truth_source,truth_sections):
    for j in idxs:
        if sources[j]!=truth_source:
            continue
        for sec in truth_sections:
            if section_match(sections[j],sec):
                return True
    return False

def weighted_candidates(query,candidates=25,alpha=0.5):
    qv=model.encode([query])
    qv=qv/np.linalg.norm(qv)
    vs,vi=index.search(qv.astype("float32"),candidates)
    bm_scores=bm25.get_scores(tokenize(query))
    bm_idx=sorted(range(len(bm_scores)),key=lambda i:bm_scores[i],reverse=True)[:candidates]
    combined={}
    vs=vs[0]; vi=vi[0]
    if max(vs)>0:
        vs=vs/max(vs)
    for s,j in zip(vs,vi):
        combined[j]=combined.get(j,0)+alpha*float(s)
    bm_scores=np.array([bm_scores[j] for j in bm_idx])
    if max(bm_scores)>0:
        bm_scores=bm_scores/max(bm_scores)
    for s,j in zip(bm_scores,bm_idx):
        combined[j]=combined.get(j,0)+(1-alpha)*float(s)
    return sorted(combined.items(),key=lambda x:x[1],reverse=True)[:candidates]

def base_top(query,top_n=5):
    cands=weighted_candidates(query,10,0.5)
    return [j for j,_ in cands[:top_n]]

def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH,encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_PATH,"w",encoding="utf-8") as f:
        json.dump(cache,f,ensure_ascii=False)

def valid_english(text):
    #英文翻译校验：不能含中文，且要完整到句尾问号或足够长
    if re.search(r"[\u4e00-\u9fff]",text):
        return False
    return len(text)>=25 or text.endswith("?")

def llm_transform(question,prompt,cache,prefix,require_english=False):
    #缓存键带前缀，避免不同变换互相覆盖
    key=f"{prefix}|{question}"
    if key in cache:
        return cache[key]
    text=question
    for attempt in range(3):
        try:
            result=safe_call_deepseek(
                [{"role":"system","content":prompt},{"role":"user","content":question}],
                temperature=0.1,
                max_tokens=200
            )
            text=result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"变换失败，回退原问题：{e}")
            text=question
            break
        if require_english and not valid_english(text):
            logger.warning(f"第{attempt+1}次翻译不合规，重试")
            continue
        break
    cache[key]=text if text else question
    save_cache(cache)
    return cache[key]

#加载索引与模型
meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
chunks=meta["documents"]; sources=meta["sources"]; sections=meta["sections"]
logger.info(f"索引切片：{len(chunks)}")
model=SentenceTransformer(MODEL_PATH)
emb=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=True)
index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))
bm25=BM25Okapi([tokenize(c) for c in chunks])

#中文100题
paper_map={
    "Attention Is All You Need":"Attention_Is_All_You_Need","BERT":"BERT",
    "Chain-of-Thought":"Chain_of_Thought","FlashAttention":"FlashAttention","GraphRAG":"GraphRAG",
    "LoRA":"LoRA","RAG Original":"RAG_Original_Paper","ReAct":"ReAct","GPT2":"GPT2",
    "GPT-2":"GPT2","RoFormer":"RoFormer_RoPE",
}
q_file=os.getenv("TEST_QUESTIONS_FILE")
questions_main=re.findall(r"问题：(.+)",open(q_file,encoding="utf-8").read())
v_content=open(os.getenv("TEST_REFERENCES_FILE"),encoding="utf-8").read()
fixed_refs=[]
for block in re.split(r"(?=第\d+题)",v_content):
    if not block.strip():
        continue
    m=re.search(r"修正后引用：(.*)",block)
    if m and m.group(1).strip():
        fixed_refs.append(m.group(1).strip())
    else:
        m2=re.search(r"原始引用：(.*)",block)
        if m2:
            fixed_refs.append(m2.group(1).strip())
test_cases=[]
for q,ref in zip(questions_main,fixed_refs):
    parts=re.split(r"\s+-\s+",ref,maxsplit=1)
    if len(parts)<2:
        continue
    paper_part,section_part=parts
    source=None
    for key,src in paper_map.items():
        if key.lower() in paper_part.lower():
            source=src; break
    if source:
        test_cases.append((q,source,[section_part]))
logger.info(f"中文测试题：{len(test_cases)}")

cache=load_cache()
variants={
    "原文":None,
    "翻译成英文":TRANSLATE_PROMPT,
    "中文改写":REWRITE_PROMPT,
    "改写+翻译":None,
}

transformed={}
for q,_,_ in test_cases:
    transformed.setdefault("原文",[]).append(q)
    if "all" in TRANSFORMS or "translate" in TRANSFORMS:
        transformed.setdefault("翻译成英文",[]).append(llm_transform(q,TRANSLATE_PROMPT,cache,"translate",require_english=True))
    if "all" in TRANSFORMS or "rewrite" in TRANSFORMS or "rewrite_translate" in TRANSFORMS:
        rewritten=llm_transform(q,REWRITE_PROMPT,cache,"rewrite")
        transformed.setdefault("中文改写",[]).append(rewritten)
    if "all" in TRANSFORMS or "rewrite_translate" in TRANSFORMS:
        transformed.setdefault("改写+翻译",[]).append(llm_transform(rewritten,TRANSLATE_PROMPT,cache,"rewrite_translate"))

t0=time.time()
for name,queries in transformed.items():
    hits=0
    for (q,source,truth_sections),tq in zip(test_cases,queries):
        hits+=chapter_hit(base_top(tq),source,truth_sections)
    print(f"{name}：章节级@5={hits/len(test_cases):.1%}（{hits}/{len(test_cases)}）")
print(f"评测耗时：{time.time()-t0:.1f}s")
