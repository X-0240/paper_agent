import os
#国内网络直连HuggingFace可能超时，必须在import sentence_transformers前设置镜像
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import json
import logging
import time
from difflib import SequenceMatcher
from dotenv import load_dotenv
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from sentence_transformers import SentenceTransformer
from rerank import rerank

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

#论文库：FAISS索引+元数据+多语言模型（路径从.env，模块加载一次）
FAISS_PATH=os.getenv("FAISS_PATH")
MODEL_PATH=os.getenv("MODEL_PATH")
index=faiss.read_index(FAISS_PATH+".faiss")
meta=json.load(open(FAISS_PATH+".json",encoding="utf-8"))
sources=meta["sources"]; sections=meta["sections"]; documents=meta["documents"]
model=SentenceTransformer(MODEL_PATH)

SECTIONS_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"papers_sections")

def read_section(source,section,max_chars=4000):
    #旧接口适配：全文展开统一收敛到tools.read_section，保持原错误语义
    from tools import read_section as _read
    result=_read(source,section,max_chars=max_chars)
    return result if result is not None else f"未找到章节：{section}"

#BM25索引：中文问题分词后构建，解决口语化/专业术语查不到的问题
tokenized_docs=[list(jieba.cut(d)) for d in documents]
bm25=BM25Okapi(tokenized_docs)

def _normalize(scores):
    #归一化到0-1，让向量分和BM25分可以加权相加
    max_s=max(scores) if scores else 0
    return [s/max_s if max_s>0 else 0 for s in scores]

def _vector_search(query,k):
    #向量检索：编码+归一化，归一化后内积=余弦相似度
    q_vec=model.encode([query])
    q_vec=q_vec/np.linalg.norm(q_vec)
    scores,idx=index.search(q_vec.astype("float32"),k)
    return scores[0],idx[0]

def _bm25_search(query,k):
    #BM25检索：分词后按关键词得分取top k
    tokenized_query=list(jieba.cut(query))
    scores=bm25.get_scores(tokenized_query)
    top=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
    return [scores[i] for i in top],top

def hybrid_search(query,k=5,alpha=0.5):
    #双路召回：向量按语义、BM25按关键词；单路失败自动降级
    vec_items=None; bm_items=None; status=[]
    try:
        vec_items=_vector_search(query,k*2)
        status.append("vector")
    except Exception as e:
        logger.warning(f"向量检索失败:{e}")
    try:
        bm_items=_bm25_search(query,k*2)
        status.append("bm25")
    except Exception as e:
        logger.warning(f"BM25检索失败:{e}")
    if not status:
        return [],"failed"
    combined={}
    #分数融合：两路分数各自归一化后加权相加，alpha控制语义/关键词占比
    if vec_items:
        scores,idx=vec_items
        for s,j in zip(_normalize(list(scores)),idx):
            combined[j]=combined.get(j,0)+alpha*s
    if bm_items:
        scores,idx=bm_items
        for s,j in zip(_normalize(list(scores)),idx):
            combined[j]=combined.get(j,0)+(1-alpha)*s
    top=sorted(combined.items(),key=lambda x:x[1],reverse=True)[:k]
    return top,"+".join(status)

def _overlap_ratio(a,b):
    #字符级相似度，判断两条切片是否来自滑动窗口重叠
    return SequenceMatcher(None,a,b).ratio()

def filter_results(items):
    #素材过滤：去短、同章节限条数、去滑动窗口重叠，防止垃圾和冗余污染下游
    cleaned=[]; kept_by_section={}
    for idx,score in items:
        doc=documents[idx].strip()
        if len(doc)<20:
            continue
        key=(sources[idx],sections[idx])
        kept_docs=kept_by_section.get(key,[])
        #同章节最多保留2条，避免一个章节淹没其他内容
        if len(kept_docs)>=2:
            continue
        #和同章节已保留片段重叠度过高，视为滑动窗口冗余，丢弃低分者
        if any(_overlap_ratio(doc,old)>0.6 for old in kept_docs):
            continue
        kept_docs.append(doc)
        kept_by_section[key]=kept_docs
        cleaned.append((idx,score))
    return cleaned

def search_papers_hybrid(query,k=5,alpha=0.5):
    start=time.time()
    top,status=hybrid_search(query,k,alpha)
    top=filter_results(top)
    if status=="failed":
        return "检索失败：向量与关键词检索均不可用，请检查索引或模型配置"
    if not top:
        return "未找到相关论文"
    #拼接工具返回：截断300字控制token，带来源和章节便于引用
    lines=[f"[{sources[idx]} - {sections[idx]}]（相关度{score:.3f}）{documents[idx][:300]}" for idx,score in top]
    logger.info(f"混合检索 query={query} status={status} 命中={len(top)} 耗时={time.time()-start:.2f}s")
    return "\n".join(lines)

def search_papers_structured(query,k=5,alpha=0.5):
    #结构化检索结果：供流水线/Agent-1使用，返回论文名、章节、得分、片段
    top,status=hybrid_search(query,k,alpha)
    top=filter_results(top)
    if status=="failed" or not top:
        return []
    results=[{
        "source":sources[idx],
        "section":sections[idx],
        "score":round(float(score),3),
        "text":documents[idx][:300]
    } for idx,score in top]
    logger.info(f"结构化检索 query={query} status={status} 命中={len(results)}")
    return results

def search_papers_rerank(query,k=5,alpha=0.5,candidates=25):
    #工业标配：Top-25粗召回 → Cross-Encoder精排 → 取Top-k
    top,status=hybrid_search(query,candidates,alpha)
    top=filter_results(top)
    if status=="failed" or not top:
        return []
    items=[{
        "source":sources[idx],
        "section":sections[idx],
        "score":round(float(score),3),
        "text":documents[idx][:300]
    } for idx,score in top]
    items=rerank(query,items,top_n=k)
    logger.info(f"Rerank检索 query={query} 候选={len(top)} 输出={len(items)}")
    return items

def search_papers_rerank_text(query,k=5,alpha=0.5,candidates=25):
    #Rerank结果的文本版，供Agent作为工具观察
    items=search_papers_rerank(query,k,alpha,candidates)
    if not items:
        return "未找到相关论文"
    return "\n".join(f"[{it['source']} - {it['section']}]（相关度{it['score']:.3f}）{it['text']}" for it in items)
