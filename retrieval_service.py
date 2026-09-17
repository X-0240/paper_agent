import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict

#国内网络直连HuggingFace可能超时，必须在加载模型前设置镜像
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from llm_api import call_deepseek_once,release_budget,reserve_budget,settle_budget
from rerank import rerank

load_dotenv()
logger=logging.getLogger(__name__)

SERVICE_VERSION="retrieval-service-v1"
QUERY_PROMPT=(
    "你是论文检索query生成器。根据中文问题生成一行英文检索query。"
    "保留实体、缩写、术语、数值、比较关系和任务目标；优先使用论文原词，不要逐字直译，不要解释。"
    "只输出一行英文query。"
)
QUERY_PROMPT_HASH=hashlib.sha1(QUERY_PROMPT.encode("utf-8")).hexdigest()[:12]


class _QueryCache:
    #查询缓存：正缓存长期，负缓存短TTL，完整key并发只生成一次
    def __init__(self,path="",negative_ttl=60):
        self.path=path
        self.negative_ttl=negative_ttl
        self._data=OrderedDict()
        self._inflight={}
        self._lock=threading.Lock()
        if self.path and os.path.exists(self.path):
            try:
                raw=json.load(open(self.path,encoding="utf-8"))
                for key,item in raw.items():
                    self._data[key]=(dict(item["value"]),float(item.get("expires_at",0)))
            except Exception as e:
                logger.warning(f"查询缓存加载失败：{e}")

    def _lookup(self,key):
        item=self._data.get(key)
        if not item:
            return None
        value,expires_at=item
        if expires_at and expires_at<time.time():
            self._data.pop(key,None)
            return None
        return value

    def get(self,key):
        with self._lock:
            return self._lookup(key)

    def set(self,key,value):
        expires_at=0.0
        if value.get("fallback_reason"):
            expires_at=time.time()+self.negative_ttl
        with self._lock:
            self._data[key]=(dict(value),expires_at)
            self._data.move_to_end(key)
            while len(self._data)>512:
                self._data.popitem(last=False)
            if self.path:
                payload={k:{"value":v[0],"expires_at":v[1]} for k,v in self._data.items()}
                os.makedirs(os.path.dirname(os.path.abspath(self.path)),exist_ok=True)
                with open(self.path,"w",encoding="utf-8") as f:
                    json.dump(payload,f,ensure_ascii=False,indent=2)

    def get_or_create(self,key,factory):
        value=self.get(key)
        if value:
            return value
        owner=False
        with self._lock:
            value=self._lookup(key)
            if value:
                return value
            event=self._inflight.get(key)
            if event is None:
                event=threading.Event()
                self._inflight[key]=event
                owner=True
        if not owner:
            event.wait(timeout=10)
            value=self.get(key)
            if value:
                return value
            return factory()
        try:
            value=factory()
            self.set(key,value)
            return value
        finally:
            with self._lock:
                self._inflight.pop(key,None)
                event.set()


def _normalize_question(text):
    return re.sub(r"\s+"," ",(text or "").strip())


def _has_cjk(text):
    return bool(re.search(r"[\u4e00-\u9fff]",text or ""))


def _valid_retrieval_query(text):
    #生成query必须单行、英文、无解释和Markdown，避免污染检索
    q=_normalize_question(text)
    if not q or len(q)>300 or _has_cjk(q):
        return False
    if "\n" in text or "```" in text or q.startswith("#"):
        return False
    if re.match(r"^(here|explanation|translation|query|answer|note)\s*[:：]",q,re.I):
        return False
    return len(re.findall(r"[A-Za-z]",q))>=3


class RetrievalService:
    #唯一检索权威：查询计划、召回、重排、去重和稳定元数据都从这里输出
    def __init__(self,faiss_path=None,model_path=None,index=None,meta=None,model=None):
        self.faiss_path=faiss_path or os.getenv("FAISS_PATH")
        self.model_path=model_path or os.getenv("MODEL_PATH")
        self.index=index
        self.model=model
        self.meta=meta or {}
        self.sources=[]
        self.sections=[]
        self.documents=[]
        self.chunk_ids=[]
        self._bm25=None
        self._section_chunk_ids={}
        self._load_metadata()
        self.candidate_k=int(os.getenv("RETRIEVAL_CANDIDATES","25"))
        self.alpha=float(os.getenv("RETRIEVAL_ALPHA","0.5"))
        self.rerank_mode=os.getenv("RERANK_MODE","conditional").lower()
        self.query_generation_enabled=os.getenv("QUERY_GENERATION_ENABLED","0")=="1"
        self.query_cache=_QueryCache(
            path=os.getenv("QUERY_CACHE_FILE",""),
            negative_ttl=int(os.getenv("QUERY_NEGATIVE_TTL","60"))
        )

    def _load_metadata(self):
        if not self.meta:
            if not self.faiss_path or not os.path.exists(self.faiss_path+".json"):
                raise FileNotFoundError(f"FAISS元数据不存在：{self.faiss_path}.json")
            self.meta=json.load(open(self.faiss_path+".json",encoding="utf-8"))
        self.sources=list(self.meta.get("sources",[]))
        self.sections=list(self.meta.get("sections",[]))
        self.documents=list(self.meta.get("documents",[]))
        stored_ids=list(self.meta.get("chunk_ids",[]))
        if len(stored_ids)==len(self.documents):
            self.chunk_ids=stored_ids
        else:
            self.chunk_ids=[self._stable_chunk_id(i,source,section,text)
                            for i,(source,section,text) in enumerate(zip(self.sources,self.sections,self.documents))]
        for chunk_id,source,section in zip(self.chunk_ids,self.sources,self.sections):
            self._section_chunk_ids.setdefault((source,section),[]).append(chunk_id)

    def _stable_chunk_id(self,index,source,section,text):
        #稳定ID包含文本哈希；相同索引重复构建得到相同ID，文本变化时自动失效
        digest=hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]
        return f"{source}::{section}::{index}::{digest}"

    def _ensure_index(self):
        if self.index is None:
            if not self.faiss_path:
                raise RuntimeError("未配置FAISS_PATH")
            self.index=faiss.read_index(self.faiss_path+".faiss")
        if self.model is None:
            if not self.model_path:
                raise RuntimeError("未配置MODEL_PATH")
            self.model=SentenceTransformer(self.model_path)

    def _ensure_bm25(self):
        if self._bm25 is None:
            tokenized=[self._tokenize(doc) for doc in self.documents]
            self._bm25=BM25Okapi(tokenized)

    def _tokenize(self,text):
        #英文论文检索统一使用拉丁词切分，避免jieba把术语切碎
        return re.findall(r"[a-z0-9]+",(text or "").lower())

    def index_snapshot(self):
        try:
            st=os.stat(self.faiss_path+".faiss")
            return f"{int(st.st_mtime)}-{st.st_size}-{len(self.documents)}"
        except Exception:
            return f"unknown-{len(self.documents)}"

    def get_chunk_ids(self,paper_id):
        return [
            chunk_id for chunk_id,source in zip(self.chunk_ids,self.sources) if source==paper_id
        ]

    def get_section_chunk_ids(self,paper_id,section_name):
        exact=self._section_chunk_ids.get((paper_id,section_name),[])
        if exact:
            return list(exact)
        target=_normalize_question(section_name).lower()
        out=[]
        for (source,section),ids in self._section_chunk_ids.items():
            if source!=paper_id:
                continue
            low=section.lower()
            if target and (target in low or low in target):
                out.extend(ids)
        return out

    def _query_cache_key(self,question):
        provider=os.getenv("QUERY_GENERATION_PROVIDER","deepseek")
        model=os.getenv("QUERY_GENERATION_MODEL","deepseek-flash")
        temperature=os.getenv("QUERY_GENERATION_TEMPERATURE","0")
        max_tokens=os.getenv("QUERY_GENERATION_MAX_TOKENS","120")
        target=os.getenv("QUERY_GENERATION_TARGET_LANG","en")
        norm_version=os.getenv("QUERY_NORMALIZATION_VERSION","v1")
        raw="|".join([norm_version,provider,model,QUERY_PROMPT_HASH,str(temperature),str(max_tokens),target,question])
        return "qp|"+hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _generate_query(self,question):
        reserved=reserve_budget(float(os.getenv("QUERY_GENERATION_BUDGET_RESERVE","0.005")))
        try:
            result=call_deepseek_once(
                [{"role":"system","content":QUERY_PROMPT},{"role":"user","content":question}],
                temperature=0.0,
                max_tokens=int(os.getenv("QUERY_GENERATION_MAX_TOKENS","120")),
                timeout=float(os.getenv("QUERY_GENERATION_TIMEOUT","3")),
                thinking="disabled",
                record_usage=False
            )
            settle_budget(reserved,result)
            return (result["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            release_budget(reserved)
            raise

    def _build_retrieval_query(self,question,search_query=None):
        #返回(实际检索query, 是否缓存命中, 回退原因)，只服务生产召回流程
        normalized=_normalize_question(question)
        if search_query:
            return _normalize_question(search_query),False,""
        if not _has_cjk(normalized) or not self.query_generation_enabled:
            return normalized,False,""
        key=self._query_cache_key(normalized)
        cached=self.query_cache.get(key)
        if cached:
            return cached["query"],True,cached.get("fallback_reason","")
        def create():
            try:
                generated=self._generate_query(normalized)
                if not _valid_retrieval_query(generated):
                    return {"query":normalized,"fallback_reason":"invalid_generated_query"}
                return {"query":generated,"fallback_reason":""}
            except Exception as e:
                logger.warning(f"检索query生成失败，回退原问题：{e}")
                return {"query":normalized,"fallback_reason":"query_generation_failed"}
        value=self.query_cache.get_or_create(key,create)
        return value["query"],False,value.get("fallback_reason","")

    def _normalize_scores(self,scores):
        max_score=max(scores) if len(scores) else 0
        return [score/max_score if max_score>0 else 0 for score in scores]

    def _encode_query(self,query):
        self._ensure_index()
        prompt_name=os.getenv("QUERY_PROMPT_NAME")
        if prompt_name:
            q_vec=self.model.encode([query],prompt_name=prompt_name)[0]
        else:
            q_vec=self.model.encode([query])[0]
        q_vec=q_vec/np.linalg.norm(q_vec)
        return q_vec

    def _vector_search(self,query,k):
        q_vec=self._encode_query(query)
        scores,ids=self.index.search(q_vec.astype("float32")[None,:],k)
        return scores[0],ids[0]

    def _bm25_search(self,query,k):
        self._ensure_bm25()
        tokens=self._tokenize(query)
        scores=self._bm25.get_scores(tokens)
        top=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
        return [scores[i] for i in top],top

    def _recall(self,query,k,alpha=None):
        alpha=self.alpha if alpha is None else alpha
        vec=None
        bm=None
        status=[]
        try:
            vec=self._vector_search(query,k*2)
            status.append("vector")
        except Exception as e:
            logger.warning(f"向量检索失败：{e}")
        try:
            bm=self._bm25_search(query,k*2)
            status.append("bm25")
        except Exception as e:
            logger.warning(f"BM25检索失败：{e}")
        if not status:
            return [],"failed"
        combined={}
        if vec:
            scores,ids=vec
            for score,idx in zip(self._normalize_scores(list(scores)),ids):
                if idx>=0:
                    combined[int(idx)]=combined.get(int(idx),0)+alpha*float(score)
        if bm:
            scores,ids=bm
            for score,idx in zip(self._normalize_scores(scores),ids):
                combined[int(idx)]=combined.get(int(idx),0)+(1-alpha)*float(score)
        top=sorted(combined.items(),key=lambda x:x[1],reverse=True)[:k]
        return top,"+".join(status)

    def recall(self,query,k=50,alpha=None):
        #仅供兼容层和离线路由使用：只做混合召回，不生成query、不重排
        normalized=_normalize_question(query)
        recalled,status=self._recall(normalized,k,alpha)
        items=[self._make_item(idx,score,normalized,"") for idx,score in recalled]
        return items,status

    def _make_item(self,idx,recall_score,retrieval_query,fallback_reason=""):
        return {
            "_index":int(idx),
            "source":self.sources[idx],
            "section":self.sections[idx],
            "text":self.documents[idx],
            "recall_score":round(float(recall_score),6),
            "rerank_score":None,
            "chunk_id":self.chunk_ids[idx],
            "final_rank":0,
            "retrieval_query":retrieval_query,
            "rerank_triggered":False,
            "fallback_reason":fallback_reason,
            "service_version":SERVICE_VERSION,
        }

    def _dedupe(self,items):
        seen=set()
        out=[]
        for item in items:
            key=item["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _should_rerank(self,items,top_k,mode=None):
        mode=mode or self.rerank_mode
        if mode=="none":
            return False
        if mode!="conditional":
            return True
        if len(items)<=1:
            return False
        margin=float(os.getenv("RERANK_TRIGGER_MARGIN","0.1"))
        return (items[0]["recall_score"]-items[min(top_k,len(items))-1]["recall_score"])<margin

    def _execute_search(self,question,retrieval_query,fallback_reason="",candidate_k=None,rerank_mode=None,top_k=5):
        candidate_k=candidate_k or self.candidate_k
        recalled,status=self._recall(retrieval_query,candidate_k)
        items=[self._make_item(idx,score,retrieval_query,fallback_reason) for idx,score in recalled]
        if not items and retrieval_query!=question:
            recalled,status=self._recall(question,candidate_k)
            items=[self._make_item(idx,score,question,fallback_reason) for idx,score in recalled]
            fallback_reason="no_candidates_use_original"
        if not items:
            return []
        mode=(rerank_mode or self.rerank_mode).lower()
        use_rerank=self._should_rerank(items,top_k,mode)
        if use_rerank:
            scope=f"c{len(items)}|{items[0]['chunk_id']}|{retrieval_query}|{question}"
            items=rerank(question,items,top_n=max(top_k,len(items)),snapshot=self.index_snapshot(),cache_scope=scope)
            for item in items:
                item["rerank_triggered"]=True
                item["fallback_reason"]=fallback_reason
                item["retrieval_query"]=retrieval_query
        else:
            items=sorted(items,key=lambda x:x["recall_score"],reverse=True)
        items=self._dedupe(items)[:top_k]
        for rank,item in enumerate(items,1):
            item["final_rank"]=rank
            item["fallback_reason"]=fallback_reason if not item.get("fallback_reason") else item["fallback_reason"]
        return items

    def search(self,question,candidate_k=None,rerank_mode=None,top_k=5,search_query=None):
        retrieval_query,cache_hit,fallback_reason=self._build_retrieval_query(question,search_query)
        return self._execute_search(question,retrieval_query,fallback_reason,candidate_k,rerank_mode,top_k)

    async def search_async(self,question,candidate_k=None,rerank_mode=None,top_k=5,search_query=None):
        return await asyncio.to_thread(
            self.search,question,candidate_k,rerank_mode,top_k,search_query
        )


_SERVICE=None
_SERVICE_LOCK=threading.Lock()

def get_service():
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE=RetrievalService()
    return _SERVICE
