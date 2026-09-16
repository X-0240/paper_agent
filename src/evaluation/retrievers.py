import json
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())

class Retriever:
    def __init__(self,faiss_path,model_path,mode="hybrid",alpha=0.5,candidates=25):
        self.meta=json.load(open(faiss_path+".json",encoding="utf-8"))
        self.chunks=self.meta["documents"]
        self.sources=self.meta["sources"]
        self.sections=self.meta["sections"]
        self.index=faiss.read_index(faiss_path+".faiss")
        self.model=SentenceTransformer(model_path)
        self.bm25=BM25Okapi([tokenize(c) for c in self.chunks])
        self.mode=mode
        self.alpha=alpha
        self.candidates=candidates

    def _item(self,idx,score):
        return {"idx":int(idx),"score":float(score),"source":self.sources[idx],
                "section":self.sections[idx],"text":self.chunks[idx]}

    def _query_vec(self,query):
        v=self.model.encode([query])[0]
        return (v/np.linalg.norm(v)).astype("float32")

    def vector_search(self,query,k):
        scores,idx=self.index.search(self._query_vec(query)[None,:],k)
        return [self._item(j,s) for j,s in zip(idx[0],scores[0]) if j>=0]

    def bm25_search(self,query,k):
        scores=self.bm25.get_scores(tokenize(query))
        order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
        return [self._item(i,scores[i]) for i in order]

    def hybrid_search(self,query,k,candidates=None,alpha=None):
        #现有加权融合：向量与BM25各自归一化后线性加权
        candidates=candidates or self.candidates
        alpha=self.alpha if alpha is None else alpha
        scores,idx=self.index.search(self._query_vec(query)[None,:],candidates)
        vs=idx[0]; ss=scores[0]
        bm=self.bm25.get_scores(tokenize(query))
        bm_idx=sorted(range(len(bm)),key=lambda i:bm[i],reverse=True)[:candidates]
        comb={}
        if len(ss)>0 and max(ss)>0:
            ss=ss/max(ss)
        for s,j in zip(ss,vs):
            comb[j]=comb.get(j,0.0)+alpha*float(s)
        bm_scores=np.array([bm[j] for j in bm_idx])
        if len(bm_scores)>0 and max(bm_scores)>0:
            bm_scores=bm_scores/max(bm_scores)
        for s,j in zip(bm_scores,bm_idx):
            comb[j]=comb.get(j,0.0)+(1-alpha)*float(s)
        rank=sorted(comb.items(),key=lambda x:x[1],reverse=True)[:k]
        return [self._item(j,s) for j,s in rank]

    def rrf_search(self,query,k,candidates=None,k_rrf=60):
        #RRF：按排名融合，不依赖分数尺度
        candidates=candidates or self.candidates
        vec=self.index.search(self._query_vec(query)[None,:],candidates)
        vs,vi=vec[0][0],vec[1][0]
        bm=self.bm25.get_scores(tokenize(query))
        bm_idx=sorted(range(len(bm)),key=lambda i:bm[i],reverse=True)[:candidates]
        comb={}
        for rank,j in enumerate(vi):
            comb[j]=comb.get(j,0.0)+1.0/(k_rrf+rank+1)
        for rank,j in enumerate(bm_idx):
            comb[j]=comb.get(j,0.0)+1.0/(k_rrf+rank+1)
        rank=sorted(comb.items(),key=lambda x:x[1],reverse=True)[:k]
        return [self._item(j,s) for j,s in rank]

    def section_aggregated_search(self,query,k,candidates=None,alpha=None,title_bonus=0.15):
        #先按论文+章节聚合，再选每章最高分片段，缓解同章证据被挤出top5
        candidates=candidates or self.candidates
        items=self.hybrid_search(query,candidates,candidates,alpha)
        q_tokens=set(tokenize(query))
        grouped={}
        for it in items:
            key=(it["source"],it["section"])
            title_tokens=set(tokenize(it["section"]))
            overlap=len(q_tokens&title_tokens)/max(1,len(q_tokens))
            score=it["score"]+title_bonus*overlap
            if key not in grouped or score>grouped[key]["score"]:
                grouped[key]={**it,"score":score}
        return sorted(grouped.values(),key=lambda x:x["score"],reverse=True)[:k]

    def multi_query_search(self,queries,k,candidates=None,alpha=None,k_rrf=60):
        #多query：每个query做混合召回，再用RRF融合
        candidates=candidates or self.candidates
        comb={}
        for qi,query in enumerate(queries):
            items=self.hybrid_search(query,candidates,candidates,alpha)
            for rank,it in enumerate(items):
                j=it["idx"]
                comb[j]=comb.get(j,0.0)+1.0/(k_rrf+rank+1)
        rank=sorted(comb.items(),key=lambda x:x[1],reverse=True)[:k]
        return [self._item(j,s) for j,s in rank]

    def search(self,query,k=5):
        mode=self.mode
        if mode=="vector":
            return self.vector_search(query,k)
        if mode=="bm25":
            return self.bm25_search(query,k)
        if mode=="rrf":
            return self.rrf_search(query,k)
        if mode=="section":
            return self.section_aggregated_search(query,k)
        return self.hybrid_search(query,k)
