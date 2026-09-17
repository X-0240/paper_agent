import re

from retrieval_service import QueryPlan,RetrievalService


def tokenize(text):
    return re.findall(r"[a-z0-9]+",text.lower())


class Retriever:
    #评测适配器：底层召回和模型全部委托RetrievalService，只保留实验用融合模式
    def __init__(self,faiss_path,model_path,mode="hybrid",alpha=0.5,candidates=25):
        self.service=RetrievalService(faiss_path,model_path)
        self.meta=self.service.meta
        self.chunks=self.service.documents
        self.sources=self.service.sources
        self.sections=self.service.sections
        self.mode=mode
        self.alpha=alpha
        self.candidates=candidates

    def _make_items(self,query,scores,ids,offset=0):
        plan=QueryPlan(query,query,"original")
        return [self.service._make_item(int(idx),float(score),plan) for idx,score in zip(ids,scores) if int(idx)>=0]

    def _query_vec(self,query):
        return self.service._encode_query(query)

    def vector_search(self,query,k):
        scores,ids=self.service._vector_search(query,k)
        return self._make_items(query,scores,ids)

    def bm25_search(self,query,k):
        scores,ids=self.service._bm25_search(query,k)
        return self._make_items(query,scores,ids)

    def hybrid_search(self,query,k,candidates=None,alpha=None):
        #hybrid底层完全复用Service，不在评测侧重新实现向量或BM25
        items,status=self.service.recall(query,candidates or self.candidates,alpha)
        return items[:k]

    def rrf_search(self,query,k,candidates=None,k_rrf=60):
        candidates=candidates or self.candidates
        vec=self.vector_search(query,candidates)
        bm=self.bm25_search(query,candidates)
        comb={}
        for rank,item in enumerate(vec):
            key=item["chunk_id"]
            comb[key]={"item":item,"score":comb.get(key,{}).get("score",0)+1.0/(k_rrf+rank+1)}
        for rank,item in enumerate(bm):
            key=item["chunk_id"]
            comb[key]={"item":item,"score":comb.get(key,{}).get("score",0)+1.0/(k_rrf+rank+1)}
        ranked=sorted(comb.values(),key=lambda x:x["score"],reverse=True)[:k]
        return [x["item"] for x in ranked]

    def section_aggregated_search(self,query,k,candidates=None,alpha=None,title_bonus=0.15):
        candidates=candidates or self.candidates
        items=self.hybrid_search(query,candidates,candidates,alpha)
        q_tokens=set(tokenize(query))
        grouped={}
        for item in items:
            key=(item["source"],item["section"])
            title_tokens=set(tokenize(item["section"]))
            overlap=len(q_tokens&title_tokens)/max(1,len(q_tokens))
            score=item["recall_score"]+title_bonus*overlap
            if key not in grouped or score>grouped[key][0]:
                grouped[key]=(score,item)
        return [item for _,item in sorted(grouped.values(),key=lambda x:x[0],reverse=True)[:k]]

    def multi_query_search(self,queries,k,candidates=None,alpha=None,k_rrf=60):
        candidates=candidates or self.candidates
        comb={}
        for query in queries:
            items=self.hybrid_search(query,candidates,candidates,alpha)
            for rank,item in enumerate(items):
                key=item["chunk_id"]
                if key not in comb:
                    comb[key]={"item":item,"score":0.0}
                comb[key]["score"]+=1.0/(k_rrf+rank+1)
        ranked=sorted(comb.values(),key=lambda x:x["score"],reverse=True)[:k]
        return [x["item"] for x in ranked]

    def search(self,query,k=5):
        if self.mode=="vector":
            return self.vector_search(query,k)
        if self.mode=="bm25":
            return self.bm25_search(query,k)
        if self.mode=="rrf":
            return self.rrf_search(query,k)
        if self.mode=="section":
            return self.section_aggregated_search(query,k)
        return self.hybrid_search(query,k)
