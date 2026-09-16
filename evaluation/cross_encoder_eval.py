import argparse
import json
import os
import sys
import time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from evaluation.schema import load_questions
from evaluation.metrics import evidence_hit,wilson_ci
from evaluation.retrievers import Retriever
from evaluation.run_eval import build_queries

class CrossEncoderReranker:
    def __init__(self,model_path,max_chars=1500,batch_size=16):
        self.model=CrossEncoder(model_path)
        self.max_chars=max_chars
        self.batch_size=batch_size

    def rerank(self,query,candidates,top_n):
        #Cross-Encoder逐对打分，全文片段参与，避免证据被截断
        pairs=[[query,c["text"].replace("\n"," ")[:self.max_chars]] for c in candidates]
        scores=self.model.predict(pairs,batch_size=self.batch_size)
        order=sorted(zip(candidates,scores),key=lambda x:float(x[1]),reverse=True)
        return [c for c,_ in order[:top_n]]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",default="evaluation/questions/legacy_88.json")
    parser.add_argument("--model",default=r"D:\Users\徐海生\Documents\agent项目\bge-reranker-v2-m3")
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--candidates",type=int,default=25)
    parser.add_argument("--max-chars",type=int,default=1500)
    parser.add_argument("--query-field",default="en",choices=["question","en"])
    parser.add_argument("--rerank-query-field",default=None,choices=["question","en"])
    parser.add_argument("--only-miss",action="store_true")
    parser.add_argument("--output",default="evaluation/results/cross_encoder.json")
    args=parser.parse_args()
    load_dotenv()
    questions=load_questions(args.questions)
    retriever=Retriever(os.getenv("FAISS_PATH"),os.getenv("MODEL_PATH"),mode="hybrid")
    reranker=CrossEncoderReranker(args.model,args.max_chars)
    base_hits=0
    rerank_hits=0
    pending=0
    total_latency=0.0
    rows=[]
    for q in questions:
        queries=build_queries(q,args.query_field)
        rerank_field=args.rerank_query_field or args.query_field
        rerank_query=q.query_en if rerank_field=="en" and q.query_en else q.question
        candidates=retriever.hybrid_search(queries[0],args.candidates)
        base=candidates[:args.k]
        base_hit=evidence_hit(base,q.evidence_sentences)
        base_hits+=base_hit
        if args.only_miss and base_hit:
            rerank_hits+=1
            rows.append({"question_id":q.question_id,"base_hit":True,"rerank_hit":True})
            continue
        pending+=1
        t0=time.time()
        reranked=reranker.rerank(rerank_query,candidates,args.k)
        total_latency+=time.time()-t0
        rerank_hit=evidence_hit(reranked,q.evidence_sentences)
        rerank_hits+=rerank_hit
        rows.append({"question_id":q.question_id,"base_hit":base_hit,"rerank_hit":rerank_hit,
                     "base_sources":[{"source":r["source"],"section":r["section"]} for r in base],
                     "rerank_sources":[{"source":r["source"],"section":r["section"]} for r in reranked]})
    n=len(questions)
    print(f"基线 hybrid@{args.k}={base_hits/n:.1%}（{base_hits}/{n}）")
    print(f"CrossEncoder@{args.k}={rerank_hits/n:.1%}（{rerank_hits}/{n}）")
    print(f"增量：{(rerank_hits-base_hits)/n*100:+.1f}pp；实际重排{pending}题，平均{total_latency/max(1,pending)*1000:.0f}ms/题")
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump({"n":n,"base_hits":base_hits,"rerank_hits":rerank_hits,
                   "avg_latency_ms":total_latency/max(1,pending)*1000,"rows":rows},f,ensure_ascii=False,indent=2)
    print(f"已保存：{args.output}")

if __name__=="__main__":
    main()
