import argparse
import json
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from evaluation.schema import load_questions
from evaluation.metrics import evidence_hit,wilson_ci
from evaluation.retrievers import Retriever
from evaluation.run_eval import build_queries

MODES=["vector","bm25","hybrid","rrf","section","multi"]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",default="evaluation/questions/legacy_88.json")
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--query-field",default="en",choices=["question","en"])
    parser.add_argument("--output",default="evaluation/results/legacy88_modes.json")
    args=parser.parse_args()
    load_dotenv()
    questions=load_questions(args.questions)
    retriever=Retriever(os.getenv("FAISS_PATH"),os.getenv("MODEL_PATH"))
    results={}
    for mode in MODES:
        retriever.mode=mode
        hits=0
        for q in questions:
            queries=build_queries(q,args.query_field)
            if mode=="multi":
                retrieved=retriever.multi_query_search(queries,args.k)
            else:
                retrieved=retriever.search(queries[0],args.k)
            hits+=evidence_hit(retrieved,q.evidence_sentences)
        n=len(questions); lo,hi=wilson_ci(hits,n)
        results[mode]={"n":n,"hits":hits,"rate":hits/n if n else 0.0,"ci95":[lo,hi]}
        print(f"{mode:8s} 证据级@{args.k}={hits/n:.1%}（{hits}/{n}，95%CI {lo:.1%}-{hi:.1%}）")
    base=results["hybrid"]["rate"]
    print("\n相对混合检索的增量：")
    for mode in MODES:
        if mode=="hybrid":
            continue
        print(f"  {mode}: {(results[mode]['rate']-base)*100:+.1f}pp")
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"已保存：{args.output}")

if __name__=="__main__":
    main()
