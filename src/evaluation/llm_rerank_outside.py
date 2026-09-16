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
from evaluation.llm_rerank import rerank_one

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",default="evaluation/questions/legacy_88.json")
    parser.add_argument("--diagnosis",default="evaluation/results/rank_diagnosis.json")
    parser.add_argument("--candidates",type=int,default=50)
    parser.add_argument("--max-chars",type=int,default=1500)
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--output",default="evaluation/results/llm_rerank_outside.json")
    args=parser.parse_args()
    load_dotenv()
    diagnosis=json.load(open(args.diagnosis,encoding="utf-8"))
    outside={r["question_id"] for r in diagnosis["rows"] if r["rank"] is None or r["rank"]>20}
    questions=[q for q in load_questions(args.questions) if q.question_id in outside]
    retriever=Retriever(os.getenv("FAISS_PATH"),os.getenv("MODEL_PATH"),mode="hybrid")
    recovered=0
    rows=[]
    for q in questions:
        queries=build_queries(q,"en")
        candidates=retriever.hybrid_search(queries[0],args.candidates)
        base=candidates[:args.k]
        base_hit=evidence_hit(base,q.evidence_sentences)
        reranked=rerank_one(q.question,candidates,args.k,args.max_chars)
        rerank_hit=evidence_hit(reranked,q.evidence_sentences)
        recovered+=rerank_hit
        rows.append({"question_id":q.question_id,"base_hit":base_hit,"rerank_hit":rerank_hit,
                     "base_sources":[{"source":r["source"],"section":r["section"]} for r in base],
                     "rerank_sources":[{"source":r["source"],"section":r["section"]} for r in reranked]})
    n=len(questions)
    print(f"top20外题目={n}，基线命中={sum(1 for r in rows if r['base_hit'])}，扩池重排救回={recovered}，救回率={recovered/max(1,n):.1%}")
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump({"n":n,"recovered":recovered,"rows":rows},f,ensure_ascii=False,indent=2)
    print(f"已保存：{args.output}")

if __name__=="__main__":
    main()
