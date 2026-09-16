import argparse
import json
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from evaluation.schema import load_questions
from evaluation.metrics import evidence_hit,wilson_ci
from evaluation.retrievers import Retriever

def eval_file(path,retriever,k):
    questions=load_questions(path)
    hits=0
    rows=[]
    for q in questions:
        retrieved=retriever.search(q.question,k=k)
        hit=evidence_hit(retrieved,q.evidence_sentences)
        hits+=hit
        rows.append({"question_id":q.question_id,"question":q.question,"hit":hit})
    n=len(questions)
    lo,hi=wilson_ci(hits,n)
    return {"path":path,"n":n,"hits":hits,"rate":hits/n if n else 0.0,"ci95":[lo,hi],"rows":rows}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--forward",default="evaluation/questions/forward_40.json")
    parser.add_argument("--reverse",default="evaluation/questions/reverse_42.json")
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--output",default="evaluation/results/leakage_ab.json")
    args=parser.parse_args()
    load_dotenv()
    retriever=Retriever(os.getenv("FAISS_PATH"),os.getenv("MODEL_PATH"),mode="hybrid")
    forward=eval_file(args.forward,retriever,args.k)
    reverse=eval_file(args.reverse,retriever,args.k)
    gap=reverse["rate"]-forward["rate"]
    result={"k":args.k,"forward":forward,"reverse":reverse,"gap_pp":gap*100,
            "need_fix":gap*100>10}
    print(f"正向题 证据级@{args.k}={forward['rate']:.1%}（{forward['hits']}/{forward['n']}，95%CI {forward['ci95'][0]:.1%}-{forward['ci95'][1]:.1%}）")
    print(f"反向题 证据级@{args.k}={reverse['rate']:.1%}（{reverse['hits']}/{reverse['n']}，95%CI {reverse['ci95'][0]:.1%}-{reverse['ci95'][1]:.1%}）")
    print(f"泄露gap={gap*100:+.1f}pp，是否需要先修出题流程：{result['need_fix']}")
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(f"已保存：{args.output}")

if __name__=="__main__":
    main()
