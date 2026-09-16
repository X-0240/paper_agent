import argparse
import json
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from evaluation.schema import load_questions
from evaluation.metrics import evidence_hit,norm
from evaluation.retrievers import Retriever
from evaluation.run_eval import build_queries

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",default="evaluation/questions/legacy_88.json")
    parser.add_argument("--k",type=int,default=100)
    parser.add_argument("--query-field",default="en",choices=["question","en"])
    parser.add_argument("--output",default="evaluation/results/rank_diagnosis.json")
    args=parser.parse_args()
    load_dotenv()
    questions=load_questions(args.questions)
    retriever=Retriever(os.getenv("FAISS_PATH"),os.getenv("MODEL_PATH"),mode="hybrid")
    buckets={"top5":0,"6-10":0,"11-20":0,"21-100":0,"not_in_top100":0}
    rows=[]
    for q in questions:
        queries=build_queries(q,args.query_field)
        items=retriever.hybrid_search(queries[0],args.k)
        targets=[norm(x) for x in q.evidence_sentences if norm(x)]
        rank=None
        for i,item in enumerate(items,1):
            text=norm(item["text"])
            if any(t and t in text for t in targets):
                rank=i; break
        if rank is None:
            buckets["not_in_top100"]+=1
        elif rank<=5:
            buckets["top5"]+=1
        elif rank<=10:
            buckets["6-10"]+=1
        elif rank<=20:
            buckets["11-20"]+=1
        else:
            buckets["21-100"]+=1
        rows.append({"question_id":q.question_id,"question":q.question,"rank":rank})
    print("证据在混合检索中的排名分布：")
    for k,v in buckets.items():
        print(f"  {k}: {v}")
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump({"buckets":buckets,"rows":rows},f,ensure_ascii=False,indent=2)
    print(f"已保存：{args.output}")

if __name__=="__main__":
    main()
